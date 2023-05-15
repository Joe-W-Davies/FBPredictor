from datetime import datetime
import argparse
import yaml
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from scipy.stats import uniform
import seaborn as sns
import matplotlib.pyplot as plt
import shap

from TrainUtils import (change_dtypes, 
    create_time_features, 
    encode_features, 
    impute_nulls, 
    MissingDict, 
    add_lags, 
    add_rolling_vars, 
    add_expanded_vars
)
from BorutaShap import BorutaShap


def main(options):
    #unpack config
    with open(options.config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)
        nominal_vars = config['nominal_vars']
        train_vars_to_roll = config['train_vars_to_roll']
        team_mapping = MissingDict(**config['team_mapping'])
        datasets = config['datasets']
        filters = config['filters']
        n_days_lag = config['n_days_lag']
        n_days_rolling = config['n_days_rolling']

    dfs = []
    for df_name in datasets:
        dfs.append( pd.read_csv(f"data/{df_name}", index_col=0) )
    df = pd.concat(dfs)

    
    #data cleaning and filtering
    df = df.query(filters)
    df = df.dropna(how='all')
    df = change_dtypes(df)
    df = impute_nulls(df, train_vars_to_roll, impute=True)
    df = create_time_features(df)
    current_date = datetime.today().strftime('%Y-%m-%d')
    df = df[df['date']< f'{current_date}']
    print(f'after cleaning, df has: {df.duplicated(keep=False).sum()} duplicate rows')

    #add target column
    df['y_true'] = (df['result']=='W').astype('int8')

    
    running_features = set(nominal_vars)

    #add lagged features for last 5 days
    df, running_features = add_lags(df, n_days_lag, running_features)

    #add rolled mean and median features for the previous n_days
    for day in n_days_rolling:
        df, running_features = add_rolling_vars(
            df, 
            day, 
            running_features, 
            train_vars_to_roll
        )
     
    
    #add expanded mean features
    df, running_features = add_expanded_vars(
        df, 
        running_features, 
        train_vars_to_roll
    )
    running_features = list(running_features)

    
    print(df[running_features+['date','y_true']].info())    
    df = df[running_features+['date','y_true']].dropna(how='any')
    df = df.sort_values(['date'])
    df.index = range(df.shape[0])
    print(df.shape)    


    #merge in opponent info
    df['opponent'] = df['opponent'].map(team_mapping)
    df = df.merge(
        df[running_features+['date']], 
        left_on=["date", "team"], 
        right_on=["date", "opponent"], 
        suffixes=("","_opp"),
        how='inner'
        )

    df = encode_features(df)

    #FIXME check why we dont have exact duplicates after merging
    df = df.drop_duplicates() 
    
    #train/test split
    final_train_vars = running_features + [v+'_opp' for v in running_features if v not in nominal_vars]
    print(f'training with {len(final_train_vars)} variables')
    
    x_train = df[df['date']<'2022-08-01'][final_train_vars] 
    y_train = df[df['date']<'2022-08-01']['y_true'] 
    x_test  = df[df['date']>'2022-08-01'][final_train_vars] 
    y_test  = df[df['date']>'2022-08-01']['y_true']


    if options.hp_opt:
        # Define the hyperparameter ranges
        params = {
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': range(3, 10),
            'n_estimators': range(50, 1000, 50)
        }
        
        
        # Set up a time series cross-validation 
        ts_cv = TimeSeriesSplit(n_splits=3)
        
        clf = xgb.XGBClassifier()
        
        grid_search = RandomizedSearchCV(
            clf,
            params,
            cv=ts_cv,
            scoring='roc_auc',
            n_iter=150,
            verbose=3
        )
        
        grid_search.fit(x_train, y_train)
        train_params = grid_search.best_params_
        print(f'best parameters: {best_params}')
        clf = xgb.XGBClassifier(**train_params)
    
    #train GBDT  
    else:
        #chose reasonable parameters and train with them
        train_params = {'n_estimators':150, 'eta':0.05, 'max_depth':4}
        clf = xgb.XGBClassifier(
            objective='binary:logistic', 
            **train_params
            )

    #Train RF
    #clf = RandomForestClassifier()
    if options.feature_select:
        final_train_vars = BorutaShap(x_train, 
            y_train, 
            final_train_vars, 
            np.ones_like(y_train), 
            i_iters=3, 
            tolerance=0.1, 
            max_vars_removed=int(0.8*len(running_features)), 
            n_trainings=20, 
            train_params=train_params
            )()
        x_train = x_train[final_train_vars]
        x_test = x_test[final_train_vars]


    clf.fit(x_train,y_train)

    
    #predict probs
    y_pred_train = clf.predict_proba(x_train)[:,1:].ravel() 
    y_pred_test = clf.predict_proba(x_test)[:,1:].ravel()
    
    baseline_roc = roc_auc_score(y_test, x_test['venue'])
    print(f'baseline roc_score {baseline_roc}')
    print(f'train roc_score: {roc_auc_score(y_train, y_pred_train)}')
    print(f'test roc_score: {roc_auc_score(y_test, y_pred_test)}')
    
    print()
    
    #predict classes
    baseline_acc = accuracy_score(y_test, x_test['venue'])
    print(f'baseline accuracy {baseline_acc}')
    y_pred_train_class = clf.predict(x_train) 
    y_pred_test_class = clf.predict(x_test)
    print(f'train accuracy: {accuracy_score(y_train, y_pred_train_class)}')
    print(f'test accuracy: {accuracy_score(y_test, y_pred_test_class)}')

    #save model
    bstr = clf.get_booster()
    bstr.save_model('models/model.json')
    print ("Saved classifier as: models/model.json")


    #make some plots

    #ROCs
    loss_eff_train, win_eff_train, _ = roc_curve(y_train, y_pred_train)
    loss_eff_test, win_eff_test, _ = roc_curve(y_test, y_pred_test)
    
    fig = plt.figure()
    axes = fig.gca()

    axes.plot(loss_eff_train, win_eff_train, color='red', label='Train set')
    axes.plot(loss_eff_test, win_eff_test, color='royalblue', label='Test set')
    
    axes.plot(
        np.linspace(0,1,100),
        np.linspace(0,1,100), 
        linestyle="--", 
        color='black', 
        zorder=0, 
        label="Random classifier"
    )

    axes.set_xlabel('False positive rate', ha='right', x=1, size=13)
    axes.set_xlim((0,1))
    axes.set_ylabel('True positive rate', ha='right', y=1, size=13)
    axes.set_ylim((0,1))
    axes.legend(bbox_to_anchor=(0.97,0.28))
    axes.grid(True, 'major', linestyle='solid', color='grey', alpha=0.5)
    fig.savefig('plots/ROC_curve.pdf')
    plt.close() 
    
    #confusion matrix
    trues_preds_test = pd.DataFrame({'actual':y_test, 'prediction':y_pred_test_class})
    c_matrix = pd.crosstab(
        index=trues_preds_test['actual'], 
        columns=trues_preds_test['prediction'], 
        normalize='columns'
    )
    fig = sns.heatmap(c_matrix, 
        vmin=0, 
        vmax=1, 
        annot=True, 
        cmap='viridis'
    )
    fig.get_figure().savefig('plots/confusion_matrix.pdf')
    plt.close() 


    #shap plots
    explainer = shap.Explainer(clf)
    shap_values = explainer(pd.DataFrame(x_train, columns=final_train_vars))
    vals = np.abs(shap_values.values).mean(0)
    n_importance = {key:value for (key,value) in zip(final_train_vars,vals)}
    n_imp_sorted = {k: v for k, v in sorted(n_importance.items(), key=lambda item: item[1])}

    plt.rcParams.update({'text.usetex':'false'})
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig('plots/shapley_beeswarm.pdf', bbox_inches="tight")
    print(' --> Saved plot: plots/shapley_beeswarm.pdf')
    plt.close()

    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig('plots/shapley_bar_chart.pdf', bbox_inches="tight")
    print(' --> Saved plot: plots/shapley_bar_chart.pdf')
    plt.close() 
   

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', action='store', required=True)
    opt_args = parser.add_argument_group('Optional Arguments')
    opt_args.add_argument('-f','--feature_select', action='store_true',default=False)
    opt_args.add_argument('-o','--hp_opt', action='store_true',default=False)
    options=parser.parse_args()
    main(options)
