from datetime import datetime
import argparse
import yaml
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from scipy.stats import uniform

from TrainUtils import (
    change_dtypes, 
    create_time_features, 
    encode_features, 
    impute_nulls, 
    MissingDict, 
    add_lags, 
    add_odds,
    add_rolling_vars, 
    add_expanded_vars
)

from PlotUtils import (
    plot_roc,
    plot_confusion_matrix,
    plot_output_score,
    plot_shaps,
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
        dfs.append( pd.read_csv(f"data/leagues/{df_name}", index_col=0) )
    df = pd.concat(dfs)

    
    #data cleaning and filtering
    df = df.query(filters)
    df = df.dropna(how='all')
    df = change_dtypes(df)
    df = impute_nulls(df, [v for v in train_vars_to_roll if v in df.columns], impute=True)
    df = create_time_features(df)
    current_date = datetime.today().strftime('%Y-%m-%d')
    df = df[df['date']< f'{current_date}']
    print(f'after cleaning, df has: {df.duplicated(keep=False).sum()} duplicate rows')

    #add target column
    df['y_true'] = (df['result']=='W').astype('int8')

    #add features
    running_features = set()


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

    
    df = df[nominal_vars+running_features+['date','y_true']].dropna(how='any')
    

    #merge in opponent info
    df['opponent'] = df['opponent'].map(team_mapping)
    df = df.merge(
        df[nominal_vars+running_features+['date']], 
        left_on=["date", "team"], 
        right_on=["date", "opponent"], 
        suffixes=("","_opp"),
        how='inner'
        )

    #add odds (must be done after above else you lose opp info)
    if options.add_odds:
        df, home_odds, away_odds, draw_odds = add_odds(df, team_mapping)
        nominal_vars = nominal_vars + home_odds + away_odds + draw_odds

    df = encode_features(df)
    
    #train/test split
    final_train_vars = nominal_vars + running_features + [v+'_opp' for v in running_features if v not in nominal_vars]

    print(f'training with {len(final_train_vars)} variables')
    
    x_train = df[df['date']<'2022-08-01'][final_train_vars] 
    y_train = df[df['date']<'2022-08-01']['y_true'] 
    x_test  = df[df['date']>'2022-08-01'][final_train_vars] 
    y_test  = df[df['date']>'2022-08-01']['y_true']


    if options.hp_opt:

        params = {
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': range(3, 10),
            'n_estimators': range(50, 1000, 50),
            'lambda': uniform(0.01, 10),
        }
        
        # Set up a time series cross-validation 
        ts_cv = TimeSeriesSplit(n_splits=3)
        clf = xgb.XGBClassifier(objective='binary:logistic')
        
        grid_search = RandomizedSearchCV(
            clf,
            params,
            cv=ts_cv,
            scoring='roc_auc',
            n_iter=500,
            verbose=3
        )
        
        grid_search.fit(x_train, y_train)
        train_params = grid_search.best_params_
        print(f'best parameters: {train_params}')
        clf = xgb.XGBClassifier(**train_params)
    
    else:
        #chose reasonable parameters and train with them
        train_params = {'n_estimators':100, 'eta':0.05, 'max_depth':4}
        clf = xgb.XGBClassifier(
            objective='binary:logistic', 
            **train_params
        )

    if options.feature_select:
        final_train_vars = BorutaShap(
            x_train, 
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
    if options.save_model:
        bstr = clf.get_booster()
        bstr.save_model('models/model.json')
        print ("Saved classifier as: models/model.json")

    #make some plots
    plot_roc(clf, y_train, y_pred_train, y_test, y_pred_test)
    plot_confusion_matrix(y_test, y_pred_test_class)
    plot_output_score(y_test, y_pred_test)
    plot_shaps(clf, x_test, final_train_vars)
   

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', action='store', required=True)
    opt_args = parser.add_argument_group('Optional Arguments')
    opt_args.add_argument('-f','--feature_select', action='store_true',default=False)
    opt_args.add_argument('-o','--hp_opt', action='store_true',default=False)
    opt_args.add_argument('-a','--add_odds', action='store_true',default=False)
    opt_args.add_argument('-s','--save_model', action='store_true',default=False)
    options=parser.parse_args()
    main(options)
