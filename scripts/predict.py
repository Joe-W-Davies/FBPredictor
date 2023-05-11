import argparse
import yaml
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

from TrainUtils import change_dtypes, create_time_features, impute_nulls, MissingDict


def main(options):
    #unpack config
    with open(options.config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)
        nominal_vars = config['nominal_vars']
        train_vars_to_roll = config['train_vars_to_roll']
        team_mapping = MissingDict(**config['team_mapping'])
        datasets = config['datasets']
        filters = config['filters']
        n_days = config['n_days']

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
    print(f'after cleaning, df has: {df.duplicated(keep=False).sum()} duplicate rows')

    #add target column
    df['y_true'] = (df['result']=='W').astype('int8')

    
    full_feature_set = set()

    #add lagged features for last 5 days

    lags = [x+1 for x in range(n_days)]
    lagged_dfs = []
    for group in df.groupby(['team','year'], group_keys=False):
        gr = group[-1].sort_values(['date'])
        for lag in lags:
            gr[f'lag_{lag}'] = gr['y_true'].shift(lag)
            full_feature_set.add(f'lag_{lag}')
        lagged_dfs.append(gr)
    df = pd.concat(lagged_dfs, ignore_index=True) 
    

    #add rolled mean features 
    rolled_dfs = []
    for group in df.groupby(['team','year'], group_keys=False):
        gr = group[-1].sort_values(['date'])
        for var in train_vars_to_roll:
            #gr[rolled_train_vars[var]] = gr[var].rolling(n_days, closed='left', win_type='exponential').mean(tau=0.5) #closed='left' does not work with exp window
            modified_var = var+'_rolling_avg'
            gr[modified_var] = gr[var].rolling(n_days, closed='left').mean()
            #gr[modified_var] = gr[var].rolling(n_days).mean()
            full_feature_set.add(modified_var)
        rolled_dfs.append(gr)
    df = pd.concat(rolled_dfs, ignore_index=True)
    
    
    
    ##add rolled median features 
    rolled_dfs = []
    for group in df.groupby(['team','year'], group_keys=False):
        gr = group[-1].sort_values(['date'])
        for var in train_vars_to_roll:
            modified_var = var+'_rolling_med'
            gr[modified_var] = gr[var].rolling(n_days, closed="left").median() #probs a way to do this without a dict but oh well
            full_feature_set.add(modified_var)
        rolled_dfs.append(gr)
    df = pd.concat(rolled_dfs, ignore_index=True)
    
    
    
    #add expanded mean features
    expanded_dfs = []
    for group in df.groupby(['team','year'], group_keys=False):
        gr = group[-1].sort_values(['date'])
        for var in train_vars_to_roll:
            modified_var = var+'_expanded'
            gr[modified_var] = gr[var].expanding().mean()
            full_feature_set.add(modified_var)
        expanded_dfs.append(gr)
    df = pd.concat(expanded_dfs, ignore_index=True)
    
    
    
    #drop any row with a null 
    df = df.dropna(how='any')
    df = df.sort_values(['date'])
    df.index = range(df.shape[0])
    





    #FIXME check i dropped repeated info e.g. date
    #FIXME check why we dont have exact duplicates after merging
    #FIXME check if it is actually doing what you expect!! i.e. check it against actual stats on the website to see if the games matched up properly
    #FIXME: make encoding the same! for all encoded features!!! e.g. dats, years - had to drop them atm. Add them back into features
    print(df.shape)
    df['opponent'] = df['opponent'].map(team_mapping)
    df = df.merge(df[list(full_feature_set)+['date','opponent','venue']], 
                  left_on=["date", "team"], 
                  right_on=["date", "opponent"], 
                  suffixes=("","_opp"),
                  how='inner'
                  )
    # HAVE TO BE CAREFUL - pandas might encode opponent and team differently in two different columns. So e.g. Arsenal = 0 in team but =10 in opponent! so kpeeing as strings for line above, then converting for line below
    #df = encode_teams(df, team_mapping)
    print(df.shape)
    df = df.drop_duplicates() 
    print(df.shape)
    
    #train/test split
    final_train_vars = list(full_feature_set) + [v+'_opp' for v in list(full_feature_set)]
    print(f'training with {len(final_train_vars)} variables')
    
    x_train = df[df['date']<'2022-08-01'][final_train_vars] 
    y_train = df[df['date']<'2022-08-01']['y_true'] 
    
    x_test  = df[df['date']>'2022-08-01'][final_train_vars] 
    y_test  = df[df['date']>'2022-08-01']['y_true']
    
    #train GBDT  
    clf = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, 
                            eta=0.05, max_depth=3)

    #Train RF
    #clf = RandomForestClassifier()
    #                                 

    clf.fit(x_train,y_train)
    
    #predict probs and classes (argmax)
    y_pred_train = clf.predict_proba(x_train)[:,1:].ravel() 
    y_pred_train_class = clf.predict(x_train) 
    y_pred_test = clf.predict_proba(x_test)[:,1:].ravel()
    y_pred_test_class = clf.predict(x_test)
    
    #baseline_roc = roc_auc_score(y_test, x_test['venue'])
    #print(f'baseline roc_score {baseline_roc}')
    print(f'train roc_score: {roc_auc_score(y_train, y_pred_train)}')
    print(f'test roc_score: {roc_auc_score(y_test, y_pred_test)}')
    
    print()
    
    #baseline_acc = accuracy_score(y_test, x_test['venue'], )
    #print(f'baseline accuracy {baseline_acc}')
    print(f'train accuracy: {accuracy_score(y_train, y_pred_train_class)}')
    print(f'test accuracy: {accuracy_score(y_test, y_pred_test_class)}')
    
    
    




    #ROCs
    
    loss_eff_train, win_eff_train, _ = roc_curve(y_train, y_pred_train)
    loss_eff_test, win_eff_test, _ = roc_curve(y_test, y_pred_test)
    
    fig = plt.figure(1)
    axes = fig.gca()
    axes.plot(loss_eff_train, win_eff_train, color='red', label='Train set')
    axes.plot(loss_eff_test, win_eff_test, color='royalblue', label='Test set')
    
    axes.plot(np.linspace(0,1,100),np.linspace(0,1,100), linestyle="--", color='black', zorder=0, label="Random classifier")
    axes.set_xlabel('False positive rate', ha='right', x=1, size=13)
    axes.set_xlim((0,1))
    axes.set_ylabel('True positive rate', ha='right', y=1, size=13)
    axes.set_ylim((0,1))
    axes.legend(bbox_to_anchor=(0.97,0.28))
    axes.grid(True, 'major', linestyle='solid', color='grey', alpha=0.5)
    fig.savefig('plots/ROC_curve.pdf')
    
    #confusion matrix
    trues_preds_test = pd.DataFrame({'actual':y_test, 'prediction':y_pred_test_class})
    c_matrix = pd.crosstab(index=trues_preds_test['actual'], columns=trues_preds_test['prediction'], normalize='columns')
    fig = sns.heatmap(c_matrix, vmin=0, vmax=1, annot=True, cmap='viridis')
    fig.get_figure().savefig('plots/confusion_matrix.pdf')
   

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', action='store', required=True)
    options=parser.parse_args()
    main(options)
