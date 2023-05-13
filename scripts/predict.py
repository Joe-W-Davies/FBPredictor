import pandas as pd
import numpy as np
import time
import argparse
import yaml
from datetime import datetime
import xgboost as xgb

from TrainUtils import change_dtypes, create_time_features, encode_features, impute_nulls, MissingDict, add_lags, add_rolling_vars, add_expanded_vars

#scrape 
def main(options):

    with open(options.config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)
        nominal_vars = config['nominal_vars']
        train_vars_to_roll = config['train_vars_to_roll']
        team_mapping = MissingDict(**config['team_mapping'])
        datasets = config['datasets']
        n_days = config['n_days']

    dfs = []
    for df_name in datasets:
        dfs.append( pd.read_csv(f"data/{df_name}", index_col=0) )
    df = pd.concat(dfs)
    
    #data cleaning and filtering
    df = df.query(f"comp=='{options.league}'")
    df = change_dtypes(df)
    df = create_time_features(df)

    #add target column
    df['y_true'] = (df['result']=='W').astype('int8')
    #FIXME: probs need to impute NA's as well since we did this in the training, but might have to be careful with accidentally doing this for unplayed games


    running_features = set(nominal_vars)

    #add lagged features for last 5 days
    df, running_features = add_lags(df, n_days, running_features)

    #add rolled mean and median features for the previous n_days
    df, running_features = add_rolling_vars(df, n_days, running_features, train_vars_to_roll)
    
    #add expanded mean features
    df, running_features = add_expanded_vars(df, running_features, train_vars_to_roll)
    running_features = list(running_features)
    
    
    #drop any row with a null 
    df = df.sort_values(['date'])
    df.index = range(df.shape[0])
    


    #FIXME check if it is actually doing what you expect!! i.e. check it against actual stats on the website to see if the games matched up properly
    df['opponent'] = df['opponent'].map(team_mapping)
    df = df.merge(df[running_features+['date']], 
                  left_on=["date", "team"], 
                  right_on=["date", "opponent"], 
                  suffixes=("","_opp"),
                  how='inner'
                  )

    df = encode_features(df)

    #FIXME check why we dont have exact duplicates after merging
    df = df.drop_duplicates() 
    
    #train/test split

    #only predict most recent match
    current_date = datetime.today().strftime('%Y-%m-%d')
    df = df[df['date']>= f'{current_date}']

    #load model
    clf = xgb.Booster()
    clf.load_model(f'{options.model}')
    print(f'loaded model from: {options.model}')
    final_train_vars = clf.feature_names

 
    #predict outome
    for team in df['team'].unique():
        df_team = df.query(f"team=={team}")
        df_team = df_team.sort_values('date').head(1)
        x_test  = df_team[final_train_vars] 
        d_test = xgb.DMatrix(x_test, feature_names=final_train_vars)
        y_pred_train = clf.predict(d_test)

        team = df_team['team_str'].values[0]
        opp = df_team['opponent_str'].values[0]
        date = df_team['date'].dt.date.values[0]
        print(f"probability of {team} winning against {opp} on {date} is {round(float(y_pred_train[0]),3)}")
   

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', action='store', required=True)
    required_args.add_argument('-l','--league', action='store', required=True)
    required_args.add_argument('-m','--model', action='store', required=True)
    options=parser.parse_args()
    main(options)
