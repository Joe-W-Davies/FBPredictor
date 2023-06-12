import pandas as pd
import numpy as np
import time
import argparse
import yaml
from datetime import datetime
import xgboost as xgb

from TrainUtils import (
    change_dtypes, 
    create_time_features, 
    encode_features, 
    impute_nulls, 
    MissingDict, 
    add_lags, 
    add_odds,
    add_rolling_vars, 
    add_expanded_vars,
    kelly_critereon
)

#scrape 
def main(options):

    with open(options.config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)
        train_vars_to_roll = config['train_vars_to_roll']
        nominal_vars = config['nominal_vars']
        team_mapping = MissingDict(**config['team_mapping'])
        datasets = config['datasets']
        n_days_lag = config['n_days_lag']
        n_days_rolling = config['n_days_rolling']

    dfs = []
    for df_name in datasets:
        dfs.append( pd.read_csv(f"data/{df_name}", index_col=0) )
    df = pd.concat(dfs)
    
    #data cleaning and filtering
    if options.league: df = df.query(f"comp=='{options.league}'")
    if options.year: df = df.query(f"year=={options.year}")

    df = change_dtypes(df)
    df = create_time_features(df)

    #add target column
    df['y_true'] = (df['result']=='W').astype('int8')
    #FIXME: probs need to impute NA's as well since we did this in the training


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
    
    
    #drop any row with a null if backtesting
    if options.bankroll: df = df.dropna(how='any')
    df = df.sort_values(['date'])
    df.index = range(df.shape[0])
    

    #add opponnent info in
    df['opponent'] = df['opponent'].map(team_mapping)
    df = df.merge(df[running_features+['date']], 
                  left_on=["date", "team"], 
                  right_on=["date", "opponent"], 
                  suffixes=("","_opp"),
                  how='inner'
                  )

    #add odds (must be done after above else you lose opp info)
    if options.add_odds or options.bankroll:
        df, home_odds, away_odds = add_odds(df, team_mapping)

    df = encode_features(df)
    

    #only predict most recent match if not backtesting
    current_date = datetime.today().strftime('%Y-%m-%d')
    if options.bankroll: df = df[df['date'] <= f'{current_date}']
    else: df = df[df['date'] >= f'{current_date}']

    #load model
    clf = xgb.Booster()
    clf.load_model(f'{options.model}')
    print(f'loaded model from: {options.model}')
    final_train_vars = clf.feature_names

    if options.bankroll: 
        #df = df.head(5)
        x_test  = df[final_train_vars] 
        d_test = xgb.DMatrix(x_test, feature_names=final_train_vars)

        df['y_pred'] = clf.predict(d_test)
        df['y_pred_class'] = np.select([df['y_pred'].gt(0.5)], [1], default=0) #FIXME: add threshold into here
        df['win'] = np.select([df['y_pred_class'].eq(df['y_true'])], [1], default=-1)
        df['bankroll_frac'] = df.apply(kelly_critereon, axis=1, args=[home_odds, away_odds]) #FIXME: add threshold into here too

        #drop cases where bet isn't advised
        df = df.query('bankroll_frac>0')
        df['bet'] = options.bankroll * df['bankroll_frac']

        print(df[['team_str','opponent_str','y_pred','y_pred_class','y_true','win','bankroll_frac','bet']].head(50))

        net_winning = sum(df['bet'] * df['win'])
        print(f'total net winning: {net_winning}')
        

 
    else:
        for team in df['team'].unique():
            df_team = df.query(f"team=={team}")
            df_team = df_team.sort_values('date').head(1)
            x_test  = df_team[final_train_vars] 
            d_test = xgb.DMatrix(x_test, feature_names=final_train_vars)
            y_pred_test = clf.predict(d_test)

            team = df_team['team_str'].values[0]
            opp = df_team['opponent_str'].values[0]
            date = df_team['date'].dt.date.values[0]
            print(f"probability of {team} winning against {opp} on {date} is {round(float(y_pred_test[0]),3)}")
   

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('Required Arguments')
    required_args.add_argument('-c','--config', action='store', required=True)
    required_args.add_argument('-m','--model', action='store', required=True)
    opt_args = parser.add_argument_group('Optional Arguments')
    opt_args.add_argument('-l','--league', action='store')
    opt_args.add_argument('-y','--year', action='store')
    opt_args.add_argument('-a','--add_odds', action='store_true',default=False)
    opt_args.add_argument('-b','--bankroll', action='store',default=False, type=float)
    opt_args.add_argument('-t','--threshold', action='store',default=False, type=float)
    options=parser.parse_args()
    main(options)
