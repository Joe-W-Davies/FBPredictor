import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import glob 

class MissingDict(dict):
  __missing__ = lambda self, key:key

def impute_nulls(
        df: pd.DataFrame, 
        train_vars: list[str], 
        impute: bool=False
) -> pd.DataFrame:

    #sometimes possesion info etc just wasn't filled in on the site. 
    #if impute, impute from avgs of similar rows. Tends not to make too much diff to final outputs
    if impute:
        for col_name in train_vars:
            df[col_name] = df[col_name].fillna(df.groupby(['team','year'])[col_name].transform('mean'))
    else:
        df = df.dropna(how='any')

    return df


def change_dtypes(df: pd.DataFrame) -> pd.DataFrame:

    #n.b. np.nan has type float
    def fix_away_games(score: pd.Series) -> int | float: 
        if score is np.nan: return np.nan
        # for 2 leg games, first leg score is given in brackets e.g. 1 (2)
        elif '(' in score: return int(score[0])
        else: return int(float(score))

    if 'date' in df.columns: df['date'] = pd.to_datetime(df['date'])
    if 'gf' in df.columns: 
        df['gf'] = df['gf'].apply(fix_away_games)
    if 'ga' in df.columns: 
        df['ga'] = df['ga'].apply(fix_away_games)

    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame: 

    if 'time' in df.columns: 
        df['hours'] = df['time'].str.replace(
            ':.+',
            '', 
            regex=True).astype('int')
    if 'date' in df.columns: 
        df['day'] = df['date'].dt.dayofweek
        df = df.sort_values('date')
        df['days_since_game'] = df['date'].diff().dt.days

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:

    if 'formation' in df.columns: 
        df['formation'] = df['formation'].astype('category').cat.codes
    if 'venue' in df.columns: 
        df['venue'] = df['venue'].astype('category').cat.codes

    #trickier because we have two columns that need encoding in the same way
    if ('team' in df.columns) and ('opponent' in df.columns): 
        #keep str versions as well for prediction
        df['team_str'] = df['team'].copy()
        df['opponent_str'] = df['opponent'].copy()

        encoder = LabelEncoder()
        encoder.fit( pd.concat([ df['team'],df['opponent'] ]) )
        df['team'] = encoder.transform(df['team'])
        df['opponent'] = encoder.transform(df['opponent'])
   
    
    return df



def add_odds(
        df: pd.DataFrame, 
        team_mapping: MissingDict,
        more_odds_vars: list[str]
) -> tuple[pd.DataFrame, list[str], list[str]]:


    files = glob.glob('data/odds/*.csv')
    odds_dfs = []
    
    for fn in files:
        odds_dfs.append(pd.read_csv(fn))
    odds = pd.concat(odds_dfs)
    
    #few options:
    #1) could average all the non-na odds
    #2) could put all the non-na odds into the model
    #4) could put all the non-na odds into the model, but check they aren't too sparse
    
    #going to do option  #2 for now
    odds_providers = ['B365', 'BW', 'IW', 'PS', 'WH', 'VC']
    home_odds = [op+'H' for op in odds_providers]
    away_odds = [op+'A' for op in odds_providers]
    draw_odds = [op+'D' for op in odds_providers]

    #odds[more_odds_vars].info()

    odds_vars_subbed = {x: x.replace(r'>', '__').replace(r'<', '_') for x in more_odds_vars}
    #print(odds_vars_subbed)
    

    odds['date'] = pd.to_datetime(odds['Date'], dayfirst=True)
    odds = odds[home_odds + away_odds + draw_odds + more_odds_vars + ['date','HomeTeam','Div']].dropna(axis='rows')
    odds['HomeTeam'] = odds['HomeTeam'].map(team_mapping)
    odds = odds.rename(columns=odds_vars_subbed)
    more_odds_vars = list(odds_vars_subbed.values())

    #DEBUGs
    #l2 = odds['HomeTeam'].unique()
    #l1 = df['team'].unique()
    #print([x for x in l1 if x not in l2] + [x for x in l2 if x not in l1])
    #END DEBUGs


    #since only one set of odds are provide per game, and we have current
    #have two rows per game, inner join will drop Away games (repeated info)
    df = df.merge(
        odds, 
        left_on=["date", "team"], 
        right_on=["date", "HomeTeam"], 
        suffixes=("","_opp"),
        how='inner'
    )


    return df, home_odds, away_odds, draw_odds, more_odds_vars


def add_lags(
        df: pd.DataFrame, 
        n_days: list[int], 
        current_vars: set[str]
) -> tuple[pd.DataFrame, set[str]]:


    lags = [x+1 for x in range(n_days)]
    lagged_dfs = []
    for group in df.groupby(['team','year'], group_keys=False):
        gr = group[-1].sort_values(['date'])
        for lag in lags:
            gr[f'lag_{lag}'] = gr['y_true'].copy().shift(lag)
            #add opponent in as a lag so it lines up with the game. Give info: hard opponent, no win
            current_vars.add(f'lag_{lag}')
        lagged_dfs.append(gr)
    df = pd.concat(lagged_dfs, ignore_index=True) 

    return df, current_vars


def add_rolling_vars(
        df: pd.DataFrame, 
        n_days: list[int], 
        current_vars: set[str],
        train_vars_to_roll: list
) -> tuple[pd.DataFrame, set[str]]:


    rolled_dfs = []
    for group in df.groupby(['team','year'], group_keys=False):
        gr = group[-1].sort_values(['date'])
        for var in train_vars_to_roll:

            mean_var = var+f'_rolling_avg_{n_days}'
            gr[mean_var] = gr[var].copy().rolling(n_days, closed='left').mean()
            #gr[mean_var] = gr[var].copy().ewm(span=1).mean().shift()
            current_vars.add(mean_var)

            med_var = var+f'_rolling_med_{n_days}'
            gr[med_var] = gr[var].copy().rolling(n_days, closed='left').median()
            current_vars.add(med_var)

            std_var = var+f'_rolling_std_{n_days}'
            gr[std_var] = gr[var].copy().rolling(n_days, closed='left').std()
            current_vars.add(std_var)

        rolled_dfs.append(gr)
    df = pd.concat(rolled_dfs, ignore_index=True)
    
    return df, current_vars


def add_expanded_vars(
        df: pd.DataFrame, 
        current_vars: set[str],
        train_vars_to_roll: list,
) -> tuple[pd.DataFrame, set[str]]:


    expanded_dfs = []
    for group in df.groupby(['team','year'], group_keys=False):
        gr = group[-1].sort_values(['date'])
        for var in train_vars_to_roll:
            modified_var = var+'_expanded'
            gr[modified_var] = gr[var].copy().expanding().mean().shift()
            current_vars.add(modified_var)
        expanded_dfs.append(gr)
    df = pd.concat(expanded_dfs, ignore_index=True)
    

    return df, current_vars


def kelly_critereon(row: pd.Series, h_odds:list, a_odds:list) -> float:
    'get the best odds and apply KC to it'
    print(row.name)
    if row['y_pred']>0.5: 
         #remember model predicts prob of home team winning
         odds = h_odds
         prob = row['y_pred']
    else:
         #remember model predicts prob of home team winning
         odds = a_odds
         prob = 1-row['y_pred']

    best_odds = max(row[odds]) - 1
    numerator =  (best_odds * prob)
    numerator -= (1-prob)

    kc =  numerator/best_odds
    return kc


