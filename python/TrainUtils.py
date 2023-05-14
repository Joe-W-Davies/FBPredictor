import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Set, Tuple

def impute_nulls(
        df: pd.DataFrame, 
        train_vars: List[str], 
        impute: bool=False
    ) -> pd.DataFrame:
    #sometimes possesion info etc just wasn't filled in on the site. 
    #if impute, impute from avgs of similar rows. Tends not to make too much diff
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



def add_lags(
        df: pd.DataFrame, 
        n_days: List[int], 
        current_vars: Set[str]
    ) -> Tuple[pd.DataFrame, Set[str]]:


    lags = [x+1 for x in range(n_days)]
    lagged_dfs = []
    for group in df.groupby(['team','year'], group_keys=False):
        gr = group[-1].sort_values(['date'])
        for lag in lags:
            gr[f'lag_{lag}'] = gr['y_true'].shift(lag)
            current_vars.add(f'lag_{lag}')
        lagged_dfs.append(gr)
    df = pd.concat(lagged_dfs, ignore_index=True) 

    return df, current_vars


def add_rolling_vars(
        df: pd.DataFrame, 
        n_days: List[int], 
        current_vars: Set[str],
        train_vars_to_roll: List
    ) -> Tuple[pd.DataFrame, Set[str]]:


    rolled_dfs = []
    for group in df.groupby(['team','year'], group_keys=False):
        gr = group[-1].sort_values(['date'])
        for var in train_vars_to_roll:
            #gr[rolled_train_vars[var]] = gr[var].rolling(n_days, closed='left', win_type='exponential').mean(tau=0.5) #closed='left' does not work with exp window
            mean_var = var+f'_rolling_avg_{n_days}'
            gr[mean_var] = gr[var].rolling(n_days, closed='left').mean()
            current_vars.add(mean_var)

            med_var = var+f'_rolling_med_{n_days}'
            gr[med_var] = gr[var].rolling(n_days, closed='left').median()
            current_vars.add(med_var)

        rolled_dfs.append(gr)
    df = pd.concat(rolled_dfs, ignore_index=True)
    
    return df, current_vars

def add_expanded_vars(
        df: pd.DataFrame, 
        current_vars: Set[str],
        train_vars_to_roll: List,
    ) -> Tuple[pd.DataFrame, Set[str]]:


    expanded_dfs = []
    for group in df.groupby(['team','year'], group_keys=False):
        gr = group[-1].sort_values(['date'])
        for var in train_vars_to_roll:
            modified_var = var+'_expanded'
            gr[modified_var] = gr[var].expanding().mean()
            current_vars.add(modified_var)
        expanded_dfs.append(gr)
    df = pd.concat(expanded_dfs, ignore_index=True)
    

    return df, current_vars


class MissingDict(dict):
  __missing__ = lambda self, key:key
