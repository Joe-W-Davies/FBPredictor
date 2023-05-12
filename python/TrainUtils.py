import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def impute_nulls(df, train_vars, impute=False):
    #sometimes possesion info etc just wasn't filled in on the site. 
    #if impite, impute from avgs of similar rows
    if impute:
        for col_name in train_vars:
            df[col_name] = df[col_name].fillna(df.groupby(['team','year'])[col_name].transform('mean'))
    else:
        df = df.dropna(how='any')

    return df


def change_dtypes(df):
    def fix_away_games(score):
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


def create_time_features(df):
    if 'time' in df.columns: df['hours'] = df['time'].str.replace(':.+','', regex=True).astype('int')
    if 'date' in df.columns: 
        df['day'] = df['date'].dt.dayofweek
        df = df.sort_values('date')
        df['days_since_game'] = df['date'].diff().dt.days

    return df

def encode_features(df):
    if 'formation' in df.columns: df['formation'] = df['formation'].astype('category').cat.codes
    if 'venue' in df.columns: df['venue'] = df['venue'].astype('category').cat.codes

    #trickier because we have two columns that need encoding in the same way
    if ('team' in df.columns) and ('opponent' in df.columns): 
        encoder = LabelEncoder()
        encoder.fit( pd.concat([ df['team'],df['opponent'] ]) )
        df['team'] = encoder.transform(df['team'])
        df['opponent'] = encoder.transform(df['opponent'])
   
    
    return df

class MissingDict(dict):
  __missing__ = lambda self, key:key
