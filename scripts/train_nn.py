from datetime import datetime
import argparse
import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
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
    plot_confusion_matrix,
    plot_output_score,
    plot_shaps,
    plot_loss
)

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from NN import FFNet


def main(options):
    #unpack config
    with open(options.config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)
        nominal_vars = config['nominal_vars']
        train_vars_to_roll = config['train_vars_to_roll']
        more_odds_vars = config['more_odds_vars']
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
    #df = df.dropna(how='all')
    df = df.dropna(thresh=int(df.shape[1]/5)) 
    df = change_dtypes(df)
    col_names = []
    for k, list_level_map in train_vars_to_roll.items():
        for dic in list_level_map:
            for t_level, bottom_list in dic.items():
                for b_level in bottom_list:
                    col_names.append((t_level+b_level).lower())
    train_vars_to_roll = col_names.copy()
    df = impute_nulls(df, [v for v in train_vars_to_roll if v in df.columns], impute=True)
    df = create_time_features(df)
    current_date = datetime.today().strftime('%Y-%m-%d')
    df = df[df['date']< f'{current_date}']
    print(f'after cleaning, df has: {df.duplicated(keep=False).sum()} duplicate rows')

    #add target column
    df['y_true'] = df['result'].astype('category').cat.codes

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
        df, home_odds, away_odds, draw_odds, more_odds_vars = add_odds(df, team_mapping, more_odds_vars)
        nominal_vars = nominal_vars + home_odds + away_odds + draw_odds + more_odds_vars

    df = encode_features(df)
    
    #train/test split
    final_train_vars = nominal_vars + running_features + [v+'_opp' for v in running_features if v not in nominal_vars]

    print(f'training with {len(final_train_vars)} variables')
    
    x_train = df[df['date']<'2022-08-01'][final_train_vars] 
    y_train = df[df['date']<'2022-08-01']['y_true']
    x_test  = df[df['date']>'2022-08-01'][final_train_vars] 
    y_test  = df[df['date']>'2022-08-01']['y_true']

    #NN stuff
    #standardize inputs
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
   

    #convert to tensor
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.long)

    train_dataset = TensorDataset(x_train,y_train)
    test_dataset = TensorDataset(x_test,y_test)

    #train model
    input_size = len(final_train_vars)
    hidden_1 = 20
    hidden_2 = 10
    num_classes = len(df['y_true'].unique())
    dropout_prob = 0.2
    learning_rate = 0.01
    batch_size = 256
    num_epochs = 10

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model = FFNet(input_size, hidden_1, hidden_2, num_classes, dropout_prob)
    print(f'Number of trainable params: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
    critereon = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=1, gamma=0.9)

   
    train_loss_epochs = []
    test_loss_epochs = []
    for epoch in range(num_epochs):
        model.train()

        train_loss = 0
        for x_batch_train, y_batch_train in train_loader:
            preds = model(x_batch_train)
            loss = critereon(preds, y_batch_train)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss/len(train_loader)
        train_loss_epochs.append(avg_train_loss)
        if epoch%10==0:
            print(f'train loss at epoch {epoch} is: {avg_train_loss}')

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x_batch_test, y_batch_test in test_loader:
                preds = model(x_batch_test)
                loss = critereon(preds, y_batch_test)
                test_loss += loss.item()

        avg_test_loss = test_loss/len(test_loader)
        test_loss_epochs.append(avg_test_loss)
        if epoch%10==0:
            print(f'test loss at epoch {epoch} is: {avg_test_loss}')
        scheduler.step()
 


    #get accuracy
    model.eval()
    with torch.no_grad():
        y_pred_train = model(x_train).detach().numpy()
        y_pred_train_class = np.argmax(y_pred_train,axis=1)

        y_pred_test = model(x_test).detach().numpy()
        y_pred_test_class = np.argmax(y_pred_test,axis=1)
        print(y_pred_test)
        print(y_pred_test_class)



    print(f'train accuracy: {accuracy_score(y_train, y_pred_train_class)}')
    print(f'test accuracy: {accuracy_score(y_test, y_pred_test_class)}')

    #save model
    if options.save_model: pass

    #make some plots
    #plot_roc(clf, y_train, y_pred_train, y_test, y_pred_test)
    plot_confusion_matrix(y_test, y_pred_test_class)
    plot_loss(num_epochs, train_loss_epochs, test_loss_epochs)
    #plot_output_score(y_test, y_pred_test)
    #plot_shaps(clf, x_test, final_train_vars)
   

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
