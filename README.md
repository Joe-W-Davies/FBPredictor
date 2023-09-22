# FBPredictor

Scrape football stats and train models to predict the outcome. 

## Scrape matches

Example scraping command:

```
python3 scripts/scrape.py -c configs/scrape_cfg.yaml
```




## Train model

Example (GBDT) training command:

```
python3 scripts/train.py -c configs/train_cfg.yaml (--feature_select) (--hp_opt) (-a)
```

There are many features to train with - if you want to reduce them, try running with a modified Boruta-SHAP algo by adding the option `--feature_select`. To optimise the model hyperparameters, add the `--hp_opt` option. To add info on betting odds from [football-data.co.uk](https://www.football-data.co.uk), add the `-a` option. This will remove duplicate home/away games since the odds are recorded once per game.

The current best test accuracy is: 67.5%.

See the `three_class` branch for predicting draws as well. The accuracy for that classifier is around 55.2%. A neural network model currently perform a little worse than a BDT.




## predict upcoming matches and back test


### predicting upcoming matches
To make a prediction on each match in a given league e.g. the EPL in 2022:

```
python3 scripts/predict.py --config configs/train_cfg.yaml -l "Premier League" -y 2022 --model models/model.json 
```

### Back testing

Adding the `-b [ammount]` option backtests with the Kelly Critereon used as the betting strategy. There is also the option to use this combo with the `-f` option to fix a bet at the ammount specified by `-b`, rather than updating the total i.e. assume same bank balance of `-b [ammount]` before each bet. This is less risky if the updated pot gets larger.



### To-Do's: scraping
* weigh up scraping more data v.s. not having expected-goals etc. for older years -[x] -> no real improvemnt (see `more_years` branch)
* scrape tables where multiple columns have the same name [DONE]
* scrape additional tables: Goalkeeping, Pass Types, Goal and Shot Creation, Defensive Actions, and Misc Stats [DONE]
* Try keeping international matches to boost rows (will probs get filtered in merge anyway)
* Try impute odds information which is currently unused due to many missing values

### To-Do's: model training
* lag other features like the last opponents played (gives info about whether they lost to a difficult/easy team)
* remove hard coding on train/test split date
* Plot inputs and make correlation map (separate script?)
* Try different models
* debug why some rows are lost in training join

### To-Do's: betting strategy
* consider how to quantify an uncertainty on the game probs

### To-Do's: general
* go through all FIXME's
* think about whether to drop either home/away match to prevent double same info being fed twice (FYI done autoamticaly if joinining odds info in)
* fix issue with having to scrape data and re-run workflow each time you want to predict
* add docstrings
* scrape live odds for upcoming matches if adding these features to models
