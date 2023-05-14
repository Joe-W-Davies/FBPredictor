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
python3 scripts/train.py -c configs/train_cfg.yaml (--feature_select) (--hp_opt)
```

There are many features to train with - if you want to reduce them, try running with a modified Boruta-SHAP algo by adding the option `--feature_select`. To optimise the model hyperparameters, add the `--hp_opt` option.





## Predict upcoming matches

To make a prediction on each match in a given league e.g. the EPL:

```
python3 scripts/predict.py --config configs/train_cfg.yaml -l "Premier League" --model models/model.json 
```




### To-Do's: scraping
* weigh up scraping more data v.s. not having expected-goals etc. for older years
* check if BundesLiga matches are somehow being scraped in EPL 

### To-Do's: model training
* remove hard coding on train/test split date
* Plot inputs and make correlation map
* plot output scores
* Try different models
* debug why some rows are lost in training join

### To-Do's: betting strategy
* add script to simulate Kelly Criterion and Expected Value Analysis (and both being >0) for past games
