# FBPredictor
Scrape football stats and train models to predict the outcome. 

Example scraping command:

```
python3 scripts/scrape.py -c configs/scrape_cfg.yaml
```

Example (GBDT) training command:

```
python3 scripts/train.py -c configs/train_cfg.yaml (--feature_select)
```

There are many features to train with - if you want to reduce them, try running with a modified Boruta-SHAP algo by adding the option:


To make a prediction on each match in a given league e.g. the EPL:

```
python3 scripts/predict.py --config configs/train_cfg.yaml -l "Premier League" --model models/model.json 
```

## To-Do's: scraping
* weigh up scraping more data v.s. not having expected-goals etc. for older years
* check if BundesLiga matches are somehow being scraped in EPL 

## To-Do's: model training
* remove hard coding on train/test split date
* Plot inputs and make correlation map
* plot output scores
* Try different models
* debug why some rows are lost in training join
