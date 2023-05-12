# FBPredictor
Scrape football stats and train models to predict the outcome. 

Example scraping command:

```
python3 scripts/scrape.py -c configs/scrape_cfg.yaml
```

Example (GBDT) training command:

```
python3 scripts/predict.py -c configs/train_cfg.yaml
```

There are many features to train with - if you want to reduce them, try running with a modified Boruta-SHAP algo by adding the option:

```
python3 scripts/predict.py -c configs/train_cfg.yaml --feature_select
```

## To-Do's: scraping
* weight up scraping more data v.s. not having expected-goals etc. for older years

## To-Do's: model training
* Plot inputs and make correlation map
* plot output scores
* Try different models
