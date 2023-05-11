# FBPredictor
Scrape football stats and train models to predict the outcome

Example scraping command:

```
python3 scripts/scrape.py -c configs/scrape_cfg.yaml
```

Example (GBDT) training command:

```
python3 scripts/predict.py -c configs/train_cfg.yaml
```

## To-Do's: scraping
* re-factor scraping additional tables to avoid code replication
* break out more functions

## To-Do's: model training
* Plot inputs and make Corr Map
* add SHAP values and port over Boruta SHAP
* plot output scores
* add date of last game by subtracting current date - previous game date (before dropping nans)
