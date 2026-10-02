# xAIBooster

- applying lightGBM to microbiome abundances 

```
python microbiome_lgbm_cli.py run -d abundance.csv -t bmi --sample-id-col sample_id
python microbiome_lgbm_cli.py predict -m microbiome_results/microbiome_lgbm_best_model.pkl -d new_samples.csv

```

Gaurav Sablok \
gsablok@proton.me
