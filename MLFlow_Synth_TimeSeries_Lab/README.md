### Lab: MLFlow + Optuna experiments for synthetic-weather + RandomForestRegressor.


### This lab demonstrates:
1. Synthetic time-series generator (signal_gen) with configurable config (SignalConfig).
2. Feature builder + metrics
3. Training script (train_rf.py) - argparse for hyperparameters, MLflow logging.
4. Optuna tuning script (tune_rf_optuna.py) - Runs multiple RF configurations, nested MLflow runs.
5. Canonical MLflow tracking store rooted at project level - shared by scripts and notebook.
6. Analysis notebook (serve_and_analyze.ipynb) that does extensive serving (see below).

### Algo:
- Regression problem;
- but supervised tabular regression, with lagged features as the custom way to expose 'past-dependency structure'.
- construct features like: `y(t-1), y(t-2), …, y(t-k)`.
- (T−W) training samples. `X_t = [temp[t-1], temp[t-2], ..., temp[t-36]], y_t = temp[t]`.
- same conceptual basis as AR, ARIMA, and even LSTMs.
- For tree-based models, lagged features are the only way to expose past-dependency structure.

#### About Data and Overview:

Quick signal designs:
- I have a few old custom - synthetic time series weather signal designs that I will be using for this lab.
- **Signal Features**: 
  - Annual cycle with moderate amplitude to encode seasonality
  - A “daily” cycle that is not physically perfect (index as proxy for hours), but gives a higher frequency oscillation.
  - A linear trend to mimic slow warming.
  - An AR(1)-like low-frequency weather component that carries memory.
  - Gaussian noise on top.
- Will add another file to hold old signal designs too.
- **classic multi-scale temporal structure.** !!
- Using RandomForestRegressor from sklearn as the model for this lab.
- Using RMSE, MAE but also SMAPE. When forecasting temporal signals like weather, demand, energy, absolute values don’t matter as much as proportional error. SMAPE gives you a relative, scale-free outlook of model.


#### About Optuna.
- Optuna: framework for automating the optimization process of these hyperparameters.
- Advantage found:
  - It will **resample from same search space** , it will auto tune and optimize for min val_rmse, its logging new parent runs and N+, 30 or 50 experiments under the same parent.

#### Using the scripts:
- meant to be served as `python -m src.train_rf --window-size 90 --noise-std 3.0`
- Every MLflow run is wrapped in:
  ```python
      with mlflow.start_run(run_name=run_name):
      ...
      mlflow.log_metric("test_rmse", rmse)
      ...
   ```
- Log parameters -> train -> evaluate -> log metrics -> log artifacts (plots) -> Finalize run
- MLflow stores: run ID, experiment ID, timestamp, tags, parameters, metrics, artifacts.

#### Can run in terminal or Jupyter Notebook. Use `%run` concept.

1. %run magic in notebook
   - %run ../src/train_rf.py --window-size 120 --noise-std 4.0 --n-estimators 400

2. Default experiment (ID = 0), Experiment #1, **15 nested child runs (each Optuna trial)**
```
mlruns/
 ├── 0
 ├── 395361917797520105
 └── 590785582627217283
```

3. Optuna + MLflow status: 
   1. Total runs in experiment: 47
   2. Optuna trial runs: 45

#### Proper Serve and Selection:
- Notebook has code which:
1. DataFrame view of all Optuna trials.
2. Several param-, exploratory plots over trials.
3. Identify and inspect the best run.
4. Rebuild dataset, load best model, and evaluate.
5. Plot test predictions and log figure back to MLflow.

- **Notebook features**: 
- can successfully: Query all runs via MlflowClient, Filter Optuna trials, Sort by val_rmse, Reconstruct the dataset, Load the best model from `runs:/<run_id>/model`, Evaluate on the test split, Plot and log forecast figures back to MLflow !`:)`
- to polish more: add infer_signature + input_example to log_model.

### Instructions for users:
- mlruns/ and src_notebooks/mlruns/ are ignored → no run blobs in Git.
- users are recommended to create their own mlruns/ at project root when running scripts or notebooks. !!

```python
# ============================================================
# INSTALLATION (UV Recommended)
# ============================================================
# 1. Create env:
#       uv venv weather_venv
#
# 2. Activate:
#       weather_venv\Scripts\activate          # Windows
#       source weather_venv/bin/activate        # Mac / Linux
#
# 3. Install:
#       uv pip install -r requirements.txt
```



#### More intuition about the Algorithm:

1. With lagged features:
   - RF learns “if the last ~36 values were rising and periodic, expect similar behavior”.
   - RF can partially capture periodicity because lagged values encode repeated motifs.
   - RF can approximate nonlinear interactions because tree splits can isolate patterns like:
     - if temp[t-1] > X and temp[t-12] < Y → temp[t] ≈ some value
   - RF can follow local oscillations (daily-decay patterns)
   - RF can smooth noise because tree ensembles aggregate across many splits.
   - Lagged features thus give RF visibility into the underlying structure, even though RF has zero built-in concept of time.
2. RandomForestRegressor actually is... `many decision trees trained on bootstrapped samples`, each using random feature subsets, averaged together.
   - Core...
   - Bootstrap sampling (bagging): Each tree sees a different slice of the dataset. reduces variance.
   - Random feature subsets at each split: Prevents all trees from learning the same dominant feature.
   - Forces diverse trees → better generalization. Decision trees as base learners
   - Each tree partitions feature space using "if-else" axis-aligned splits.
   - Trees can capture nonlinear interactions and threshold effects.
3. RandomForestRegressor specifically implements: CART regression trees, MSE as impurity measure, Averaging predictions.
   
