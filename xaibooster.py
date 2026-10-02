#!/usr/bin/env python
# ============================================================
# MICROBIOME LIGHTGBM REGRESSION PIPELINE
# with RandomizedSearchCV hyperparameter tuning
# and a click-based command line interface
# ============================================================
#
# Expected input: a CSV where
#   - rows       = samples
#   - columns    = taxa / OTU / ASV abundance features (+ a target column)
#   - one column = the continuous phenotype/metadata you want to predict
#     (e.g. BMI, age, disease severity score, alpha-diversity, etc.)
#
# Install if necessary:
#   pip install lightgbm scikit-learn pandas numpy joblib matplotlib click
#
# Example usage:
#   python microbiome_lgbm_cli.py run \
#       --data-path microbiome_abundance.csv \
#       --target bmi \
#       --sample-id-col sample_id \
#       --transform clr \
#       --prevalence-threshold 0.1 \
#       --n-iter 50 \
#       --output-dir results/
#
#   python microbiome_lgbm_cli.py run --help
# ============================================================

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # safe for headless / CLI use
import matplotlib.pyplot as plt

import click

from lightgbm import LGBMRegressor

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    KFold
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# MICROBIOME-SPECIFIC PREPROCESSING
# ============================================================

def filter_by_prevalence(X: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Drop taxa (columns) that are present (non-zero) in fewer than
    `threshold` fraction of samples. Standard microbiome QC step to
    remove rare/noisy taxa before modeling.
    """
    if threshold <= 0:
        return X

    prevalence = (X > 0).mean(axis=0)
    keep_cols = prevalence[prevalence >= threshold].index.tolist()

    dropped = X.shape[1] - len(keep_cols)
    click.echo(
        f"[prevalence filter] threshold={threshold:.2f} -> "
        f"kept {len(keep_cols)} taxa, dropped {dropped} taxa"
    )

    return X[keep_cols]


def to_relative_abundance(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw counts to relative abundance (each sample/row sums to 1).
    """
    row_sums = X.sum(axis=1)
    row_sums = row_sums.replace(0, np.nan)
    X_rel = X.div(row_sums, axis=0).fillna(0)
    return X_rel


def clr_transform(X: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    """
    Centered log-ratio (CLR) transform. Standard approach for
    compositional microbiome abundance data, since raw/relative
    abundances are constrained to sum to 1 and violate the
    independence assumptions of most ML models.
    """
    X_pc = X + pseudocount
    log_X = np.log(X_pc)
    geometric_mean_log = log_X.mean(axis=1)
    X_clr = log_X.sub(geometric_mean_log, axis=0)
    return X_clr


def preprocess_microbiome_features(
    X: pd.DataFrame,
    transform: str,
    prevalence_threshold: float
) -> pd.DataFrame:
    """
    Full microbiome feature preprocessing chain:
      1. prevalence filtering
      2. relative abundance and/or CLR transform
    """
    X = filter_by_prevalence(X, prevalence_threshold)

    if transform == "relative":
        X = to_relative_abundance(X)
    elif transform == "clr":
        X_rel = to_relative_abundance(X)
        X = clr_transform(X_rel)
    elif transform == "none":
        pass
    else:
        raise ValueError(f"Unknown transform: {transform}")

    return X


# ============================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================

def build_param_distributions() -> dict:
    return {
        "n_estimators": [200, 500, 1000, 1500, 2000],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.10],
        "num_leaves": [15, 31, 63, 127],
        "max_depth": [-1, 5, 8, 10, 15],
        "min_child_samples": [10, 20, 30, 50, 100],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0, 0.01, 0.1, 1.0],
        "reg_lambda": [0, 0.01, 0.1, 1.0],
    }


# ============================================================
# CORE PIPELINE
# ============================================================

def run_pipeline(
    data_path: str,
    target: str,
    sample_id_col: str,
    transform: str,
    prevalence_threshold: float,
    test_size: float,
    n_iter: int,
    cv_folds: int,
    n_jobs: int,
    random_state: int,
    output_dir: str,
    top_n_features: int,
    no_plots: bool,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------
    click.echo(f"Loading microbiome data from: {data_path}")
    df = pd.read_csv(data_path)
    click.echo(f"Dataset shape: {df.shape}")

    if target not in df.columns:
        raise click.ClickException(
            f"Target column '{target}' not found in data. "
            f"Available columns: {list(df.columns)[:20]}..."
        )

    if sample_id_col and sample_id_col in df.columns:
        df = df.set_index(sample_id_col)

    # --------------------------------------------------------
    # 2. DROP ROWS WITH MISSING TARGET
    # --------------------------------------------------------
    before = df.shape[0]
    df = df.dropna(subset=[target])
    click.echo(f"Dropped {before - df.shape[0]} rows with missing target")

    # --------------------------------------------------------
    # 3. SPLIT FEATURES / TARGET
    # --------------------------------------------------------
    y = df[target]
    X = df.drop(columns=[target])

    # Keep only numeric abundance columns as microbiome features.
    # Any leftover categorical metadata columns are kept aside and
    # re-attached as categorical features for LightGBM.
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    click.echo(f"Numeric (abundance) features: {len(numeric_cols)}")
    click.echo(f"Categorical metadata features: {len(categorical_cols)}")

    X_numeric = X[numeric_cols].fillna(0.0)
    X_numeric = preprocess_microbiome_features(
        X_numeric, transform=transform, prevalence_threshold=prevalence_threshold
    )

    if categorical_cols:
        X_cat = X[categorical_cols].astype("category")
        X_final = pd.concat([X_numeric, X_cat], axis=1)
    else:
        X_final = X_numeric

    click.echo(f"Final feature matrix shape: {X_final.shape}")

    # --------------------------------------------------------
    # 4. TRAIN / TEST SPLIT
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=test_size, random_state=random_state
    )
    click.echo(f"Training shape: {X_train.shape} | Test shape: {X_test.shape}")

    # --------------------------------------------------------
    # 5. MODEL + SEARCH SPACE
    # --------------------------------------------------------
    model = LGBMRegressor(
        objective="regression",
        random_state=random_state,
        verbosity=-1,
    )

    param_distributions = build_param_distributions()

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1,
        return_train_score=True,
    )

    # --------------------------------------------------------
    # 6. RUN SEARCH
    # --------------------------------------------------------
    click.echo("\n==============================================")
    click.echo("STARTING RANDOMIZED SEARCH")
    click.echo("==============================================")

    random_search.fit(X_train, y_train)

    click.echo("\n==============================================")
    click.echo("BEST PARAMETERS")
    click.echo("==============================================")
    click.echo(json.dumps(random_search.best_params_, indent=2))

    best_cv_rmse = -random_search.best_score_
    click.echo(f"\nBest CV RMSE: {best_cv_rmse:.4f}")

    best_model = random_search.best_estimator_

    # --------------------------------------------------------
    # 7. TEST SET EVALUATION
    # --------------------------------------------------------
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    click.echo("\n==============================================")
    click.echo("FINAL TEST PERFORMANCE")
    click.echo("==============================================")
    click.echo(f"MAE  : {mae:.4f}")
    click.echo(f"RMSE : {rmse:.4f}")
    click.echo(f"R\u00b2   : {r2:.4f}")

    results = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}, index=y_test.index)

    # --------------------------------------------------------
    # 8. FEATURE (TAXA) IMPORTANCE
    # --------------------------------------------------------
    importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": best_model.feature_importances_,
    }).sort_values(by="Importance", ascending=False)

    click.echo(f"\nTop {top_n_features} taxa/features by importance:")
    click.echo(importance.head(top_n_features).to_string(index=False))

    # --------------------------------------------------------
    # 9. PLOTS
    # --------------------------------------------------------
    if not no_plots:
        top_features = importance.head(top_n_features)

        plt.figure(figsize=(10, 8))
        plt.barh(top_features["Feature"], top_features["Importance"])
        plt.xlabel("Importance")
        plt.ylabel("Taxon / Feature")
        plt.title("Microbiome LightGBM Feature Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_dir / "feature_importance.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, y_pred, alpha=0.5)
        min_value = min(y_test.min(), y_pred.min())
        max_value = max(y_test.max(), y_pred.max())
        plt.plot([min_value, max_value], [min_value, max_value])
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Actual vs Predicted ({target})")
        plt.tight_layout()
        plt.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
        plt.close()

        click.echo(f"\nPlots saved to: {output_dir}")

    # --------------------------------------------------------
    # 10. SAVE ARTIFACTS
    # --------------------------------------------------------
    joblib.dump(best_model, output_dir / "microbiome_lgbm_best_model.pkl")
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    results.to_csv(output_dir / "predictions.csv")

    # Save the preprocessing config + exact training feature columns so
    # `predict` can reproduce the same prevalence filter / transform and
    # feature order at inference time (LightGBM requires an exact match).
    preprocess_config = {
        "transform": transform,
        "prevalence_threshold": prevalence_threshold,
        "numeric_feature_columns": numeric_cols,
        "categorical_feature_columns": categorical_cols,
        "kept_numeric_columns": X_numeric.columns.tolist(),
        "final_feature_order": X_final.columns.tolist(),
        "target": target,
    }
    joblib.dump(preprocess_config, output_dir / "preprocess_config.pkl")

    cv_results = pd.DataFrame(random_search.cv_results_)
    cv_results.to_csv(output_dir / "randomized_search_results.csv", index=False)

    metrics = {
        "target": target,
        "transform": transform,
        "prevalence_threshold": prevalence_threshold,
        "n_features_final": int(X_final.shape[1]),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "best_cv_rmse": float(best_cv_rmse),
        "test_mae": float(mae),
        "test_rmse": float(rmse),
        "test_r2": float(r2),
        "best_params": random_search.best_params_,
    }
    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)

    click.echo("\n==============================================")
    click.echo("ARTIFACTS SAVED")
    click.echo("==============================================")
    for fname in [
        "microbiome_lgbm_best_model.pkl",
        "preprocess_config.pkl",
        "feature_importance.csv",
        "predictions.csv",
        "randomized_search_results.csv",
        "run_summary.json",
    ]:
        click.echo(f"  - {output_dir / fname}")

    click.echo("\nPIPELINE COMPLETE")

    return metrics


# ============================================================
# CLICK CLI
# ============================================================

@click.group()
def cli():
    """Microbiome LightGBM regression pipeline with hyperparameter tuning."""
    pass


@cli.command()
@click.option(
    "--data-path", "-d",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to CSV with samples as rows, taxa abundances + target as columns.",
)
@click.option(
    "--target", "-t",
    required=True,
    help="Name of the target/phenotype column to predict.",
)
@click.option(
    "--sample-id-col",
    default=None,
    help="Optional column name to use as the sample index (e.g. 'sample_id').",
)
@click.option(
    "--transform",
    type=click.Choice(["clr", "relative", "none"]),
    default="clr",
    show_default=True,
    help="Compositional transform applied to numeric abundance features.",
)
@click.option(
    "--prevalence-threshold",
    type=float,
    default=0.1,
    show_default=True,
    help="Drop taxa present in fewer than this fraction of samples (0 disables filtering).",
)
@click.option(
    "--test-size",
    type=float,
    default=0.20,
    show_default=True,
    help="Fraction of data held out as the final test set.",
)
@click.option(
    "--n-iter",
    type=int,
    default=50,
    show_default=True,
    help="Number of random hyperparameter combinations to try.",
)
@click.option(
    "--cv-folds",
    type=int,
    default=5,
    show_default=True,
    help="Number of cross-validation folds.",
)
@click.option(
    "--n-jobs",
    type=int,
    default=-1,
    show_default=True,
    help="Number of parallel jobs (-1 uses all CPU cores).",
)
@click.option(
    "--random-state",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for reproducibility.",
)
@click.option(
    "--output-dir", "-o",
    default="microbiome_results",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Directory where models, plots, and result CSVs are saved.",
)
@click.option(
    "--top-n-features",
    type=int,
    default=20,
    show_default=True,
    help="Number of top features/taxa to show and plot.",
)
@click.option(
    "--no-plots",
    is_flag=True,
    default=False,
    help="Skip generating and saving plots (useful for headless/batch runs).",
)
def run(
    data_path,
    target,
    sample_id_col,
    transform,
    prevalence_threshold,
    test_size,
    n_iter,
    cv_folds,
    n_jobs,
    random_state,
    output_dir,
    top_n_features,
    no_plots,
):
    """
    Run the full microbiome regression pipeline:
    load data -> preprocess abundances -> tune LightGBM ->
    evaluate on held-out test set -> save model, plots, and metrics.
    """
    run_pipeline(
        data_path=data_path,
        target=target,
        sample_id_col=sample_id_col,
        transform=transform,
        prevalence_threshold=prevalence_threshold,
        test_size=test_size,
        n_iter=n_iter,
        cv_folds=cv_folds,
        n_jobs=n_jobs,
        random_state=random_state,
        output_dir=output_dir,
        top_n_features=top_n_features,
        no_plots=no_plots,
    )


@cli.command()
@click.option(
    "--model-path", "-m",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a saved .pkl model from a previous 'run'.",
)
@click.option(
    "--config-path", "-c",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to preprocess_config.pkl saved alongside the model. "
         "Defaults to 'preprocess_config.pkl' in the same directory as --model-path.",
)
@click.option(
    "--data-path", "-d",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="CSV of new samples (same raw abundance columns as training) to predict on.",
)
@click.option(
    "--sample-id-col",
    default=None,
    help="Optional column name to use as the sample index.",
)
@click.option(
    "--output-path", "-o",
    default="new_predictions.csv",
    show_default=True,
    type=click.Path(dir_okay=False),
    help="Where to save predictions CSV.",
)
def predict(model_path, config_path, data_path, sample_id_col, output_path):
    """
    Load a saved model and predict on new microbiome samples, applying the
    exact same prevalence filtering / CLR transform used at training time.
    """
    model_path = Path(model_path)
    config_path = Path(config_path) if config_path else model_path.parent / "preprocess_config.pkl"

    if not config_path.exists():
        raise click.ClickException(
            f"Could not find preprocess_config.pkl at {config_path}. "
            f"Pass --config-path explicitly if it's stored elsewhere."
        )

    click.echo(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    click.echo(f"Loading preprocessing config from: {config_path}")
    config = joblib.load(config_path)

    df = pd.read_csv(data_path)
    if sample_id_col and sample_id_col in df.columns:
        df = df.set_index(sample_id_col)

    # Drop the target column if it's present in the new data (harmless if absent).
    df = df.drop(columns=[config["target"]], errors="ignore")

    # Re-apply the same numeric abundance transform used during training.
    numeric_cols = [c for c in config["numeric_feature_columns"] if c in df.columns]
    missing_numeric = set(config["numeric_feature_columns"]) - set(numeric_cols)
    if missing_numeric:
        click.echo(
            f"Warning: {len(missing_numeric)} training taxa columns missing "
            f"from new data; treating them as zero abundance."
        )

    X_numeric = df.reindex(columns=config["numeric_feature_columns"], fill_value=0.0).fillna(0.0)

    # Restrict to the taxa that survived prevalence filtering at train time,
    # then apply the same compositional transform (filtering itself is not
    # re-run, since the training set defines which taxa the model uses).
    X_numeric = X_numeric[config["kept_numeric_columns"]]
    transform = config["transform"]
    if transform == "relative":
        X_numeric = to_relative_abundance(X_numeric)
    elif transform == "clr":
        X_numeric = clr_transform(to_relative_abundance(X_numeric))
    # transform == "none" -> leave as-is

    categorical_cols = config["categorical_feature_columns"]
    if categorical_cols:
        X_cat = df.reindex(columns=categorical_cols).astype("category")
        X_final = pd.concat([X_numeric, X_cat], axis=1)
    else:
        X_final = X_numeric

    # Ensure exact column order LightGBM was trained on.
    X_final = X_final.reindex(columns=config["final_feature_order"])

    preds = model.predict(X_final)
    out = pd.DataFrame({"Predicted": preds}, index=df.index)
    out.to_csv(output_path)
    click.echo(f"Saved {len(out)} predictions to: {output_path}")


if __name__ == "__main__":
    cli()
