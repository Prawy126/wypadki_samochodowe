"""
Pipeline LightGBM — model porównawczy dla klasyfikacji ciężkości wypadków.

Kluczowe cechy:
  - Natywne wsparcie dla cech kategorycznych (int IDs z preprocessingu)
  - class_weight='balanced' zamiast ręcznych wag
  - Early stopping na zbiorze walidacyjnym
  - Wykres feature importance (Top-25)
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from . import config
from .calibration import apply_multipliers, save_multipliers, tune_class_multipliers
from .evaluate import evaluate_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wczytanie danych
# ---------------------------------------------------------------------------

def _load_data(
    metadata: dict,
    data_dir: Path,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, list]:
    """
    Wczytuje train/val/test z plików csv.gz.
    Zwraca (X_train, y_train, X_val, y_val, X_test, y_test, feature_cols).
    """
    target_col   = metadata["target_col"]
    feature_cols = metadata["num_cols"] + metadata["cat_cols"] + metadata["bin_cols"]
    all_cols     = [target_col] + feature_cols

    if "Year" in metadata.get("cat_cols", []) or "Year" not in metadata.get("num_cols", []):
        raise ValueError(
            "Niespójna schema: 'Year' musi być cechą numeryczną (num_cols), "
            "a nie kategoryczną. Uruchom preprocessing ponownie."
        )

    logger.info("Wczytywanie danych dla LightGBM...")
    train_df = pd.read_csv(data_dir / "train_preprocessed.csv.gz", compression="gzip", usecols=all_cols)
    val_df   = pd.read_csv(data_dir / "val_preprocessed.csv.gz",   compression="gzip", usecols=all_cols)
    test_df  = pd.read_csv(data_dir / "test_preprocessed.csv.gz",  compression="gzip", usecols=all_cols)

    # Upewniamy się że cat cols są int (LightGBM categorical_feature tego wymaga)
    for col in metadata["cat_cols"]:
        train_df[col] = train_df[col].astype(int)
        val_df[col]   = val_df[col].astype(int)
        test_df[col]  = test_df[col].astype(int)

    X_train, y_train = train_df[feature_cols], train_df[target_col].values
    X_val,   y_val   = val_df[feature_cols],   val_df[target_col].values
    X_test,  y_test  = test_df[feature_cols],  test_df[target_col].values

    # Drift guard: Year zakodowany jako kategoria (np. 1..8) to błąd schemy.
    year_vals = X_train["Year"].to_numpy()
    if np.allclose(year_vals, np.round(year_vals), atol=1e-6):
        year_min = float(year_vals.min())
        year_max = float(year_vals.max())
        year_unique = int(np.unique(year_vals).size)
        if 0.0 <= year_min and year_max <= 50.0 and year_unique <= 50:
            raise ValueError(
                "Wykryto drift danych: kolumna 'Year' wygląda na zakodowaną "
                f"kategorycznie (min={year_min:.1f}, max={year_max:.1f}, unique={year_unique}). "
                "Uruchom preprocessing/preprocessing.py ponownie."
            )

    logger.info(f"Train: {X_train.shape}  |  Val: {X_val.shape}  |  Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


# ---------------------------------------------------------------------------
# Wykres Feature Importance
# ---------------------------------------------------------------------------

def _plot_feature_importance(
    model: LGBMClassifier,
    feature_cols: list,
    output_dir: Path,
    top_n: int = 25,
) -> None:
    import matplotlib.pyplot as plt

    imp_df = (
        pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#1976D2" if i < 5 else "#64B5F6" for i in range(len(imp_df))]
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color=colors[::-1])
    ax.set_title(f"LightGBM — Top {top_n} Feature Importances (split)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance (number of splits)")
    ax.axvline(x=imp_df["importance"].mean(), color="red", linestyle="--", alpha=0.6, label="Średnia")
    ax.legend()
    plt.tight_layout()

    path = output_dir / "lgbm_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature importance zapisany: {path}")


# ---------------------------------------------------------------------------
# Główna funkcja treningowa
# ---------------------------------------------------------------------------

def train_lgbm(
    metadata: dict,
    cfg: Optional[dict] = None,
) -> Tuple[LGBMClassifier, dict]:
    """
    Trenuje model LightGBM.

    Args:
        metadata : słownik z metadata.json
        cfg      : hiperparametry (domyślnie config.LGBM)

    Returns:
        (model, [results_raw, results_calibrated]) — wytrenowany model + metryki
        na zbiorze testowym (surowy argmax oraz po kalibracji progów)
    """
    if cfg is None:
        cfg = config.LGBM

    output_dir = config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = _load_data(
        metadata, config.DATA_DIR
    )

    # ---- Model ----
    model = LGBMClassifier(
        objective         = "multiclass",
        num_class         = 4,
        n_estimators      = cfg["n_estimators"],
        learning_rate     = cfg["learning_rate"],
        num_leaves        = cfg["num_leaves"],
        max_depth         = cfg["max_depth"],
        min_child_samples = cfg["min_child_samples"],
        subsample         = cfg["subsample"],
        colsample_bytree  = cfg["colsample_bytree"],
        reg_alpha         = cfg["reg_alpha"],
        reg_lambda        = cfg["reg_lambda"],
        class_weight      = "balanced",   # obsługa niezbalansowania
        random_state      = cfg["seed"],
        n_jobs            = -1,
        verbosity         = -1,           # wycisza warnings LightGBM
    )

    logger.info(
        f"Start treningu LightGBM | n_estimators={cfg['n_estimators']} | "
        f"num_leaves={cfg['num_leaves']} | early_stopping={cfg['early_stopping_rounds']}"
    )

    model.fit(
        X_train,
        y_train,
        eval_set            = [(X_val, y_val)],
        categorical_feature = metadata["cat_cols"],
        callbacks           = [
            early_stopping(cfg["early_stopping_rounds"], verbose=True),
            log_evaluation(cfg["verbose"]),
        ],
    )

    logger.info(f"Najlepsza iteracja: {model.best_iteration_}")

    # ---- Zapis modelu ----
    model_txt_path = output_dir / "lgbm_model.txt"
    model.booster_.save_model(str(model_txt_path))
    logger.info(f"LightGBM booster zapisany: {model_txt_path}")

    # Zapis też jako joblib (łatwiejszy do załadowania w GUI)
    model_pkl_path = output_dir / "lgbm_model.joblib"
    joblib.dump(model, model_pkl_path)
    logger.info(f"LightGBM joblib zapisany: {model_pkl_path}")

    # ---- Feature importance ----
    _plot_feature_importance(model, feature_cols, output_dir)

    # ---- Kalibracja progów decyzyjnych (strojona na walidacji) ----
    logger.info("Kalibracja progów decyzyjnych na zbiorze walidacyjnym...")
    val_proba = model.predict_proba(X_val)
    multipliers, val_f1_raw, val_f1_cal = tune_class_multipliers(y_val, val_proba)
    save_multipliers(
        output_dir / "lgbm_calibration.json",
        multipliers,
        info={
            "val_macro_f1_raw":        round(val_f1_raw, 4),
            "val_macro_f1_calibrated": round(val_f1_cal, 4),
        },
    )

    # ---- Ewaluacja ----
    logger.info("Ewaluacja LightGBM na zbiorze testowym...")
    y_proba = model.predict_proba(X_test)

    results_raw = evaluate_model(
        y_true     = y_test,
        y_pred     = y_proba.argmax(axis=1),
        y_proba    = y_proba,
        model_name = "LightGBM",
        output_dir = str(output_dir),
    )
    results_cal = evaluate_model(
        y_true     = y_test,
        y_pred     = apply_multipliers(y_proba, multipliers),
        y_proba    = y_proba,
        model_name = "LightGBM + kalibracja",
        output_dir = str(output_dir),
    )

    return model, [results_raw, results_cal]
