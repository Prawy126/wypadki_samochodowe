"""
Skrypt do ewaluacji zapisanych modeli (MLP i LightGBM) bez konieczności ich ponownego uczenia.

Dodatkowo stroi kalibrację progów decyzyjnych na zbiorze walidacyjnym
(maksymalizacja macro F1) i zapisuje mnożniki do outputs/*_calibration.json,
dzięki czemu GUI/predictor również z nich korzysta.

Generuje ostateczne metryki porównawcze oraz plik results_summary.json.
"""

import json
import logging
import sys
from pathlib import Path
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

# Dodanie katalogu głównego projektu do sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from uczenie import config
from uczenie.calibration import apply_multipliers, save_multipliers, tune_class_multipliers
from uczenie.dataset import AccidentDataset
from uczenie.model_mlp import EntityEmbeddingMLP
from uczenie.train_mlp import get_device, _collect_probas
from uczenie.evaluate import evaluate_model, compare_models
from uczenie.model_lgbm import _load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    data_dir = config.DATA_DIR
    output_dir = config.OUTPUT_DIR

    meta_path = data_dir / "metadata.json"
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)

    device = get_device()
    results_list = []

    # 1. Ewaluacja MLP
    mlp_path = output_dir / "best_mlp.pt"
    if mlp_path.exists():
        logger.info("\n" + "=" * 60)
        logger.info("  EWALUACJA: MLP z Entity Embeddings")
        logger.info("=" * 60)
        val_ds  = AccidentDataset(data_dir / "val_preprocessed.csv.gz",  metadata)
        test_ds = AccidentDataset(data_dir / "test_preprocessed.csv.gz", metadata)
        loader_kwargs = dict(
            batch_size=config.MLP["batch_size"] * 2,
            shuffle=False,
            num_workers=0,  # sama inferencja — workery zbędne (unika problemów z /dev/shm)
            pin_memory=(device.type == "cuda"),
        )
        val_loader  = DataLoader(val_ds,  **loader_kwargs)
        test_loader = DataLoader(test_ds, **loader_kwargs)

        checkpoint = torch.load(mlp_path, map_location=device, weights_only=False)
        model_config = checkpoint["model_config"]

        model = EntityEmbeddingMLP(
            embedding_dims = model_config["embedding_dims"],
            cat_col_order  = model_config["cat_col_order"],
            num_dim        = model_config["num_dim"],
            bin_dim        = model_config["bin_dim"],
            hidden_dims    = model_config["hidden_dims"],
            dropout        = model_config["dropout"],
            num_classes    = model_config["num_classes"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Kalibracja progów na walidacji
        logger.info("Kalibracja progów decyzyjnych MLP (zbiór walidacyjny)...")
        val_labels, val_probas = _collect_probas(model, val_loader, device)
        multipliers, val_f1_raw, val_f1_cal = tune_class_multipliers(val_labels, val_probas)
        save_multipliers(
            output_dir / "mlp_calibration.json",
            multipliers,
            info={
                "val_macro_f1_raw":        round(val_f1_raw, 4),
                "val_macro_f1_calibrated": round(val_f1_cal, 4),
            },
        )

        test_labels, test_probas = _collect_probas(model, test_loader, device)

        results_list.append(evaluate_model(
            y_true     = test_labels,
            y_pred     = test_probas.argmax(axis=1),
            y_proba    = test_probas,
            model_name = "MLP (Entity Embeddings)",
            output_dir = str(output_dir),
        ))
        results_list.append(evaluate_model(
            y_true     = test_labels,
            y_pred     = apply_multipliers(test_probas, multipliers),
            y_proba    = test_probas,
            model_name = "MLP + kalibracja",
            output_dir = str(output_dir),
        ))
    else:
        logger.warning(f"Nie znaleziono checkpointu MLP pod: {mlp_path}")

    # 2. Ewaluacja LightGBM
    lgbm_path = output_dir / "lgbm_model.joblib"
    if lgbm_path.exists():
        logger.info("\n" + "=" * 60)
        logger.info("  EWALUACJA: LightGBM")
        logger.info("=" * 60)
        _, _, X_val, y_val, X_test, y_test, _ = _load_data(metadata, data_dir)

        model = joblib.load(lgbm_path)

        # Kalibracja progów na walidacji
        logger.info("Kalibracja progów decyzyjnych LightGBM (zbiór walidacyjny)...")
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

        y_proba = model.predict_proba(X_test)

        results_list.append(evaluate_model(
            y_true     = y_test,
            y_pred     = y_proba.argmax(axis=1),
            y_proba    = y_proba,
            model_name = "LightGBM",
            output_dir = str(output_dir),
        ))
        results_list.append(evaluate_model(
            y_true     = y_test,
            y_pred     = apply_multipliers(y_proba, multipliers),
            y_proba    = y_proba,
            model_name = "LightGBM + kalibracja",
            output_dir = str(output_dir),
        ))
    else:
        logger.warning(f"Nie znaleziono modelu LightGBM pod: {lgbm_path}")

    # 3. Porównanie
    if len(results_list) > 1:
        compare_models(results_list, output_dir=str(output_dir))


if __name__ == "__main__":
    main()
