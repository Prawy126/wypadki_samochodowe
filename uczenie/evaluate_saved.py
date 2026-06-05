"""
Skrypt do ewaluacji zapisanych modeli (MLP i LightGBM) bez konieczności ich ponownego uczenia.
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
from uczenie.dataset import AccidentDataset
from uczenie.model_mlp import EntityEmbeddingMLP
from uczenie.train_mlp import get_device
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
        test_ds = AccidentDataset(data_dir / "test_preprocessed.csv.gz", metadata)
        test_loader = DataLoader(
            test_ds,
            batch_size=config.MLP["batch_size"] * 2,
            shuffle=False,
            num_workers=config.MLP["num_workers"],
            pin_memory=(device.type == "cuda"),
        )
        
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
        
        all_preds = []
        all_labels = []
        all_probas = []
        
        with torch.no_grad():
            for batch in test_loader:
                cat = batch["cat"].to(device)
                num = batch["num"].to(device)
                bin_feats = batch["bin"].to(device)
                labels = batch["label"]
                
                logits = model(cat, num, bin_feats)
                proba = torch.softmax(logits, dim=1).cpu().numpy()
                
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probas.extend(proba)
                
        mlp_results = evaluate_model(
            y_true     = np.array(all_labels),
            y_pred     = np.array(all_preds),
            y_proba    = np.array(all_probas),
            model_name = "MLP (Entity Embeddings)",
            output_dir = str(output_dir),
        )
        results_list.append(mlp_results)
    else:
        logger.warning(f"Nie znaleziono checkpointu MLP pod: {mlp_path}")
        
    # 2. Ewaluacja LightGBM
    lgbm_path = output_dir / "lgbm_model.joblib"
    if lgbm_path.exists():
        logger.info("\n" + "=" * 60)
        logger.info("  EWALUACJA: LightGBM")
        logger.info("=" * 60)
        _, _, _, _, X_test, y_test, _ = _load_data(metadata, data_dir)
        
        model = joblib.load(lgbm_path)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        lgbm_results = evaluate_model(
            y_true     = y_test,
            y_pred     = y_pred.astype(int),
            y_proba    = y_proba,
            model_name = "LightGBM",
            output_dir = str(output_dir),
        )
        results_list.append(lgbm_results)
    else:
        logger.warning(f"Nie znaleziono modelu LightGBM pod: {lgbm_path}")
        
    # 3. Porównanie
    if len(results_list) > 1:
        compare_models(results_list, output_dir=str(output_dir))

if __name__ == "__main__":
    main()
