"""
Moduł inferencji — zaprojektowany pod integrację z GUI.

Klasy:
    AccidentPredictor   — ładuje wytrenowany MLP i udostępnia proste API predict()
    LGBMPredictor       — analogiczne API dla modelu LightGBM

Użycie (GUI):
    from uczenie.predictor import AccidentPredictor

    predictor = AccidentPredictor.from_checkpoint("uczenie/outputs/best_mlp.pt")
    result = predictor.predict(row_dict)
    # {'class': 1, 'label': 'Severity 2 — Umiarkowany', 'probabilities': [...], 'confidence': 0.87}
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import torch

from .calibration import load_multipliers
from .model_mlp import EntityEmbeddingMLP
from . import config

logger = logging.getLogger(__name__)


def _calibrated_result(proba: np.ndarray, multipliers: Optional[np.ndarray]) -> Dict:
    """
    Buduje wynik predykcji. Gdy dostępne są mnożniki kalibracji, klasa jest
    wybierana z przeważonych prawdopodobieństw (przesunięta granica decyzyjna),
    a zwracane prawdopodobieństwa są renormalizowane — spójne z wybraną klasą.
    """
    proba = np.asarray(proba, dtype=np.float64)
    if multipliers is not None:
        proba = proba * multipliers
        proba = proba / proba.sum()

    pred_class = int(np.argmax(proba))
    return {
        "class":         pred_class,
        "label":         config.SEVERITY_LABELS[pred_class],
        "probabilities": proba.tolist(),
        "confidence":    float(proba.max()),
    }


# ---------------------------------------------------------------------------
# MLP Predictor
# ---------------------------------------------------------------------------

class AccidentPredictor:
    """
    Wysokopoziomowy interfejs inferencji dla modelu MLP z Entity Embeddings.

    Zaprojektowany pod GUI — przyjmuje dict z wartościami cech (preprocessed)
    i zwraca słownik z predykcją i prawdopodobieństwami.
    """

    def __init__(
        self,
        model: EntityEmbeddingMLP,
        metadata: dict,
        device: torch.device,
        multipliers: Optional[np.ndarray] = None,
    ):
        self.model       = model.to(device)
        self.model.eval()
        self.metadata    = metadata
        self.device      = device
        self.multipliers = multipliers

        self.num_cols = metadata["num_cols"]
        self.cat_cols = metadata["cat_cols"]
        self.bin_cols = metadata["bin_cols"]

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
    ) -> "AccidentPredictor":
        """
        Wczytuje model z pliku .pt (checkpointu z train_mlp.py).

        Args:
            checkpoint_path : ścieżka do best_mlp.pt
            device          : None → auto-detect (CUDA lub CPU)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint   = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_config = checkpoint["model_config"]
        metadata     = checkpoint["metadata"]

        model = EntityEmbeddingMLP(
            embedding_dims = model_config["embedding_dims"],
            cat_col_order  = model_config["cat_col_order"],
            num_dim        = model_config["num_dim"],
            bin_dim        = model_config["bin_dim"],
            hidden_dims    = model_config["hidden_dims"],
            dropout        = model_config["dropout"],
            num_classes    = model_config["num_classes"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        multipliers = load_multipliers(
            Path(checkpoint_path).parent / "mlp_calibration.json"
        )
        if multipliers is not None:
            logger.info(f"Kalibracja progów MLP: w={np.round(multipliers, 3).tolist()}")

        logger.info(
            f"Załadowano checkpoint MLP: epoch={checkpoint['epoch']}, "
            f"val_macro_f1={checkpoint['best_val_f1']:.4f}, device={device}"
        )
        return cls(model, metadata, device, multipliers=multipliers)

    def predict(self, row: Dict) -> Dict:
        """
        Predykcja dla jednego przykładu.

        Args:
            row : dict z kluczami = num_cols + cat_cols + bin_cols
                  Wartości jak w przetworzonym CSV:
                    - numeryczne: float (RobustScaler)
                    - kategoryczne: int (integer ID z categorical_mappings)
                    - binarne: 0 lub 1

        Returns:
            {
                'class':         int   — klasa 0-3,
                'label':         str   — opis słowny (np. 'Severity 2 — Umiarkowany'),
                'probabilities': list  — [p_class0, p_class1, p_class2, p_class3],
                'confidence':    float — max(probabilities),
            }
        """
        num_arr = np.array([[row[c] for c in self.num_cols]], dtype=np.float32)
        cat_arr = np.array([[row[c] for c in self.cat_cols]], dtype=np.int64)
        bin_arr = np.array([[row[c] for c in self.bin_cols]], dtype=np.float32)

        num_t = torch.tensor(num_arr).to(self.device)
        cat_t = torch.tensor(cat_arr).to(self.device)
        bin_t = torch.tensor(bin_arr).to(self.device)

        proba = self.model.predict_proba(cat_t, num_t, bin_t)[0].cpu().numpy()
        return _calibrated_result(proba, self.multipliers)

    def predict_batch(self, rows: List[Dict]) -> List[Dict]:
        """Predykcja dla listy przykładów (np. z DataFrame.to_dict('records'))."""
        return [self.predict(row) for row in rows]

    def get_metadata(self) -> dict:
        """
        Zwraca metadane modelu — przydatne do budowania formularza GUI.
        Zawiera listy kolumn i wymiary embeddingów.
        """
        return {
            "num_cols":      self.num_cols,
            "cat_cols":      self.cat_cols,
            "bin_cols":      self.bin_cols,
            "embedding_dims": self.metadata.get("embedding_dims", {}),
            "severity_labels": config.SEVERITY_LABELS,
            "severity_colors": config.SEVERITY_COLORS,
        }


# ---------------------------------------------------------------------------
# LightGBM Predictor
# ---------------------------------------------------------------------------

class LGBMPredictor:
    """
    Wysokopoziomowy interfejs inferencji dla modelu LightGBM.
    Takie samo API jak AccidentPredictor — wymienne w GUI.
    """

    def __init__(self, model, metadata: dict, multipliers: Optional[np.ndarray] = None):
        self.model       = model
        self.metadata    = metadata
        self.multipliers = multipliers

        self.num_cols = metadata["num_cols"]
        self.cat_cols = metadata["cat_cols"]
        self.bin_cols = metadata["bin_cols"]
        self.feature_cols = self.num_cols + self.cat_cols + self.bin_cols

    @classmethod
    def from_joblib(
        cls,
        model_path: Union[str, Path],
        metadata: dict,
    ) -> "LGBMPredictor":
        """
        Wczytuje model z pliku .joblib (zapisanego przez train_lgbm()).

        Args:
            model_path : ścieżka do lgbm_model.joblib
            metadata   : słownik z metadata.json
        """
        model = joblib.load(model_path)
        multipliers = load_multipliers(
            Path(model_path).parent / "lgbm_calibration.json"
        )
        if multipliers is not None:
            logger.info(f"Kalibracja progów LGBM: w={np.round(multipliers, 3).tolist()}")
        logger.info(f"Załadowano model LightGBM: {model_path}")
        return cls(model, metadata, multipliers=multipliers)

    def predict(self, row: Dict) -> Dict:
        """Predykcja dla jednego przykładu (identyczne API jak AccidentPredictor)."""
        import pandas as pd

        row_df = pd.DataFrame([{col: row[col] for col in self.feature_cols}])

        # Rzutowanie cat cols na int
        for col in self.cat_cols:
            row_df[col] = row_df[col].astype(int)

        proba = self.model.predict_proba(row_df)[0]
        return _calibrated_result(proba, self.multipliers)

    def predict_batch(self, rows: List[Dict]) -> List[Dict]:
        """Predykcja wsadowa (bardziej wydajna niż pętla po predict())."""
        import pandas as pd

        rows_df = pd.DataFrame([{col: r[col] for col in self.feature_cols} for r in rows])
        for col in self.cat_cols:
            rows_df[col] = rows_df[col].astype(int)

        probas = self.model.predict_proba(rows_df)
        return [_calibrated_result(probas[i], self.multipliers) for i in range(len(rows))]

    def get_metadata(self) -> dict:
        """Zwraca metadane modelu — identyczne API jak AccidentPredictor.get_metadata()."""
        return {
            "num_cols":       self.num_cols,
            "cat_cols":       self.cat_cols,
            "bin_cols":       self.bin_cols,
            "severity_labels": config.SEVERITY_LABELS,
            "severity_colors": config.SEVERITY_COLORS,
        }
