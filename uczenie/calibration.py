"""
Kalibracja progów decyzyjnych dla klasyfikacji wieloklasowej.

Problem: przy silnym niezbalansowaniu (Sev.2 ≈ 78% danych) argmax(proba)
faworyzuje klasę większościową — klasy mniejszościowe (zwłaszcza Sev.4)
mają niski recall mimo sensownych prawdopodobieństw.

Rozwiązanie: zamiast surowego argmax używamy argmax(proba * w), gdzie
w = [w_0..w_3] to mnożniki per-klasa strojone na zbiorze WALIDACYJNYM
tak, aby zmaksymalizować macro F1. Nie zmienia to modelu ani jego
prawdopodobieństw — przesuwa tylko granicę decyzyjną w stronę klas
mniejszościowych. Tani zabieg (bez retreningu) o dużym wpływie na macro F1.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

# Siatka mnożników przeszukiwana dla każdej klasy (log-skala wokół 1.0)
_DEFAULT_GRID = np.geomspace(0.2, 8.0, 49)


def apply_multipliers(proba: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    """Zwraca predykcje klas: argmax(proba * w)."""
    proba = np.asarray(proba, dtype=np.float64)
    w = np.asarray(multipliers, dtype=np.float64)
    return (proba * w[None, :]).argmax(axis=1)


def tune_class_multipliers(
    y_true: np.ndarray,
    proba: np.ndarray,
    grid: Optional[np.ndarray] = None,
    sweeps: int = 4,
) -> Tuple[np.ndarray, float, float]:
    """
    Stroi mnożniki per-klasa metodą coordinate ascent (maksymalizacja macro F1).

    Args:
        y_true : etykiety walidacyjne [N]
        proba  : prawdopodobieństwa modelu [N, K]
        grid   : kandydaci na mnożnik pojedynczej klasy
        sweeps : liczba pełnych przejść po wszystkich klasach

    Returns:
        (multipliers [K], macro_f1_przed, macro_f1_po) — wszystko na walidacji
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=np.float64)
    num_classes = proba.shape[1]
    if grid is None:
        grid = _DEFAULT_GRID

    w = np.ones(num_classes, dtype=np.float64)
    f1_before = f1_score(y_true, proba.argmax(axis=1), average="macro", zero_division=0)
    best_f1 = f1_before

    for sweep in range(sweeps):
        improved = False
        for k in range(num_classes):
            best_wk = w[k]
            for candidate in grid:
                w_try = w.copy()
                w_try[k] = candidate
                preds = (proba * w_try[None, :]).argmax(axis=1)
                f1 = f1_score(y_true, preds, average="macro", zero_division=0)
                if f1 > best_f1 + 1e-6:
                    best_f1 = f1
                    best_wk = candidate
            if best_wk != w[k]:
                w[k] = best_wk
                improved = True
        logger.info(
            f"Kalibracja sweep {sweep + 1}/{sweeps}: macro_f1={best_f1:.4f}, "
            f"w={np.round(w, 3).tolist()}"
        )
        if not improved:
            break

    # Normalizacja (skala mnożników nie ma znaczenia dla argmax — czytelniejszy zapis)
    w = w / w.min()
    return w, float(f1_before), float(best_f1)


def save_multipliers(
    path: Union[str, Path],
    multipliers: np.ndarray,
    info: Optional[dict] = None,
) -> None:
    """Zapisuje mnożniki + metryki strojenia do pliku JSON."""
    payload = {"multipliers": np.asarray(multipliers, dtype=float).tolist()}
    if info:
        payload.update(info)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f"Mnożniki kalibracji zapisane: {path}")


def load_multipliers(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Wczytuje mnożniki z JSON. Zwraca None gdy plik nie istnieje."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return np.asarray(payload["multipliers"], dtype=np.float64)
