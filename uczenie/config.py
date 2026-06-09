"""
Centralna konfiguracja projektu.
Zmień hiperparametry tutaj — bez dotykania kodu treningowego.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Ścieżki
# ---------------------------------------------------------------------------
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "preprocessing" / "data"
OUTPUT_DIR = ROOT_DIR / "uczenie" / "outputs"

# ---------------------------------------------------------------------------
# MLP z Entity Embeddings
# ---------------------------------------------------------------------------
MLP = dict(
    # Mniej warstw/parametrów = mniejsza skłonność do uczenia na pamięć
    hidden_dims  = [512, 256, 128],  # wymiary warstw ukrytych (było [512,512,256,128])
    dropout      = 0.4,              # mocniejsza regularyzacja (było 0.25)
    batch_norm   = True,
    lr           = 1e-3,
    batch_size   = 2048,
    max_epochs   = 100,
    patience     = 7,        # Early Stopping — szybsze zatrzymanie (było 12)
    lr_patience  = 3,        # ReduceLROnPlateau — szybsza redukcja lr (było 5)
    lr_factor    = 0.5,
    weight_decay = 5e-4,     # silniejszy L2 (było 1e-4)
    seed         = 42,
    num_workers  = 4,        # DataLoader workers (Linux)

    # ---- Balans klas (kluczowe dla macro F1 vs accuracy) ----
    # Pełna odwrotność częstości (power=1.0) daje wagi do ~80x i rozwala
    # accuracy. power<1.0 "spłaszcza" wagi: 0.0 = brak wag, 0.5 = sqrt-balanced.
    # 0.3-0.5 to zwykle najlepszy kompromis acc <-> macro_f1.
    class_weight_power = 0.5,
    class_weight_cap   = 10.0,   # górny limit wagi po normalizacji (None = brak)

    # label_smoothing wprost ogranicza overconfidence → mniejszy rozjazd
    # train/val loss. To główny środek na przeuczenie tutaj.
    label_smoothing    = 0.1,
)

# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------
LGBM = dict(
    n_estimators          = 8000,
    learning_rate         = 0.02,
    num_leaves            = 127,
    max_depth             = -1,
    min_child_samples     = 20,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    reg_alpha             = 0.1,
    reg_lambda            = 0.1,
    early_stopping_rounds = 100,
    verbose               = 200,
    seed                  = 42,
)

# ---------------------------------------------------------------------------
# Etykiety klas (używane przez GUI i raportowanie)
# ---------------------------------------------------------------------------
SEVERITY_LABELS = {
    0: "Severity 1 — Łagodny",
    1: "Severity 2 — Umiarkowany",
    2: "Severity 3 — Poważny",
    3: "Severity 4 — Krytyczny",
}

SEVERITY_COLORS = {
    0: "#4CAF50",   # zielony
    1: "#FF9800",   # pomarańczowy
    2: "#F44336",   # czerwony
    3: "#7B0000",   # ciemnoczerwony
}
