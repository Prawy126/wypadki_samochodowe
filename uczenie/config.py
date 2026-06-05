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
    hidden_dims  = [512, 512, 256, 128],  # wymiary warstw ukrytych
    dropout      = 0.25,
    batch_norm   = True,
    lr           = 1e-3,
    batch_size   = 2048,
    max_epochs   = 100,
    patience     = 12,       # Early Stopping (val_macro_f1)
    lr_patience  = 5,        # ReduceLROnPlateau
    lr_factor    = 0.5,
    weight_decay = 1e-4,
    seed         = 42,
    num_workers  = 4,        # DataLoader workers (Linux)
)

# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------
LGBM = dict(
    n_estimators          = 5000,
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
