"""
Pętla treningowa MLP z Entity Embeddings.

Funkcje:
    get_device()    — auto-detect GPU (CUDA) lub CPU
    train_mlp()     — pełny trening, zwraca (model, results_dict)
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .calibration import apply_multipliers, save_multipliers, tune_class_multipliers
from .dataset import AccidentDataset
from .evaluate import evaluate_model
from .model_mlp import EntityEmbeddingMLP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wykrywanie urządzenia
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Automatycznie wybiera najlepsze dostępne urządzenie:
      - CUDA (NVIDIA GPU, np. RTX 4050) — pełna obsługa AMP
      - CPU   — fallback (np. zintegrowana AMD Radeon)

    Nie wymaga żadnej konfiguracji — działa out-of-the-box.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logger.info(f"Używam GPU: {gpu_name}  ({vram:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        logger.info("GPU niedostępne — trening na CPU (bez AMP).")
    return device


# ---------------------------------------------------------------------------
# Wagi klas (power scaling)
# ---------------------------------------------------------------------------

def compute_class_weights(
    class_counts: dict,
    num_classes: int = 4,
    power: float = 0.5,
    cap: Optional[float] = None,
) -> np.ndarray:
    """
    Liczy złagodzone wagi klas z rozkładu treningowego.

    Pełna odwrotność częstości (power=1.0) przy silnym imbalanse daje wagi
    rzędu kilkudziesięciu i zmusza model do ignorowania klasy większościowej
    → katastrofalny spadek accuracy. Skalowanie potęgowe spłaszcza te wagi:

        w_i = (N / (K * count_i)) ** power

    następnie normalizujemy do średniej 1.0 (stabilna skala loss) i opcjonalnie
    przycinamy do 'cap'.

    Args:
        class_counts : {klasa: liczność} z treningu
        num_classes  : liczba klas (K)
        power        : 0.0 = brak wag, 0.5 = sqrt-balanced, 1.0 = pełny balanced
        cap          : górny limit wagi po normalizacji (None = brak)

    Returns:
        np.ndarray [K] z wagami float32
    """
    counts = np.array(
        [max(class_counts.get(i, 0), 1) for i in range(num_classes)],
        dtype=np.float64,
    )
    n_total = counts.sum()

    # power=0 → wszystkie wagi równe 1 (brak ważenia)
    raw = (n_total / (num_classes * counts)) ** power
    weights = raw / raw.mean()  # normalizacja do średniej 1.0

    if cap is not None:
        weights = np.minimum(weights, cap)

    return weights.astype(np.float32)


def validate_year_feature_schema(metadata: dict, train_ds: AccidentDataset) -> None:
    """
    Sprawdza spójność traktowania kolumny Year.

    Year ma być cechą numeryczną. Jeśli wygląda jak zakodowana kategoria
    (np. 1..8), zatrzymujemy trening z czytelnym komunikatem, żeby uniknąć
    cichego driftu między preprocessingiem i uczeniem.
    """
    num_cols = metadata.get("num_cols", [])
    cat_cols = metadata.get("cat_cols", [])

    if "Year" in cat_cols or "Year" not in num_cols:
        raise ValueError(
            "Niespójna schema: 'Year' musi być w num_cols i nie może być w cat_cols. "
            "Uruchom preprocessing ponownie."
        )

    year_idx = num_cols.index("Year")
    year_vals = train_ds.num[:, year_idx].cpu().numpy()

    # Heurystyka driftu: Year zakodowany jako kategoria zazwyczaj ma mały zakres
    # dodatnich liczb całkowitych (np. 1..8).
    is_integer_like = np.allclose(year_vals, np.round(year_vals), atol=1e-6)
    if is_integer_like:
        year_min = float(year_vals.min())
        year_max = float(year_vals.max())
        year_unique = int(np.unique(year_vals).size)
        if 0.0 <= year_min and year_max <= 50.0 and year_unique <= 50:
            raise ValueError(
                "Wykryto drift danych: kolumna 'Year' wygląda na zakodowaną "
                f"kategorycznie (min={year_min:.1f}, max={year_max:.1f}, unique={year_unique}). "
                "Uruchom preprocessing/preprocessing.py ponownie, aby zapisać "
                "spójne artefakty (Year jako cecha numeryczna)."
            )


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Zatrzymuje trening gdy val_loss nie poprawia się przez 'patience' epok."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ---------------------------------------------------------------------------
# Epoka treningowa
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    amp_scaler: Optional[torch.cuda.amp.GradScaler],
) -> Tuple[float, float]:
    """Jedna epoka treningu. Zwraca (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct    = 0
    n          = 0

    for batch in loader:
        cat       = batch["cat"].to(device, non_blocking=True)
        num       = batch["num"].to(device, non_blocking=True)
        bin_feats = batch["bin"].to(device, non_blocking=True)
        labels    = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if amp_scaler is not None:
            # Mixed precision — tylko na CUDA
            with torch.cuda.amp.autocast():
                logits = model(cat, num, bin_feats)
                loss   = criterion(logits, labels)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            logits = model(cat, num, bin_feats)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs          = labels.size(0)
        total_loss += loss.item() * bs
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        n          += bs

    return total_loss / n, correct / n


# ---------------------------------------------------------------------------
# Epoka walidacyjna
# ---------------------------------------------------------------------------

@torch.no_grad()
def _val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Walidacja. Zwraca (avg_loss, accuracy, macro_f1)."""
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for batch in loader:
        cat       = batch["cat"].to(device, non_blocking=True)
        num       = batch["num"].to(device, non_blocking=True)
        bin_feats = batch["bin"].to(device, non_blocking=True)
        labels    = batch["label"].to(device, non_blocking=True)

        logits = model(cat, num, bin_feats)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n          = len(all_labels)
    avg_loss   = total_loss / n
    acc        = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    macro_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, macro_f1


# ---------------------------------------------------------------------------
# Zbieranie predykcji z DataLoadera
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_probas(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Zwraca (labels [N], probas [N, 4]) dla całego loadera."""
    model.eval()
    all_labels = []
    all_probas = []

    for batch in loader:
        cat       = batch["cat"].to(device, non_blocking=True)
        num       = batch["num"].to(device, non_blocking=True)
        bin_feats = batch["bin"].to(device, non_blocking=True)

        logits = model(cat, num, bin_feats)
        all_probas.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_labels.append(batch["label"].numpy())

    return np.concatenate(all_labels), np.concatenate(all_probas)


# ---------------------------------------------------------------------------
# Wykres historii treningu
# ---------------------------------------------------------------------------

def _plot_history(history: list, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("MLP — Historia treningu", fontsize=13, fontweight="bold")

    # Loss
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train")
    axes[0].plot(epochs, [h["val_loss"]   for h in history], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoka")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, [h["train_acc"] for h in history], label="Train")
    axes[1].plot(epochs, [h["val_acc"]   for h in history], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoka")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Macro F1
    axes[2].plot(epochs, [h["val_macro_f1"] for h in history], color="green")
    axes[2].set_title("Val Macro F1")
    axes[2].set_xlabel("Epoka")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "mlp_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Wykres historii zapisany: {output_dir / 'mlp_training_curves.png'}")


# ---------------------------------------------------------------------------
# Główna funkcja treningowa
# ---------------------------------------------------------------------------

def train_mlp(
    metadata: dict,
    cfg: Optional[dict] = None,
) -> Tuple[EntityEmbeddingMLP, dict]:
    """
    Trenuje MLP z Entity Embeddings.

    Args:
        metadata : słownik z metadata.json
        cfg      : hiperparametry (domyślnie config.MLP)

    Returns:
        (model, [results_raw, results_calibrated]) — najlepszy model + metryki
        na zbiorze testowym (surowy argmax oraz po kalibracji progów)
    """
    if cfg is None:
        cfg = config.MLP

    output_dir = config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device    = get_device()
    data_dir  = config.DATA_DIR

    # ---- Dane ----
    logger.info("Wczytywanie zbiorów danych...")
    train_ds = AccidentDataset(data_dir / "train_preprocessed.csv.gz", metadata)
    val_ds   = AccidentDataset(data_dir / "val_preprocessed.csv.gz",   metadata)
    test_ds  = AccidentDataset(data_dir / "test_preprocessed.csv.gz",  metadata)

    logger.info(
        f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}  |  Test: {len(test_ds):,}"
    )
    logger.info(f"Rozkład klas (train): {train_ds.class_counts}")
    validate_year_feature_schema(metadata, train_ds)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=pin,
        persistent_workers=(cfg["num_workers"] > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=pin,
        persistent_workers=(cfg["num_workers"] > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=pin,
        persistent_workers=(cfg["num_workers"] > 0),
    )

    # ---- Model ----
    model = EntityEmbeddingMLP(
        embedding_dims=metadata["embedding_dims"],
        cat_col_order=metadata["cat_cols"],
        num_dim=train_ds.num_dim,
        bin_dim=train_ds.bin_dim,
        hidden_dims=cfg["hidden_dims"],
        dropout=cfg["dropout"],
    ).to(device)

    logger.info(f"Parametry modelu: {model.count_parameters():,}")
    logger.info(f"Wymiar wejściowy: {model.config['input_dim']}")

    # ---- Loss z wagami klas (złagodzonymi) ----
    weights_np = compute_class_weights(
        train_ds.class_counts,
        num_classes=4,
        power=cfg.get("class_weight_power", 0.5),
        cap=cfg.get("class_weight_cap", None),
    )
    class_weights = torch.tensor(weights_np, dtype=torch.float32).to(device)
    logger.info(
        f"Wagi klas (power={cfg.get('class_weight_power', 0.5)}, "
        f"cap={cfg.get('class_weight_cap', None)}): "
        f"{ {i: round(float(w), 3) for i, w in enumerate(weights_np)} }"
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg.get("label_smoothing", 0.0),
    )

    # ---- Optymalizator i scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",          # maksymalizujemy val_macro_f1
        factor=cfg["lr_factor"],
        patience=cfg["lr_patience"],
    )

    # AMP Scaler — tylko na CUDA (brak obsługi na CPU/zintegrowanej AMD)
    amp_scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ---- Pętla treningowa ----
    early_stopping    = EarlyStopping(patience=cfg["patience"])
    best_val_f1       = 0.0
    best_model_path   = output_dir / "best_mlp.pt"
    history           = []

    logger.info(
        f"Start treningu | urządzenie={device} | max_epochs={cfg['max_epochs']} | "
        f"batch={cfg['batch_size']} | AMP={'tak' if amp_scaler else 'nie'}"
    )

    for epoch in range(1, cfg["max_epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = _train_epoch(
            model, train_loader, optimizer, criterion, device, amp_scaler
        )
        val_loss, val_acc, val_f1 = _val_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(val_f1)   # scheduler śledzi F1, nie loss

        elapsed    = time.time() - t0
        lr_current = optimizer.param_groups[0]["lr"]

        log_msg = (
            f"Epoch {epoch:03d}/{cfg['max_epochs']} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} | "
            f"lr={lr_current:.2e} | {elapsed:.1f}s"
        )
        logger.info(log_msg)

        history.append({
            "epoch":        epoch,
            "train_loss":   round(train_loss,  5),
            "train_acc":    round(train_acc,    5),
            "val_loss":     round(val_loss,     5),
            "val_acc":      round(val_acc,      5),
            "val_macro_f1": round(val_f1,       5),
            "lr":           lr_current,
        })

        # Zapis najlepszego modelu (wg val macro F1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "model_config":     model.config,
                    "best_val_f1":      best_val_f1,
                    "metadata":         metadata,
                    "history":          history,
                },
                best_model_path,
            )
            logger.info(f"  ✓ Najlepszy model zapisany (val_macro_f1={best_val_f1:.4f})")

        if early_stopping.step(-val_f1):  # negujemy — EarlyStopping szuka minimum
            logger.info(f"Early stopping po epoce {epoch} (patience={cfg['patience']})")
            break

    # ---- Historia i wykresy ----
    with open(output_dir / "mlp_training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    _plot_history(history, output_dir)

    # ---- Ewaluacja na zbiorze testowym ----
    logger.info("Wczytywanie najlepszego modelu do ewaluacji na test set...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Kalibracja progów decyzyjnych — strojona na walidacji (nie na test!)
    logger.info("Kalibracja progów decyzyjnych na zbiorze walidacyjnym...")
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

    results_raw = evaluate_model(
        y_true     = test_labels,
        y_pred     = test_probas.argmax(axis=1),
        y_proba    = test_probas,
        model_name = "MLP (Entity Embeddings)",
        output_dir = str(output_dir),
    )
    results_cal = evaluate_model(
        y_true     = test_labels,
        y_pred     = apply_multipliers(test_probas, multipliers),
        y_proba    = test_probas,
        model_name = "MLP + kalibracja",
        output_dir = str(output_dir),
    )

    logger.info(f"Trening MLP zakończony. Najlepszy val_macro_f1={best_val_f1:.4f}")
    return model, [results_raw, results_cal]
