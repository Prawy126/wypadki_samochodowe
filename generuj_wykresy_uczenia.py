"""
Generowanie wykresów efektów uczenia (na zbiorze testowym):
  1. Macierz błędów (liczby + znormalizowana per klasa) — dla MLP i LightGBM
  2. Precision / Recall / F1 per klasa — dla MLP i LightGBM
  3. Porównanie F1 per klasa: MLP vs LightGBM

Użycie:
    python generuj_wykresy_uczenia.py
    python generuj_wykresy_uczenia.py --output-dir raport_wykresy

Wymaga wytrenowanych modeli w uczenie/outputs/ (best_mlp.pt, lgbm_model.joblib).
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CLASS_LABELS = [
    "Sev.1\n(Łagodny)",
    "Sev.2\n(Umiark.)",
    "Sev.3\n(Poważny)",
    "Sev.4\n(Krytyczny)",
]
CLASS_LABELS_FLAT = ["Sev.1", "Sev.2", "Sev.3", "Sev.4"]
SEVERITY_COLORS = ["#4CAF50", "#FF9800", "#F44336", "#7B0000"]

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})


# ---------------------------------------------------------------------------
# Predykcje z zapisanych modeli
# ---------------------------------------------------------------------------

def get_mlp_predictions(metadata: dict, data_dir: Path, output_dir: Path):
    """Zwraca (y_true, y_pred) dla MLP na zbiorze testowym lub None."""
    import torch
    from torch.utils.data import DataLoader
    from uczenie import config
    from uczenie.dataset import AccidentDataset
    from uczenie.model_mlp import EntityEmbeddingMLP
    from uczenie.train_mlp import get_device

    ckpt_path = output_dir / "best_mlp.pt"
    if not ckpt_path.exists():
        logger.warning(f"Brak checkpointu MLP: {ckpt_path} — pomijam.")
        return None

    device = get_device()
    test_ds = AccidentDataset(data_dir / "test_preprocessed.csv.gz", metadata)
    test_loader = DataLoader(
        test_ds,
        batch_size=config.MLP["batch_size"] * 2,
        shuffle=False,
        num_workers=config.MLP["num_workers"],
    )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    mc = checkpoint["model_config"]
    model = EntityEmbeddingMLP(
        embedding_dims=mc["embedding_dims"],
        cat_col_order=mc["cat_col_order"],
        num_dim=mc["num_dim"],
        bin_dim=mc["bin_dim"],
        hidden_dims=mc["hidden_dims"],
        dropout=mc["dropout"],
        num_classes=mc["num_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                batch["cat"].to(device),
                batch["num"].to(device),
                batch["bin"].to(device),
            )
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            labels.extend(batch["label"].numpy())

    logger.info(f"MLP: predykcje gotowe ({len(labels):,} przykładów)")
    return np.array(labels), np.array(preds)


def get_lgbm_predictions(metadata: dict, data_dir: Path, output_dir: Path):
    """Zwraca (y_true, y_pred) dla LightGBM na zbiorze testowym lub None."""
    from uczenie.model_lgbm import _load_data

    model_path = output_dir / "lgbm_model.joblib"
    if not model_path.exists():
        logger.warning(f"Brak modelu LightGBM: {model_path} — pomijam.")
        return None

    _, _, _, _, X_test, y_test, _ = _load_data(metadata, data_dir)
    model = joblib.load(model_path)
    y_pred = model.predict(X_test).astype(int)

    logger.info(f"LightGBM: predykcje gotowe ({len(y_test):,} przykładów)")
    return y_test, y_pred


# ---------------------------------------------------------------------------
# Wykresy
# ---------------------------------------------------------------------------

def plot_confusion_matrices(y_true, y_pred, model_name: str, out_dir: Path) -> None:
    """Macierz błędów: liczby bezwzględne + znormalizowana (recall per klasa)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    acc = (y_true == y_pred).mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        f"Macierz błędów — {model_name}  (accuracy = {acc:.4f})",
        fontsize=13, fontweight="bold",
    )

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, cbar_kws={"shrink": 0.8},
    )
    axes[0].set_title("Liczby bezwzględne")
    axes[0].set_ylabel("Prawdziwa klasa")
    axes[0].set_xlabel("Predykowana klasa")

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
        vmin=0.0, vmax=1.0, cbar_kws={"shrink": 0.8},
    )
    axes[1].set_title("Znormalizowana (recall per klasa)")
    axes[1].set_ylabel("Prawdziwa klasa")
    axes[1].set_xlabel("Predykowana klasa")

    plt.tight_layout()
    safe = model_name.lower().replace(" ", "_")
    path = out_dir / f"07_macierz_bledow_{safe}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Zapisano: {path}")


def plot_per_class_metrics(y_true, y_pred, model_name: str, out_dir: Path) -> None:
    """Precision / Recall / F1 per klasa — grupowany wykres słupkowy."""
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
    )

    x = np.arange(4)
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_p = ax.bar(x - width, prec, width, label="Precision", color="#4F46E5", edgecolor="black", alpha=0.85)
    bars_r = ax.bar(x,         rec,  width, label="Recall",    color="#06B6D4", edgecolor="black", alpha=0.85)
    bars_f = ax.bar(x + width, f1,   width, label="F1-score",  color="#EC4899", edgecolor="black", alpha=0.85)

    for bars in (bars_p, bars_r, bars_f):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Liczność klas pod etykietami — pokazuje kontekst niezbalansowania
    xtick_labels = [
        f"{CLASS_LABELS_FLAT[i]}\n(n={support[i]:,})" for i in range(4)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Wartość metryki")
    ax.set_title(f"Wyniki per klasa — {model_name} (zbiór testowy)", fontweight="bold", pad=15)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    safe = model_name.lower().replace(" ", "_")
    path = out_dir / f"08_metryki_per_klasa_{safe}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Zapisano: {path}")


def plot_f1_comparison(results: dict, out_dir: Path) -> None:
    """Porównanie F1 per klasa między modelami (jeśli są >= 2)."""
    if len(results) < 2:
        return

    x = np.arange(4)
    width = 0.8 / len(results)
    colors = ["#06B6D4", "#4F46E5", "#EC4899"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, (y_true, y_pred)) in enumerate(results.items()):
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
        )
        offset = (i - (len(results) - 1) / 2) * width
        bars = ax.bar(x + offset, f1, width, label=name,
                      color=colors[i % len(colors)], edgecolor="black", alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1-score")
    ax.set_title("Porównanie F1 per klasa: MLP vs LightGBM (zbiór testowy)", fontweight="bold", pad=15)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = out_dir / "09_porownanie_f1_per_klasa.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Zapisano: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Wykresy efektów uczenia (confusion matrix + metryki per klasa).")
    parser.add_argument("--output-dir", default="raport_wykresy", help="Katalog na wykresy")
    args = parser.parse_args()

    from uczenie import config

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = config.DATA_DIR / "metadata.json"
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    logger.info(f"Metadane: {meta_path}")

    results = {}

    mlp = get_mlp_predictions(metadata, config.DATA_DIR, config.OUTPUT_DIR)
    if mlp is not None:
        results["MLP (Entity Embeddings)"] = mlp

    lgbm = get_lgbm_predictions(metadata, config.DATA_DIR, config.OUTPUT_DIR)
    if lgbm is not None:
        results["LightGBM"] = lgbm

    if not results:
        logger.error("Brak wytrenowanych modeli — najpierw uruchom main.py")
        return

    for name, (y_true, y_pred) in results.items():
        plot_confusion_matrices(y_true, y_pred, name, out_dir)
        plot_per_class_metrics(y_true, y_pred, name, out_dir)

    plot_f1_comparison(results, out_dir)

    logger.info(f"Gotowe! Wykresy zapisane w: {out_dir}/")


if __name__ == "__main__":
    main()
