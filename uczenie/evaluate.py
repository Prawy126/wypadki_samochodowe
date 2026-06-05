"""
Wspólny moduł ewaluacji dla obu modeli (MLP i LightGBM).
Generuje metryki, raporty tekstowe i wykresy confusion matrix.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# Etykiety osi na confusion matrix
_CLASS_LABELS = [
    "Sev.1\n(Łagodny)",
    "Sev.2\n(Umiark.)",
    "Sev.3\n(Poważny)",
    "Sev.4\n(Krytyczny)",
]


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "Model",
    output_dir: str = "uczenie/outputs",
    save_plots: bool = True,
) -> dict:
    """
    Oblicza i wyświetla pełny zestaw metryk dla klasyfikacji 4-klasowej.

    Args:
        y_true      : tablice prawdziwych etykiet (0-3)
        y_pred      : tablice predykcji (0-3)
        y_proba     : opcjonalne prawdopodobieństwa [N, 4] (do ROC-AUC)
        model_name  : nazwa modelu do wyświetlenia i nazw plików
        output_dir  : katalog, gdzie zapisać wykresy
        save_plots  : czy zapisywać confusion matrix do pliku PNG

    Returns:
        dict z metrykami
    """
    os.makedirs(output_dir, exist_ok=True)

    acc         = accuracy_score(y_true, y_pred)
    macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    kappa       = cohen_kappa_score(y_true, y_pred)

    results = {
        "model":       model_name,
        "accuracy":    round(float(acc),         4),
        "macro_f1":    round(float(macro_f1),    4),
        "weighted_f1": round(float(weighted_f1), 4),
        "cohen_kappa": round(float(kappa),       4),
    }

    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
            results["roc_auc_macro"] = round(float(roc_auc), 4)
        except ValueError as e:
            logger.warning(f"ROC-AUC skipped: {e}")

    # ---- Raport tekstowy ----
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  WYNIKI MODELU: {model_name}")
    print(sep)
    print(f"  Accuracy:        {acc:.4f}")
    print(f"  Macro F1:        {macro_f1:.4f}")
    print(f"  Weighted F1:     {weighted_f1:.4f}")
    print(f"  Cohen's Kappa:   {kappa:.4f}")
    if "roc_auc_macro" in results:
        print(f"  ROC-AUC (macro): {results['roc_auc_macro']:.4f}")
    print()
    print(classification_report(
        y_true, y_pred,
        target_names=["Sev.1", "Sev.2", "Sev.3", "Sev.4"],
        zero_division=0,
    ))

    # ---- Confusion matrix ----
    if save_plots:
        _save_confusion_matrix(y_true, y_pred, model_name, output_dir, results)

    return results


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: str,
    results: dict,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"{model_name}  |  Acc={results['accuracy']:.4f}  "
        f"MacroF1={results['macro_f1']:.4f}  Kappa={results['cohen_kappa']:.4f}",
        fontsize=12, fontweight="bold",
    )

    # Bezwzględne liczby
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
        xticklabels=_CLASS_LABELS, yticklabels=_CLASS_LABELS,
    )
    axes[0].set_title("Confusion Matrix (liczby)")
    axes[0].set_ylabel("Prawdziwa klasa")
    axes[0].set_xlabel("Predykowana klasa")

    # Znormalizowana
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
        xticklabels=_CLASS_LABELS, yticklabels=_CLASS_LABELS,
        vmin=0.0, vmax=1.0,
    )
    axes[1].set_title("Confusion Matrix (recall per class)")
    axes[1].set_ylabel("Prawdziwa klasa")
    axes[1].set_xlabel("Predykowana klasa")

    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    plot_path = os.path.join(output_dir, f"confusion_matrix_{safe_name}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    results["confusion_matrix_path"] = plot_path
    logger.info(f"Confusion matrix saved: {plot_path}")


def compare_models(results_list: list, output_dir: str = "uczenie/outputs") -> None:
    """
    Wyświetla tabelę porównawczą wszystkich modeli i zapisuje wyniki do JSON.

    Args:
        results_list : lista dictów zwróconych przez evaluate_model()
        output_dir   : katalog wyjściowy
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = ["accuracy", "macro_f1", "weighted_f1", "cohen_kappa", "roc_auc_macro"]

    sep = "=" * 72
    print(f"\n{sep}")
    print("  PORÓWNANIE MODELI")
    print(sep)

    col_w = 22
    header = f"{'Metryka':<20}" + "".join(
        f"{r['model']:>{col_w}}" for r in results_list
    )
    print(header)
    print("-" * 72)

    for m in metrics:
        values = []
        for r in results_list:
            v = r.get(m)
            values.append(f"{v:>{col_w}.4f}" if isinstance(v, float) else f"{'N/A':>{col_w}}")
        print(f"{m:<20}" + "".join(values))

    print(sep + "\n")

    # Zapis do JSON
    summary_path = os.path.join(output_dir, "results_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=4, ensure_ascii=False)
    print(f"Wyniki zapisane: {summary_path}")
