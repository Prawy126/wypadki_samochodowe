"""
Punkt wejścia — uruchamianie modeli klasyfikacji wypadków.

Użycie:
    python main.py                        # oba modele
    python main.py --model mlp            # tylko MLP
    python main.py --model lgbm           # tylko LightGBM
    python main.py --model both           # oba + porównanie (domyślnie)
    python main.py --sanity-check         # sprawdza 1 batch MLP bez pełnego treningu
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wczytanie metadanych
# ---------------------------------------------------------------------------

def load_metadata(data_dir: Path) -> dict:
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono {meta_path}. "
            "Uruchom najpierw preprocessing/preprocessing.py"
        )
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    logger.info(f"Metadane wczytane z: {meta_path}")
    logger.info(
        f"Cechy: {metadata['num_features_count']} num + "
        f"{metadata['cat_features_count']} cat + "
        f"{metadata['bin_features_count']} bin = "
        f"{metadata['total_features_count']} łącznie"
    )
    return metadata


# ---------------------------------------------------------------------------
# Sanity check — 1 batch przez MLP
# ---------------------------------------------------------------------------

def sanity_check(metadata: dict) -> None:
    """Sprawdza forward pass MLP na 1 batchu bez pełnego treningu."""
    import torch
    from uczenie.model_mlp import EntityEmbeddingMLP
    from uczenie.train_mlp import get_device

    logger.info("=== SANITY CHECK MLP ===")
    device = get_device()

    model = EntityEmbeddingMLP(
        embedding_dims = metadata["embedding_dims"],
        cat_col_order  = metadata["cat_cols"],
        num_dim        = len(metadata["num_cols"]),
        bin_dim        = len(metadata["bin_cols"]),
    ).to(device)

    B = 8  # mały batch
    cat_t = torch.zeros(B, len(metadata["cat_cols"]), dtype=torch.long).to(device)
    num_t = torch.randn(B, len(metadata["num_cols"])).to(device)
    bin_t = torch.zeros(B, len(metadata["bin_cols"])).to(device)

    with torch.no_grad():
        logits = model(cat_t, num_t, bin_t)

    assert logits.shape == (B, 4), f"Oczekiwano [8, 4], got {logits.shape}"
    logger.info(f"✓ Forward pass OK — logits shape: {logits.shape}")
    logger.info(f"  Parametry modelu:   {model.count_parameters():,}")
    logger.info(f"  Wymiar wejściowy:   {model.config['input_dim']}")
    logger.info(f"  Wymiar embeddings:  {model.config['embed_out_dim']}")
    logger.info("=== SANITY CHECK PASSED ===")


# ---------------------------------------------------------------------------
# Główna funkcja
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Trening modeli klasyfikacji ciężkości wypadków.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady:
  python main.py                   # oba modele sekwencyjnie
  python main.py --model mlp       # tylko MLP z Entity Embeddings
  python main.py --model lgbm      # tylko LightGBM
  python main.py --sanity-check    # test forward pass (bez treningu)
        """,
    )
    parser.add_argument(
        "--model",
        choices=["mlp", "lgbm", "both"],
        default="both",
        help="Który model trenować (domyślnie: both)",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Uruchom tylko sanity check MLP i zakończ",
    )
    args = parser.parse_args()

    # Wczytaj metadane
    from uczenie import config
    metadata = load_metadata(config.DATA_DIR)

    # Sanity check
    if args.sanity_check:
        sanity_check(metadata)
        return

    results_list = []

    # ---- MLP ----
    if args.model in ("mlp", "both"):
        logger.info("\n" + "=" * 60)
        logger.info("  TRENING: MLP z Entity Embeddings")
        logger.info("=" * 60)
        from uczenie.train_mlp import train_mlp
        _, mlp_results = train_mlp(metadata)
        results_list.extend(mlp_results)

    # ---- LightGBM ----
    if args.model in ("lgbm", "both"):
        logger.info("\n" + "=" * 60)
        logger.info("  TRENING: LightGBM")
        logger.info("=" * 60)
        from uczenie.model_lgbm import train_lgbm
        _, lgbm_results = train_lgbm(metadata)
        results_list.extend(lgbm_results)

    # ---- Porównanie ----
    if len(results_list) > 1:
        from uczenie.evaluate import compare_models
        compare_models(results_list, output_dir=str(config.OUTPUT_DIR))

    logger.info("\nGotowe! Wyniki zapisane w: uczenie/outputs/")


if __name__ == "__main__":
    main()
