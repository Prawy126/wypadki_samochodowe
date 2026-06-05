"""
PyTorch Dataset dla przetworzonych danych wypadków.
Wczytuje csv.gz i przechowuje tensory w pamięci (szybki dostęp).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AccidentDataset(Dataset):
    """
    Dataset ładujący plik csv.gz z przetworzonym zbiorem danych.

    Zwraca dict z kluczami:
        - 'num'   : FloatTensor [num_dim]   — cechy numeryczne (scaled)
        - 'cat'   : LongTensor  [num_cats]  — integer IDs dla nn.Embedding
        - 'bin'   : FloatTensor [bin_dim]   — cechy binarne 0/1
        - 'label' : LongTensor  scalar      — klasa 0-3
    """

    def __init__(self, csv_gz_path: str, metadata: dict):
        """
        Args:
            csv_gz_path : ścieżka do pliku *_preprocessed.csv.gz
            metadata    : słownik z metadata.json (num_cols, cat_cols, bin_cols, ...)
        """
        self.num_cols = metadata["num_cols"]
        self.cat_cols = metadata["cat_cols"]
        self.bin_cols = metadata["bin_cols"]
        self.target_col = metadata["target_col"]

        df = pd.read_csv(csv_gz_path, compression="gzip")

        # Konwersja do tensorów raz przy inicjalizacji → szybkie __getitem__
        self.y = torch.tensor(df[self.target_col].values, dtype=torch.long)
        self.num = torch.tensor(
            df[self.num_cols].values.astype(np.float32), dtype=torch.float32
        )
        self.cat = torch.tensor(
            df[self.cat_cols].values.astype(np.int64), dtype=torch.long
        )
        self.bin = torch.tensor(
            df[self.bin_cols].values.astype(np.float32), dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict:
        return {
            "num":   self.num[idx],
            "cat":   self.cat[idx],
            "bin":   self.bin[idx],
            "label": self.y[idx],
        }

    @property
    def num_dim(self) -> int:
        """Liczba cech numerycznych."""
        return len(self.num_cols)

    @property
    def bin_dim(self) -> int:
        """Liczba cech binarnych."""
        return len(self.bin_cols)

    @property
    def class_counts(self) -> dict:
        """Zwraca dict {klasa: liczba} — do debugowania."""
        counts = self.y.bincount().tolist()
        return {i: counts[i] for i in range(len(counts))}
