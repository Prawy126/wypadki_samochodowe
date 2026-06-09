"""
Architektura MLP z Entity Embeddings dla cech kategorycznych.

Schemat:
    cat → [Embedding × 8] → concat (160 dim)
    num (19) + bin (20) + embed (160) → 199 dim wejściowych
    → BatchNorm → [Linear → BN → ReLU → Dropout] × N → Linear(→4)
"""

import torch
import torch.nn as nn
from typing import Dict, List


class EntityEmbeddingMLP(nn.Module):
    """
    MLP klasyfikator z warstwami Entity Embedding dla cech kategorycznych.

    Args:
        embedding_dims : dict {nazwa_kolumny: [vocab_size, embed_dim]}
                         wczytany z metadata["embedding_dims"]
        cat_col_order  : lista nazw kolumn kategorycznych w tej samej kolejności
                         co tensor 'cat' z AccidentDataset
        num_dim        : liczba cech numerycznych (18)
        bin_dim        : liczba cech binarnych (20)
        hidden_dims    : wymiary warstw ukrytych, np. [512, 256, 128]
        dropout        : współczynnik Dropout
        num_classes    : liczba klas (4)
    """

    def __init__(
        self,
        embedding_dims: Dict[str, List[int]],
        cat_col_order: List[str],
        num_dim: int,
        bin_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        num_classes: int = 4,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.cat_col_order = cat_col_order

        # ---------- Entity Embedding layers ----------
        self.embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=embedding_dims[col][0],
                embedding_dim=embedding_dims[col][1],
                padding_idx=0,  # 0 → 'Unknown' → zerowy wektor
            )
            for col in cat_col_order
        ])

        # Suma wymiarów wszystkich embeddingów
        embed_out_dim = sum(embedding_dims[col][1] for col in cat_col_order)

        # Łączny wymiar wejściowy: num + bin + embed
        input_dim = num_dim + bin_dim + embed_out_dim

        # ---------- MLP bloki ----------
        layers: List[nn.Module] = [nn.BatchNorm1d(input_dim)]  # normalizacja wejścia
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ])
            in_dim = h_dim

        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_classes)

        # Inicjalizacja wag warstw liniowych (He initialization dla ReLU)
        self._init_weights()

        # Zapis konfiguracji — potrzebny do odtworzenia modelu z checkpointu
        self.config = {
            "embedding_dims":  embedding_dims,
            "cat_col_order":   cat_col_order,
            "num_dim":         num_dim,
            "bin_dim":         bin_dim,
            "hidden_dims":     hidden_dims,
            "dropout":         dropout,
            "num_classes":     num_classes,
            "embed_out_dim":   embed_out_dim,
            "input_dim":       input_dim,
        }

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                # Zeruj padding_idx ręcznie
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

    def forward(
        self,
        cat: torch.Tensor,       # LongTensor  [B, num_cat_cols]
        num: torch.Tensor,       # FloatTensor [B, num_dim]
        bin_feats: torch.Tensor, # FloatTensor [B, bin_dim]
    ) -> torch.Tensor:
        """Zwraca logity [B, num_classes]."""

        # Entity embeddings dla każdej kolumny kategorycznej
        emb_out = torch.cat(
            [emb(cat[:, i]) for i, emb in enumerate(self.embeddings)],
            dim=1,
        )  # [B, embed_out_dim]

        # Połączenie wszystkich cech
        x = torch.cat([num, bin_feats, emb_out], dim=1)  # [B, input_dim]

        # Przejście przez MLP
        x = self.mlp(x)
        return self.classifier(x)  # [B, num_classes]

    @torch.no_grad()
    def predict_proba(
        self,
        cat: torch.Tensor,
        num: torch.Tensor,
        bin_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Zwraca prawdopodobieństwa klas [B, num_classes] — dla inferencji/GUI."""
        self.eval()
        logits = self.forward(cat, num, bin_feats)
        return torch.softmax(logits, dim=1)

    def count_parameters(self) -> int:
        """Zwraca liczbę trenowalnych parametrów."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
