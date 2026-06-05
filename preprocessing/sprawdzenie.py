"""
Moduł walidacji danych dla US Accidents dataset.
Sprawdza:
  1. Wartości poza domeną fizyczną (przed cappingiem)
  2. Outliery IQR per kolumna i per klasa Severity
  3. Czy outliery koncentrują się w jednej klasie (ryzyko wycięcia klasy)

Uruchomienie (PRZED preprocessingiem, na surowych danych):
    python validation_report.py --input ../US_Accidents_1M.csv
"""

import argparse
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Granice domenowe — to samo co CAPPING_RULES w preprocessingu
DOMAIN_BOUNDS = {
    'Temperature(F)':    (-40.0, 120.0),
    'Wind_Speed(mph)':   (0.0,   60.0),
    'Precipitation(in)': (0.0,    3.0),
    'Visibility(mi)':    (0.0,   10.0),
    'Pressure(in)':      (27.0,  32.0),
    'Humidity(%)':        (0.0, 100.0),
    'Distance(mi)':      (0.0,   30.0),
    'Duration_min':      (0.0, 1440.0),
}

# Kolumny które nigdy nie powinny być ujemne
NON_NEGATIVE_COLS = [
    'Distance(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
    'Visibility(mi)', 'Humidity(%)',
]

def sep(title=""):
    print(f"\n{'=' * 65}")
    if title:
        print(title)
        print('=' * 65)


def check_domain_bounds(df: pd.DataFrame, n_total: int) -> None:
    """Sprawdza wartości poza granicami domenowymi."""
    sep("1. WARTOŚCI POZA GRANICAMI DOMENOWYMI")
    print(f"  (zostaną przycięte przez capping — wiersze NIE są usuwane)")
    print(f"\n  {'Kolumna':<28} {'Min':>10} {'Max':>10} {'<dolna':>8} {'>górna':>8}  Status")
    print("  " + "-" * 75)

    for col, (lo, hi) in DOMAIN_BOUNDS.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        n_below = (s < lo).sum()
        n_above = (s > hi).sum()
        col_min = s.min()
        col_max = s.max()
        status = "OK" if (n_below + n_above) == 0 else f"! {n_below+n_above:,} poza zakresem"
        print(f"  {col:<28} {col_min:>10.2f} {col_max:>10.2f} {n_below:>8,} {n_above:>8,}  {status}")


def check_non_negative(df: pd.DataFrame) -> None:
    """Sprawdza kolumny które nie mogą być ujemne."""
    sep("2. KOLUMNY KTÓRE NIE MOGĄ BYĆ UJEMNE")
    print(f"\n  {'Kolumna':<28} {'Ujemnych':>10}  Status")
    print("  " + "-" * 50)

    for col in NON_NEGATIVE_COLS:
        if col not in df.columns:
            continue
        n_neg = (df[col].dropna() < 0).sum()
        status = "OK" if n_neg == 0 else f"! {n_neg:,} ujemnych wartości"
        print(f"  {col:<28} {n_neg:>10,}  {status}")


def check_outliers_iqr(df: pd.DataFrame) -> dict:
    """
    Wykrywa outliery metodą IQR (Q1 - 1.5*IQR, Q3 + 1.5*IQR).
    Zwraca słownik: col -> maska outlierów.
    """
    sep("3. OUTLIERY METODĄ IQR (1.5×IQR)")
    print(f"  (outliery = poniżej Q1-1.5·IQR lub powyżej Q3+1.5·IQR)")
    print(f"\n  {'Kolumna':<28} {'Q1':>8} {'Q3':>8} {'IQR':>8} {'Outlierów':>10} {'%':>7}")
    print("  " + "-" * 75)

    outlier_masks = {}
    for col in DOMAIN_BOUNDS.keys():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo_iqr = q1 - 1.5 * iqr
        hi_iqr = q3 + 1.5 * iqr
        mask = (df[col] < lo_iqr) | (df[col] > hi_iqr)
        n_out = mask.sum()
        pct = n_out / len(df) * 100
        outlier_masks[col] = mask
        print(f"  {col:<28} {q1:>8.2f} {q3:>8.2f} {iqr:>8.2f} {n_out:>10,} {pct:>6.2f}%")

    return outlier_masks


def check_outliers_per_class(df: pd.DataFrame, outlier_masks: dict) -> None:
    """
    KLUCZOWE wg Bartmana: sprawdza czy outliery koncentrują się w jednej klasie.
    Jeśli tak — wycinając outliery możemy wyciąć całą klasę.
    """
    sep("4. OUTLIERY PER KLASA SEVERITY — RYZYKO WYCIĘCIA KLASY")
    print("  Wg Bartmana: jeśli cała klasa to outliery → NIE usuwać!")
    print()

    severity_col = 'Severity' if df['Severity'].max() <= 3 else None
    if severity_col is None:
        print("  Brak kolumny Severity — pomijam.")
        return

    class_counts = df['Severity'].value_counts().sort_index()

    for col, mask in outlier_masks.items():
        n_outliers = mask.sum()
        if n_outliers == 0:
            continue

        print(f"  ── {col} ({n_outliers:,} outlierów) ──")
        print(f"  {'Klasa':>7} {'Całość':>9} {'Outliery':>10} {'% klasy':>9}  Ryzyko")
        print("  " + "-" * 55)

        risk_flag = False
        for cls in sorted(df['Severity'].unique()):
            cls_mask = df['Severity'] == cls
            cls_total = cls_mask.sum()
            cls_outliers = (mask & cls_mask).sum()
            pct_of_class = cls_outliers / cls_total * 100 if cls_total > 0 else 0

            if pct_of_class >= 50:
                risk = "⚠ WYSOKIE — >50% klasy to outliery!"
                risk_flag = True
            elif pct_of_class >= 20:
                risk = "! umiarkowane — >20% klasy"
            else:
                risk = "OK"

            print(f"  {cls:>7} {cls_total:>9,} {cls_outliers:>10,} {pct_of_class:>8.1f}%  {risk}")

        if risk_flag:
            print(f"  >>> ZALECENIE: NIE usuwaj outlierów w '{col}' — ryzyko wycięcia klasy!")
        print()


def check_class_balance(df: pd.DataFrame) -> None:
    """Pokazuje rozkład klas w surowych danych."""
    sep("5. ROZKŁAD KLAS SEVERITY (surowe dane)")
    counts = df['Severity'].value_counts().sort_index()
    total = len(df)
    print(f"\n  {'Klasa':>7} {'Liczba':>10} {'%':>8}  Uwaga")
    print("  " + "-" * 45)
    for cls, cnt in counts.items():
        pct = cnt / total * 100
        note = "⚠ bardzo mała klasa!" if pct < 2 else ("! mała klasa" if pct < 5 else "")
        print(f"  {cls:>7} {cnt:>10,} {pct:>7.2f}%  {note}")


def run(input_csv: str) -> None:
    logger.info("Wczytywanie danych...")
    df = pd.read_csv(input_csv)
    n_total = len(df)
    logger.info(f"Wczytano: {df.shape}")

    # Przygotuj Duration_min jeśli nie ma
    if 'Duration_min' not in df.columns and 'Start_Time' in df.columns:
        df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
        df['End_Time']   = pd.to_datetime(df['End_Time'],   errors='coerce')
        df['Duration_min'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60.0

    # Mapuj Severity do 0-3 jeśli jest 1-4
    if 'Severity' in df.columns and df['Severity'].max() == 4:
        df['Severity'] = df['Severity'].astype(int) - 1

    check_domain_bounds(df, n_total)
    check_non_negative(df)
    outlier_masks = check_outliers_iqr(df)
    check_outliers_per_class(df, outlier_masks)
    check_class_balance(df)

    sep("PODSUMOWANIE")
    print("  Preprocessing stosuje CAPPING (nie usuwa wierszy) — bezpieczne.")
    print("  Sprawdź sekcję 4 czy nie ma ryzyka wycięcia klasy przez IQR.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../US_Accidents_1M.csv")
    args = parser.parse_args()
    run(args.input)