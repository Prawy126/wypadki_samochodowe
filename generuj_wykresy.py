import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants and Colors
SEVERITY_COLORS = {
    1: "#4CAF50",   # Green (Mild)
    2: "#FF9800",   # Orange (Moderate)
    3: "#F44336",   # Red (Severe)
    4: "#7B0000",   # Dark Red (Critical)
}
SEVERITY_LABELS = {
    1: "Severity 1 (Łagodny)",
    2: "Severity 2 (Umiarkowany)",
    3: "Severity 3 (Poważny)",
    4: "Severity 4 (Krytyczny)",
}
PRIMARY_COLOR = "#4F46E5"  # Beautiful Indigo
SECONDARY_COLOR = "#06B6D4" # Cyan
ACCENT_COLOR = "#EC4899"    # Pink

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "font.family": "sans-serif",
    "figure.dpi": 150,
})

def prepare_directories(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Katalog na wykresy: {output_dir}")

def generate_severity_dist(df: pd.DataFrame, output_dir: Path):
    logger.info("Generowanie wykresu rozkładu severity...")
    plt.figure(figsize=(9, 5))
    
    counts = df["Severity"].value_counts().sort_index()
    total = len(df)
    
    colors = [SEVERITY_COLORS.get(idx, PRIMARY_COLOR) for idx in counts.index]
    labels = [SEVERITY_LABELS.get(idx, f"Severity {idx}") for idx in counts.index]
    
    bars = plt.bar(labels, counts.values, color=colors, edgecolor="black", alpha=0.85, width=0.6)
    
    # Add values and percentages on top of bars
    for bar in bars:
        yval = bar.get_height()
        pct = (yval / total) * 100
        plt.text(
            bar.get_x() + bar.get_width()/2.0, 
            yval + total*0.01, 
            f"{yval:,}\n({pct:.2f}%)", 
            ha="center", 
            va="bottom", 
            fontweight="bold",
            fontsize=10
        )
        
    plt.title("Rozkład ciężkości wypadków (Severity) w zbiorze danych", pad=20, fontweight="bold")
    plt.ylabel("Liczba wypadków")
    plt.ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    plt.savefig(output_dir / "01_rozkład_severity.png", dpi=300)
    plt.close()

def generate_temporal_analysis(df: pd.DataFrame, output_dir: Path):
    logger.info("Generowanie wykresów czasowych...")
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df = df.dropna(subset=["Start_Time"])
    
    df["Hour"] = df["Start_Time"].dt.hour
    df["DayOfWeek"] = df["Start_Time"].dt.dayofweek
    df["Month"] = df["Start_Time"].dt.month
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 14))
    
    # 1. Accidents by Hour
    hour_counts = df["Hour"].value_counts().sort_index()
    axes[0].plot(hour_counts.index, hour_counts.values, marker="o", color=PRIMARY_COLOR, linewidth=2.5, markersize=6)
    axes[0].fill_between(hour_counts.index, hour_counts.values, color=PRIMARY_COLOR, alpha=0.15)
    axes[0].set_title("Liczba wypadków w zależności od pory dnia (godziny)", fontweight="bold")
    axes[0].set_xlabel("Godzina")
    axes[0].set_ylabel("Liczba wypadków")
    axes[0].set_xticks(range(0, 24))
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    # Highlight rush hours
    axes[0].axvspan(7, 9, color="red", alpha=0.08, label="Szczyt poranny (7:00 - 9:00)")
    axes[0].axvspan(16, 19, color="orange", alpha=0.08, label="Szczyt popołudniowy (16:00 - 19:00)")
    axes[0].legend(loc="upper left")
    
    # 2. Accidents by Day of Week
    dow_counts = df["DayOfWeek"].value_counts().sort_index()
    dow_labels = ["Pon", "Wt", "Śr", "Czw", "Pt", "Sob", "Niedz"]
    dow_colors = [PRIMARY_COLOR if i < 5 else ACCENT_COLOR for i in range(7)]
    
    bars = axes[1].bar(dow_labels, dow_counts.values, color=dow_colors, edgecolor="black", alpha=0.8, width=0.55)
    axes[1].set_title("Liczba wypadków w poszczególne dni tygodnia", fontweight="bold")
    axes[1].set_xlabel("Dzień tygodnia")
    axes[1].set_ylabel("Liczba wypadków")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.6)
    
    # Add values
    for bar in bars:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + max(dow_counts.values)*0.01, f"{yval:,}", ha="center", va="bottom", fontsize=9)
        
    # 3. Accidents by Month
    month_counts = df["Month"].value_counts().sort_index()
    month_labels = ["Sty", "Lut", "Mar", "Kwi", "Maj", "Cze", "Lip", "Sie", "Wrze", "Paź", "Lis", "Gru"]
    axes[2].bar(month_labels, month_counts.values, color=SECONDARY_COLOR, edgecolor="black", alpha=0.8, width=0.6)
    axes[2].plot(month_labels, month_counts.values, color=PRIMARY_COLOR, marker="s", markersize=5, linewidth=1.5)
    axes[2].set_title("Liczba wypadków w ujęciu miesięcznym", fontweight="bold")
    axes[2].set_xlabel("Miesiąc")
    axes[2].set_ylabel("Liczba wypadków")
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_dir / "02_analiza_czasowa.png", dpi=300)
    plt.close()

def generate_geo_map(df: pd.DataFrame, output_dir: Path):
    logger.info("Generowanie mapy geograficznej wypadków...")
    # Filter coordinates to mainland USA bounds to avoid outliers
    df_geo = df[
        (df["Start_Lat"] > 24) & (df["Start_Lat"] < 50) &
        (df["Start_Lng"] > -125) & (df["Start_Lng"] < -66)
    ].copy()
    
    # Stratified sample to avoid overplotting (100k points is enough to see US shape)
    sample_size = min(100000, len(df_geo))
    df_sample = df_geo.sample(n=sample_size, random_state=42)
    
    plt.figure(figsize=(12, 7))
    
    # Scatter plot with low alpha to see density
    # Plot lower severity first, critical on top to ensure visibility
    for sev in sorted(df_sample["Severity"].unique()):
        sub = df_sample[df_sample["Severity"] == sev]
        color = SEVERITY_COLORS.get(sev, PRIMARY_COLOR)
        label = SEVERITY_LABELS.get(sev, f"Severity {sev}")
        alpha = 0.08 if sev <= 2 else 0.35
        size = 1.0 if sev <= 2 else 4.0
        
        plt.scatter(
            sub["Start_Lng"], 
            sub["Start_Lat"], 
            color=color, 
            label=label, 
            s=size, 
            alpha=alpha, 
            edgecolors="none"
        )
        
    plt.title("Geograficzny rozkład wypadków w USA z podziałem na ciężkość", pad=15, fontweight="bold")
    plt.xlabel("Długość geograficzna (Longitude)")
    plt.ylabel("Szerokość geograficzna (Latitude)")
    
    # Legend with opaque markers
    lgnd = plt.legend(loc="lower left", scatterpoints=1, fontsize=10, title="Ciężkość wypadku")
    handles = getattr(lgnd, "legend_handles", None) or getattr(lgnd, "legendHandles", [])
    for handle in handles:
        handle.set_alpha(1.0)
        handle.set_sizes([30.0])
        
    plt.tight_layout()
    plt.savefig(output_dir / "03_mapa_geograficzna.png", dpi=300)
    plt.close()

def generate_weather_vs_severity(df: pd.DataFrame, output_dir: Path):
    logger.info("Generowanie wykresu wpływu pogody...")
    
    # Weather map from preprocessing.py
    WEATHER_MAP = {
        'Fair': 'Czysto', 'Clear': 'Czysto', 'Fair / Windy': 'Czysto',
        'Mostly Cloudy': 'Zachmurzenie', 'Cloudy': 'Zachmurzenie', 'Partly Cloudy': 'Zachmurzenie',
        'Overcast': 'Zachmurzenie', 'Scattered Clouds': 'Zachmurzenie',
        'Mostly Cloudy / Windy': 'Zachmurzenie', 'Cloudy / Windy': 'Zachmurzenie',
        'Partly Cloudy / Windy': 'Zachmurzenie',
        'Light Rain': 'Deszcz', 'Rain': 'Deszcz', 'Heavy Rain': 'Deszcz',
        'Light Drizzle': 'Deszcz', 'Drizzle': 'Deszcz', 'Heavy Rain / Windy': 'Deszcz',
        'Light Rain / Windy': 'Deszcz', 'Rain / Windy': 'Deszcz',
        'Light Snow': 'Śnieg', 'Snow': 'Śnieg', 'Heavy Snow': 'Śnieg',
        'Light Snow / Windy': 'Śnieg', 'Snow / Windy': 'Śnieg',
        'Fog': 'Mgła', 'Haze': 'Mgła', 'Mist': 'Mgła', 'Smoke': 'Mgła',
        'Patches of Fog': 'Mgła', 'Shallow Fog': 'Mgła',
        'T-Storm': 'Burza', 'Thunder in the Vicinity': 'Burza', 'Thunder': 'Burza',
        'Heavy T-Storm': 'Burza', 'Light Thunderstorms and Rain': 'Burza',
        'Heavy T-Storm / Windy': 'Burza', 'T-Storm / Windy': 'Burza',
    }
    
    df_weather = df.copy()
    df_weather["Weather_Cat"] = df_weather["Weather_Condition"].map(WEATHER_MAP).fillna("Inne")
    
    # Cross tabulation
    ct = pd.crosstab(df_weather["Weather_Cat"], df_weather["Severity"], normalize="index") * 100
    ct = ct.sort_index()
    
    # Map column names for display
    ct.columns = [SEVERITY_LABELS.get(col, f"Severity {col}") for col in ct.columns]
    
    # Plot 100% stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [SEVERITY_COLORS[i] for i in sorted(SEVERITY_COLORS.keys())]
    ct.plot(kind="barh", stacked=True, color=colors, edgecolor="black", alpha=0.85, ax=ax)
    
    ax.set_title("Procentowy rozkład ciężkości wypadków w zależności od pogody", pad=15, fontweight="bold")
    ax.set_xlabel("Procent wypadków (%)")
    ax.set_ylabel("Warunki pogodowe")
    ax.legend(title="Ciężkość", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add labels on bars
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        if width > 4: # only write if segment is wide enough
            x, y = p.get_xy()
            ax.text(x + width/2, 
                    y + height/2, 
                    f"{width:.1f}%", 
                    horizontalalignment='center', 
                    verticalalignment='center', 
                    color='white', 
                    fontweight='bold',
                    fontsize=8)
            
    plt.tight_layout()
    plt.savefig(output_dir / "04_pogoda_vs_severity.png", dpi=300)
    plt.close()

def generate_road_features(df: pd.DataFrame, output_dir: Path):
    logger.info("Generowanie wykresu elementów infrastruktury...")
    
    BOOL_COLS = [
        'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
        'No_Exit', 'Railway', 'Station', 'Stop', 'Traffic_Calming',
        'Traffic_Signal',
    ]
    
    feature_counts = {}
    feature_severity_means = {}
    
    for col in BOOL_COLS:
        if col in df.columns:
            # Map True/False to 1/0
            df[col] = df[col].astype(bool)
            sub = df[df[col] == True]
            feature_counts[col] = len(sub)
            feature_severity_means[col] = sub["Severity"].mean()
            
    feat_df = pd.DataFrame({
        "Count": feature_counts,
        "Mean_Severity": feature_severity_means
    }).sort_values(by="Count", ascending=False)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar chart for counts
    color_bars = "#3B82F6"
    bars = ax1.bar(feat_df.index, feat_df["Count"], color=color_bars, alpha=0.7, edgecolor="black", width=0.55, label="Liczba wypadków")
    ax1.set_xlabel("Element infrastruktury drogowej")
    ax1.set_ylabel("Liczba wypadków w pobliżu", color=color_bars)
    ax1.tick_params(axis="y", labelcolor=color_bars)
    ax1.set_xticklabels(feat_df.index, rotation=45, ha="right")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
    
    # Line chart for mean severity (secondary axis)
    ax2 = ax1.twinx()
    color_line = "#EF4444"
    ax2.plot(feat_df.index, feat_df["Mean_Severity"], color=color_line, marker="o", linewidth=2, label="Średnia Severity (0-3)")
    ax2.set_ylabel("Średnia ciężkość wypadku (skala 0-3)", color=color_line)
    ax2.tick_params(axis="y", labelcolor=color_line)
    
    # Add values above bars
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + max(feat_df["Count"])*0.01, f"{yval:,}", ha="center", va="bottom", fontsize=8, color="#1E3A8A")
        
    plt.title("Wypadki w pobliżu elementów infrastruktury a ich średnia ciężkość", pad=15, fontweight="bold")
    fig.tight_layout()
    plt.savefig(output_dir / "05_infrastruktura_wplyw.png", dpi=300)
    plt.close()

def generate_model_comparison(output_dir: Path):
    logger.info("Generowanie wykresu porównania modeli...")
    summary_path = Path("uczenie/outputs/results_summary.json")
    if not summary_path.exists():
        logger.warning(f"Brak pliku podsumowania wyników modeli: {summary_path}")
        return
        
    with open(summary_path, "r") as f:
        results = json.load(f)
        
    models_df = pd.DataFrame(results)
    
    metrics = ["accuracy", "macro_f1", "weighted_f1", "cohen_kappa", "roc_auc_macro"]
    metric_labels = ["Accuracy", "Macro F1-score", "Weighted F1-score", "Cohen's Kappa", "ROC AUC Macro"]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    
    # Find MLP and LGBM rows
    mlp_row = models_df[models_df["model"].str.contains("MLP", case=False)].iloc[0]
    lgbm_row = models_df[models_df["model"].str.contains("LightGBM", case=False)].iloc[0]
    
    mlp_vals = [mlp_row[m] for m in metrics]
    lgbm_vals = [lgbm_row[m] for m in metrics]
    
    rects1 = plt.bar(x - width/2, mlp_vals, width, label="Entity Embedding MLP", color=SECONDARY_COLOR, edgecolor="black", alpha=0.85)
    rects2 = plt.bar(x + width/2, lgbm_vals, width, label="LightGBM", color=PRIMARY_COLOR, edgecolor="black", alpha=0.85)
    
    plt.ylabel("Wartość metryki")
    plt.title("Porównanie wyników klasyfikacji: MLP vs LightGBM", pad=15, fontweight="bold")
    plt.xticks(x, metric_labels)
    plt.ylim(0, 1.15)
    plt.legend(loc="upper left")
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f"{height:.4f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight="bold")
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(output_dir / "06_porownanie_modeli.png", dpi=300)
    plt.close()

def main():
    csv_path = Path("US_Accidents_1M.csv")
    if not csv_path.exists():
        logger.error(f"Nie znaleziono pliku {csv_path} w bieżącym katalogu.")
        return
        
    output_dir = Path("raport_wykresy")
    prepare_directories(output_dir)
    
    logger.info("Wczytywanie zbioru danych (US_Accidents_1M.csv)...")
    df = pd.read_csv(csv_path)
    logger.info(f"Wczytano {len(df):,} wierszy.")
    
    generate_severity_dist(df, output_dir)
    generate_temporal_analysis(df, output_dir)
    generate_geo_map(df, output_dir)
    generate_weather_vs_severity(df, output_dir)
    generate_road_features(df, output_dir)
    generate_model_comparison(output_dir)
    
    logger.info("Wszystkie wykresy zostały pomyślnie wygenerowane w folderze: raport_wykresy/")

if __name__ == "__main__":
    main()
