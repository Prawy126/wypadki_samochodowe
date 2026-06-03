import pandas as pd
from sklearn.model_selection import train_test_split

def reduce_dataset(input_path, output_path, target_size=1000000, target_column='Severity'):
    print("Wczytywanie ogromnego pliku... (to może chwilę potrwać)")
    df = pd.read_csv(input_path)
    print(f"Wczytano. Oryginalny rozmiar: {df.shape}")

    # Obliczamy frakcję, jaką chcemy zachować (np. 1/7)
    fraction = target_size / len(df)

    # Usuwamy wiersze z brakującą wartością docelową
    df = df.dropna(subset=[target_column])

    print("Próbkowanie danych...")
    # Podział ze stratyfikacją względem zmiennej docelowej
    df_reduced, _ = train_test_split(
        df,
        train_size=fraction,
        stratify=df[target_column],
        random_state=42
    )

    print(f"Nowy rozmiar: {df_reduced.shape}")

    # Zapisujemy mniejszy plik
    df_reduced.to_csv(output_path, index=False)
    print(f"Zapisano pomniejszony zbiór do: {output_path}")


# Ten blok MUSI być przy samej lewej krawędzi (bez żadnych spacji na początku)
if __name__ == "__main__":
    reduce_dataset(
        input_path="../US_Accidents_March23.csv",
        output_path="../US_Accidents_1M.csv"
    )