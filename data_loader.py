from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


DATA_DIR = Path(__file__).resolve().parent
S5P_FILE = DATA_DIR / "raw_data/UrbanTwin_S5P_daily_2024_2026.csv"
OPENAQ_FILE = DATA_DIR / "raw_data/openaq_no2_bucuresti_2024_2026.csv"

BATCH_SIZE = 32
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


class NO2Dataset(Dataset):
    def __init__(self, dataframe):
        self.features = torch.tensor(dataframe[["no2_column_mol_m2"]].to_numpy(),dtype=torch.float32)
        self.targets = torch.tensor(dataframe["no2_ug_m3"].to_numpy(),dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


def load_data():
    if not S5P_FILE.exists():
        raise FileNotFoundError(f"Fișier S5P negăsit: {S5P_FILE}")

    if not OPENAQ_FILE.exists():
        raise FileNotFoundError(f"Fișier OpenAQ negăsit: {OPENAQ_FILE}")

    s5p_data = pd.read_csv(S5P_FILE)
    openaq_data = pd.read_csv(OPENAQ_FILE)

    return s5p_data, openaq_data


def preprocess_data(s5p_data, openaq_data):
    """Curata si combins datele S5P si OpenAQ."""

    # convertim la tipul de date corespunzator
    s5p_data["date"] = pd.to_datetime(s5p_data["date"], errors="coerce").dt.normalize()
    openaq_data["date"] = pd.to_datetime(openaq_data["date"], errors="coerce").dt.normalize()
    s5p_data["no2_column_mol_m2"] = pd.to_numeric(s5p_data["no2_column_mol_m2"], errors="coerce")
    openaq_data["no2_ug_m3"] = pd.to_numeric(openaq_data["no2_ug_m3"], errors="coerce")

    # eliminam valorile lipsa
    s5p_data = s5p_data.dropna(subset=["date", "no2_column_mol_m2"])


    # Cclculam media zilnica S5P
    s5p_daily = (s5p_data.groupby("date", as_index=False).agg(no2_column_mol_m2=("no2_column_mol_m2", "mean")))

    # agregram dupa zi
    combined_data = openaq_data.merge(
        s5p_daily,
        on="date",
        how="inner",
        validate="many_to_one"
    )

    combined_data = combined_data.sort_values(["date", "sensor_id"]).reset_index(drop=True)

    if combined_data.empty:
        raise ValueError("Nu există observasii comune valide.")

    return combined_data


def split_data(dataframe):
    """Impartim cronologic datele în train, validation si test."""

    unique_dates = sorted(dataframe["date"].unique())

    if len(unique_dates) < 3:
        raise ValueError("Sunt necesare cel puțin trei date distincte.")

    train_index = int(len(unique_dates) * TRAIN_RATIO)
    val_index = int(len(unique_dates) * (TRAIN_RATIO + VAL_RATIO))

    train_index = max(1, min(train_index, len(unique_dates) - 2))
    val_index = max(train_index + 1, min(val_index, len(unique_dates) - 1))

    train_end = pd.Timestamp(unique_dates[train_index])
    val_end = pd.Timestamp(unique_dates[val_index])

    train_data = dataframe[dataframe["date"] < train_end].copy()

    val_data = dataframe[
        (dataframe["date"] >= train_end)
        & (dataframe["date"] < val_end)
    ].copy()

    test_data = dataframe[dataframe["date"] >= val_end].copy()

    return train_data, val_data, test_data


def create_data_loaders(train_data, val_data, test_data):
    train_dataset = NO2Dataset(train_data)
    val_dataset = NO2Dataset(val_data)
    test_dataset = NO2Dataset(test_data)
    generator = torch.Generator().manual_seed(42)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


def main():
    s5p_data, openaq_data = load_data()
    combined_data = preprocess_data(s5p_data, openaq_data)

    train_data, val_data, test_data = split_data(combined_data)

    train_loader, val_loader, test_loader = create_data_loaders(train_data, val_data, test_data)

    print("Pipeline-ul a rulat cu succes.")
    print(f"Observații totale: {len(combined_data)}")
    print(f"Train: {len(train_data)} observații")
    print(f"Validation: {len(val_data)} observații")
    print(f"Test: {len(test_data)} observații")

    print(
        f"Perioada totală: {combined_data['date'].min().date()} - "
        f"{combined_data['date'].max().date()}"
    )

    print(f"Batch-uri train: {len(train_loader)}")
    print(f"Batch-uri validation: {len(val_loader)}")
    print(f"Batch-uri test: {len(test_loader)}")


if __name__ == "__main__":
    main()