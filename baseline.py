"""Baseline-uri (persistență, climatologie, Random Forest) pe 3 split-uri: temporal, stație nouă, an nou."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

COL, TGT = "no2_column_mol_m2", "no2_ug_m3"
DIR = Path(__file__).resolve().parent
s5p_file = Path(sys.argv[1]) if len(sys.argv) > 2 else DIR / "raw_data/UrbanTwin_S5P_daily_2024_2026.csv"
oaq_file = Path(sys.argv[2]) if len(sys.argv) > 2 else DIR / "raw_data/openaq_no2_bucuresti_2024_2026.csv"

# datele: aceeași preprocesare ca în data_loader.py
s5p = pd.read_csv(s5p_file, parse_dates=["date"]).dropna(subset=[COL]).groupby("date", as_index=False)[COL].mean()
oaq = pd.read_csv(oaq_file, parse_dates=["date"])
df = oaq.merge(s5p, on="date").dropna(subset=[TGT]).sort_values(["date", "sensor_id"]).reset_index(drop=True)
df["station"] = df["statie"]
print(f"observatii={len(df)} zile={df.date.nunique()} statii={df.station.nunique()} senzori={df.sensor_id.nunique()}")

# reprezentări: A = coloana S5P, B = A + calendar
doy = df.date.dt.day_of_year
df["sin"], df["cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
df["dow"] = df.date.dt.day_of_week
df["month"], df["year"] = df.date.dt.month, df.date.dt.year
df["col"] = df[COL] * 1e6
df["prev"] = df.groupby("sensor_id")[TGT].shift(1)  # ultima valoare anterioară a senzorului
FEATURES = {"RF A": ["col"], "RF B": ["col", "sin", "cos", "dow"]}

# split temporal: primele 85% din zile (train+val), ultimele 15% test
dates = np.sort(df.date.unique())
cut = int(0.85 * len(dates))
train_val, test = df[df.date.isin(dates[:cut])], df[df.date.isin(dates[cut:])]
print(f"test: {pd.Timestamp(dates[cut]).date()} -> {pd.Timestamp(dates[-1]).date()}")


def predict(name, train, test):
    if name == "persistenta":
        return test["prev"].fillna(train[TGT].mean())
    if name == "climatologie":  # media senzor x lună; dacă senzorul nu e în train, media lunii
        sensor_luna = train.groupby(["sensor_id", "month"])[TGT].mean()
        luna = train.groupby("month")[TGT].mean()
        return [sensor_luna.get((s, m), luna.get(m, train[TGT].mean())) for s, m in zip(test.sensor_id, test.month)]
    f = FEATURES[name]
    rf = lambda seed: RandomForestRegressor(300, min_samples_leaf=5, n_jobs=-1, random_state=seed).fit(train[f], train[TGT])
    return np.mean([rf(s).predict(test[f]) for s in range(5)], axis=0)


def ic95(err, days, n=1000):
    """Interval de încredere 95% pentru MAE, bootstrap pe zile."""
    rng = np.random.default_rng(0)
    groups = [np.where(days == d)[0] for d in np.unique(days)]
    maes = [err[np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])].mean() for _ in range(n)]
    return np.percentile(maes, [2.5, 97.5])


names = ["persistenta", "climatologie", "RF A", "RF B"]
rows = []
for name in names:
    err = (test[TGT] - predict(name, train_val, test)).abs().values
    rows.append((name, "temporal", "test", err.mean(), *ic95(err, test.date.values)))
    for st in sorted(df.station.unique()):  # stație nouă
        a, b = df[df.station != st], df[df.station == st]
        rows.append((name, "spatial", st, (b[TGT] - predict(name, a, b)).abs().mean(), None, None))
    for yr in sorted(df.year.unique()):  # an nou (2026 e parțial)
        a, b = df[df.year != yr], df[df.year == yr]
        rows.append((name, "LOYO", str(yr), (b[TGT] - predict(name, a, b)).abs().mean(), None, None))

res = pd.DataFrame(rows, columns=["model", "split", "fold", "MAE", "IC_lo", "IC_hi"])
res.to_csv("results_cp2_long.csv", index=False)

t = res[res.split == "temporal"].set_index("model")
spatial = res[res.split == "spatial"].groupby("model").MAE
loyo = res[res.split == "LOYO"].pivot(index="model", columns="fold", values="MAE")
summary = pd.DataFrame({"temporal": t.MAE, "IC95_lo": t.IC_lo, "IC95_hi": t.IC_hi, "spatial": spatial.mean(),
                        "spatial_min": spatial.min(), "spatial_max": spatial.max(), "LOYO": loyo.mean(axis=1)})
summary = summary.join(loyo.add_prefix("LOYO_")).loc[names].round(2)
summary.to_csv("results_cp2.csv")
print(summary.to_string())