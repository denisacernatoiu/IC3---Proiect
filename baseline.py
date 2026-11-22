import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")  

COL, TARGET = "no2_column_mol_m2", "no2_ug_m3"
N_SEEDS = 10          
N_TREES = 300         # arbori per Random Forest
BLOCK = 7             # zile per bloc
N_BOOT = 2000         # repetari bootstrap
REF = "climatologie"  # model de referinta

DIR = Path(__file__).resolve().parent
s5p_file = Path(sys.argv[1]) if len(sys.argv) > 2 else DIR / "raw_data/UrbanTwin_S5P_daily_2024_2026.csv"
oaq_file = Path(sys.argv[2]) if len(sys.argv) > 2 else DIR / "raw_data/openaq_no2_bucuresti_2024_2026.csv"

# media zilnica S5P, apoi asociere cu valorile senzorilor dupa data
s5p = pd.read_csv(s5p_file, parse_dates=["date"]).dropna(subset=[COL]).groupby("date", as_index=False)[COL].mean()
oaq = pd.read_csv(oaq_file, parse_dates=["date"])
df = oaq.merge(s5p, on="date", validate="many_to_one").dropna(subset=[TARGET])
df = df.sort_values(["date", "sensor_id"]).reset_index(drop=True)
df["station"] = df["statie"]
print(f"observatii={len(df)} zile={df.date.nunique()} statii={df.station.nunique()} senzori={df.sensor_id.nunique()}")


day_of_year = df.date.dt.day_of_year
df["sin"] = np.sin(2 * np.pi * day_of_year / 365.25)   # sezonalitate
df["cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
df["dow"] = df.date.dt.day_of_week                      # ziua saptamanii
df["month"] = df.date.dt.month
df["year"] = df.date.dt.year
df["col"] = df[COL] * 1e6                               # coloana satelitara redimensionata

# valoarea anterioara si numarul de zile de la ultima masuratoare
df["prev"] = df.groupby("sensor_id")[TARGET].shift(1)
df["gap"] = (df.date - df.groupby("sensor_id").date.shift(1)).dt.days
print(f"persistenta: lag exact 1 zi pe {np.mean(df.gap == 1):.1%} din randuri (restul = ultima valoare disponibila)")


sensor_cols = pd.get_dummies(df.sensor_id, prefix="s", dtype=float)
df = pd.concat([df, sensor_cols], axis=1)
SENSORS = list(sensor_cols.columns)
CAL = ["sin", "cos", "dow"]  # variabile calendaristice


# A = doar satelit | B = satelit + calendar | C = satelit + calendar + senzor | D = calendar + senzor
FEATURES = {
    "RF A": ["col"],
    "RF B": ["col"] + CAL,
    "RF C": ["col"] + CAL + SENSORS,
    "RF D": CAL + SENSORS,
    "GB B": ["col"] + CAL,
    "GB C": ["col"] + CAL + SENSORS,
    "MLP B": ["col"] + CAL,
    "MLP C": ["col"] + CAL + SENSORS,
}
NAMES = ["persistenta", "climatologie", "RF A", "RF B", "RF C", "RF D", "GB B", "GB C", "MLP B", "MLP C"]

# nu pot fi folosite pentru statii noi
NO_NEW_STATION = ["persistenta", "RF C", "RF D", "GB C", "MLP C"]



def make_model(name, seed):
    if name.startswith("MLP"):
        return make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
            n_iter_no_change=20, max_iter=500, random_state=seed))
    if name.startswith("GB"):
        return HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=500, min_samples_leaf=20, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=20, random_state=seed)
    return RandomForestRegressor(N_TREES, min_samples_leaf=5, n_jobs=-1, random_state=seed)


def predict(name, train, test):
    if name == "persistenta":
        # valoarea anterioara sau media datelor de antrenare
        return test["prev"].fillna(train[TARGET].mean()).to_numpy()[None, :]

    if name == "climatologie":
        # media senzorului pe luna sau media lunii pentru senzori noi
        sensor_month = train.groupby(["sensor_id", "month"])[TARGET].mean()
        month = train.groupby("month")[TARGET].mean()
        overall = train[TARGET].mean()
        return np.array([[sensor_month.get((s, m), month.get(m, overall))
                          for s, m in zip(test.sensor_id, test.month)]])

    cols = FEATURES[name]
    return np.array([make_model(name, seed).fit(train[cols], train[TARGET]).predict(test[cols])
                     for seed in range(N_SEEDS)])



def block_ci(x, rng):
    n_days = len(x)
    block = max(1, min(BLOCK, n_days))
    n_blocks = int(np.ceil(n_days / block))
    starts = rng.integers(n_days, size=(N_BOOT, n_blocks)) 
    picks = ((starts[:, :, None] + np.arange(block)) % n_days).reshape(N_BOOT, -1)[:, :n_days]
    means = x[picks].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def evaluate(split, fold, train, test, rows, names):
    days = np.sort(test.date.unique())
    day_idx = np.searchsorted(days, test.date.values)   # indexul zilei

    errors = {name: np.abs(test[TARGET].to_numpy()[None, :] - predict(name, train, test)) for name in names}

    def per_day(err):
        """Calculeaza eroarea medie zilnica."""
        return np.bincount(day_idx, weights=err.mean(axis=0), minlength=len(days)) / np.bincount(day_idx, minlength=len(days))

    rng = np.random.default_rng(0)
    for name in names:
        err = errors[name]
        daily = per_day(err)
        lo, hi = block_ci(daily, rng)
        row = dict(model=name, split=split, fold=fold, n_obs=len(test), n_days=len(days),
                   MAE=err.mean(),
                   MAE_seed_std=err.mean(axis=1).std(ddof=1) if len(err) > 1 else np.nan,
                   IC_lo=lo, IC_hi=hi, dMAE_vs_ref=np.nan, dIC_lo=np.nan, dIC_hi=np.nan)
        if REF in errors and name != REF:
            # diferenta fata de referinta: pozitiv inseamna eroare mai mare
            diff = daily - per_day(errors[REF])
            row["dMAE_vs_ref"] = diff.mean()
            row["dIC_lo"], row["dIC_hi"] = block_ci(diff, rng)
        rows.append(row)



rows = []

# Test 1 - temporal: primele 85% din zile pentru antrenare, ultimele 15% pentru testare
dates = np.sort(df.date.unique())
cut = int(0.85 * len(dates))
print(f"test temporal: {pd.Timestamp(dates[cut]).date()} -> {pd.Timestamp(dates[-1]).date()}")
evaluate("temporal", "test", df[df.date.isin(dates[:cut])], df[df.date.isin(dates[cut:])], rows, NAMES)

# Test 2 - LOYO: fiecare an este folosit pe rand pentru testare
for year in sorted(df.year.unique()):
    evaluate("LOYO", str(year), df[df.year != year], df[df.year == year], rows, NAMES)

# Test 3 - spatial: fiecare statie este exclusa pe rand
for station in sorted(df.station.unique()):
    evaluate("spatial", station, df[df.station != station], df[df.station == station], rows,
             [n for n in NAMES if n not in NO_NEW_STATION])

results = pd.DataFrame(rows)
results.to_csv(DIR / "results_cp3_long.csv", index=False)


pd.set_option("display.width", 200)
for split in ["temporal", "LOYO", "spatial"]:
    r = results[results.split == split]
    if split == "spatial":
        # media pe statii
        g = r.groupby("model").MAE
        print(f"\n== spatial (media pe {r.fold.nunique()} statii; [min, max] intre statii)")
        print(pd.DataFrame({"MAE": g.mean(), "min": g.min(), "max": g.max()}).round(2).to_string())
    else:
        print(f"\n== {split} (MAE, IC95 bootstrap pe blocuri de {BLOCK} zile; dMAE = model - {REF})")
        cols = ["fold", "model", "MAE", "MAE_seed_std", "IC_lo", "IC_hi", "dMAE_vs_ref", "dIC_lo", "dIC_hi"]
        print(r[cols].round(2).to_string(index=False))