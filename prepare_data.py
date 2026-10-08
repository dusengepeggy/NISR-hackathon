"""Build one household-level modelling table from the raw EICV7 files.

    python prepare_data.py --inspect      # list columns of every file in data/raw
    python prepare_data.py                # build data/processed/model_table.csv
    python prepare_data.py --synthetic    # fake data, to test the pipeline only
"""
import argparse
import sys

import numpy as np
import pandas as pd

import config as C


def find_file(stem):
    for ext in (".csv", ".dta", ".sav"):
        p = C.RAW_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def read_any(path):
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".dta":
        return pd.read_stata(path, convert_categoricals=True)
    if path.suffix == ".sav":
        return pd.read_spss(path, convert_categoricals=True)
    raise ValueError(f"Unsupported file type: {path}")


def inspect():
    files = sorted(p for p in C.RAW_DIR.glob("*") if p.suffix in {".csv", ".dta", ".sav"})
    if not files:
        sys.exit(f"No .csv/.dta/.sav files in {C.RAW_DIR}. Download the EICV7 files first.")
    for p in files:
        df = read_any(p)
        print(f"\n=== {p.name}: {len(df):,} rows x {df.shape[1]} cols")
        for col in df.columns:
            print(f"  {col:<28} {str(df[col].dtype):<10} n_unique={df[col].nunique():<6} missing={df[col].isna().mean():.1%}")


def to_household_level(df, prefix):
    """Collapse a multi-row-per-household file to one row: numeric -> mean and sum,
    categorical -> most frequent value, plus a row count."""
    df = df.drop(columns=[c for c in ("pid", C.CLUSTER) if c in df.columns])
    num = df.select_dtypes("number").columns.difference([C.HHID])
    cat = df.columns.difference(num).difference([C.HHID])
    g = df.groupby(C.HHID)
    parts = [g.size().rename(f"{prefix}_rows")]
    if len(num):
        m = g[num].mean().add_prefix(f"{prefix}_mean_")
        parts.append(m)
    if len(cat):
        mode = g[cat].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan).add_prefix(f"{prefix}_mode_")
        parts.append(mode)
    return pd.concat(parts, axis=1).reset_index()


def drop_leakage(df):
    dropped = []
    for col in list(df.columns):
        if col in C.NON_FEATURES:
            continue
        low = col.lower()
        if any(pat in low for pat in C.LEAKAGE_PATTERNS):
            dropped.append(col)
    if dropped:
        print(f"[leakage guard] dropping {len(dropped)} columns: {dropped}")
        print("  Review this list against the data dictionary; edit LEAKAGE_PATTERNS if it is wrong.")
    return df.drop(columns=dropped)


def build_real():
    pov_path = find_file(C.POVERTY_FILE)
    if pov_path is None:
        sys.exit(f"Missing {C.POVERTY_FILE} in {C.RAW_DIR}")
    pov = read_any(pov_path)
    for needed in (C.HHID, C.CLUSTER, C.POVERTY, C.WEIGHT, C.CONSUMPTION):
        if needed not in pov.columns:
            sys.exit(f"Column '{needed}' not found in the poverty file. Run --inspect and fix config.py.\n"
                     f"Columns: {list(pov.columns)}")
    # Targets and weights only; every other poverty-file column is consumption-derived.
    table = pov[[C.HHID, C.CLUSTER, C.POVERTY, C.WEIGHT, C.CONSUMPTION]].copy()

    features = []
    for stem in (C.HOUSEHOLD_FILE, C.SERVICES_FILE):
        p = find_file(stem)
        if p is None:
            print(f"[warn] missing {stem}, skipping")
            continue
        features.append(drop_leakage(read_any(p)).drop(columns=[C.CLUSTER], errors="ignore"))
    p = find_file(C.PERSON_FILE)
    if p is not None:
        features.append(drop_leakage(to_household_level(read_any(p), "person")))
    for stem in C.AGGREGATED_FILES:
        p = find_file(stem)
        if p is None:
            print(f"[warn] missing {stem}, skipping")
            continue
        features.append(drop_leakage(to_household_level(read_any(p), stem.split("_")[1].lower())))

    for f in features:
        if C.HHID not in f.columns:
            print("[warn] a feature file has no household id; skipped")
            continue
        table = table.merge(f.drop_duplicates(C.HHID), on=C.HHID, how="left")

    # Programme participation: audit table only, never merged into features.
    prog = []
    for stem in C.PROGRAMME_FILES:
        p = find_file(stem)
        if p is not None:
            ids = read_any(p)[C.HHID].unique()
            prog.append(pd.DataFrame({C.HHID: ids, stem: 1}))
    if prog:
        audit = prog[0]
        for d in prog[1:]:
            audit = audit.merge(d, on=C.HHID, how="outer")
        audit.fillna(0).to_csv(C.PROCESSED_DIR / "programme_audit.csv", index=False)
    return table


def build_synthetic(n=3000, seed=0):
    """Fake data with the same shape as the real table. For smoke tests only."""
    rng = np.random.default_rng(seed)
    clust = rng.integers(0, 300, n)
    province = rng.choice(["Kigali", "South", "West", "North", "East"], n)
    urban = rng.choice(["urban", "rural"], n, p=[0.3, 0.7])
    roof = rng.choice(["iron", "tile", "thatch", "other"], n, p=[0.6, 0.2, 0.1, 0.1])
    water = rng.choice(["piped", "protected", "unprotected", "surface"], n)
    hh_size = rng.integers(1, 10, n)
    rooms = rng.integers(1, 7, n)
    assets = rng.poisson(3, n)
    land_ha = np.where(rng.random(n) < 0.2, np.nan, rng.gamma(1.0, 0.4, n))
    head_age = rng.integers(18, 85, n)
    latent = (0.5 * hh_size - 0.6 * rooms - 0.5 * assets + 1.2 * (urban == "rural") + 1.0 * (roof == "thatch")
              + 0.8 * (water == "surface") + rng.normal(0, 1.2, n))
    log_cons = 12.8 - 0.25 * latent + rng.normal(0, 0.15, n)
    poor = (latent > np.quantile(latent, 0.72)).astype(int)
    return pd.DataFrame({
        C.HHID: np.arange(n), C.CLUSTER: clust, C.POVERTY: poor,
        C.WEIGHT: rng.uniform(0.5, 2.0, n) * 100, C.CONSUMPTION: np.exp(log_cons),
        "province": province, "urban_rural": urban, "roof_material": roof, "water_source": water,
        "household_size": hh_size, "rooms": rooms, "durable_count": assets,
        "land_ha": land_ha, "head_age": head_age,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    if a.inspect:
        return inspect()
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table = build_synthetic() if a.synthetic else build_real()
    table.to_csv(C.TABLE_PATH, index=False)
    print(f"Wrote {C.TABLE_PATH}: {len(table):,} households x {table.shape[1]} columns")
    print(f"Poverty prevalence (unweighted): {table[C.POVERTY].mean():.1%}")


if __name__ == "__main__":
    main()
