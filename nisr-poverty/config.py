"""Dataset-specific names. EDIT THESE after running:  python prepare_data.py --inspect

The EICV7 catalog lists file names (e.g. CS_EICV7_poverty_file) but this code
cannot know the exact column names inside each file until you open them, so
everything that depends on them lives here and nowhere else.
"""
from pathlib import Path

RAW_DIR = Path("data/raw")            # put the downloaded EICV7 files here
PROCESSED_DIR = Path("data/processed")
TABLE_PATH = PROCESSED_DIR / "model_table.csv"

# --- Keys and targets (check against --inspect output) -----------------------
HHID = "hhid"                 # household id (confirmed in the catalog)
CLUSTER = "clust"             # enumeration area (confirmed in the catalog)
POVERTY = "poverty"           # total poverty headcount flag, 1 = poor (confirmed)
WEIGHT = "hhweight"           # household weight        <- CHECK exact name
CONSUMPTION = "consae"        # consumption per adult equivalent, Jan 2024 prices <- CHECK exact name

# --- Files (stem of file name, any of .csv/.dta/.sav) ------------------------
POVERTY_FILE = "CS_EICV7_poverty_file"
HOUSEHOLD_FILE = "CS_S01_S5_S7_Household"
PERSON_FILE = "CS_S0_S1_S2_S3_S4_S6A_S6B_S6C_Person"
SERVICES_FILE = "CS_S5F_access_to_services"
# Multi-row-per-household files that are collapsed to one row per household.
AGGREGATED_FILES = [
    "CS_S10A1_A2_credits",
    "CS_S10B_durables",
    "CS_S10C_Savings",
]
# Programme files: NOT features. Kept in a separate audit table.
PROGRAMME_FILES = [
    "CS_S9D1_Direct_Support",
    "CS_S9D2_Classic_Public_Work",
    "CS_S9D3_Expanded_Public_Work",
    "CS_S9D4_NSDS",
    "CS_S9D5_Financial_Services",
]

# --- Leakage safety net -------------------------------------------------------
# Any feature column whose lower-cased name contains one of these is dropped.
# This is a backstop, not a substitute for reading the data dictionary.
LEAKAGE_PATTERNS = ["exp", "cons", "pov", "quint", "food", "aeq", "ae_", "pcexp", "poor", "weight", "wgt"]
# Columns that are never features.
NON_FEATURES = {HHID, CLUSTER, POVERTY, WEIGHT, CONSUMPTION, "pid"}

# --- Model / training defaults -----------------------------------------------
N_FOLDS = 5
SEED = 42
MIN_CATEGORY_COUNT = 20       # rarer categories are merged into "other"
