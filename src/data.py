"""Person-level EICV7 table for the secondary-progression early-warning score.

The modelling files are the public-use microdata, not the DDI codebook PDF.
NISR serves those files only after a catalog login, so this module reads them
from data/raw once they have been downloaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_PATH = ROOT / "data" / "processed" / "person_risk.csv"

# Ages covered by the published secondary net attendance rate.
MIN_AGE = 12
MAX_AGE = 17

# s4aq8: 5 = lower secondary, 6 = upper secondary.
SECONDARY_LEVELS = {5, 6}

# Predictors. Current attendance (s4aq7, s4aq8) defines the target, so those
# columns stay out of this list. Reasons for leaving school stay out too.
FEATURE_COLUMNS = [
    "age",
    "sex",
    "province",
    "district",
    "urban_rural",
    "ever_attended",
    "highest_class",
    "years_completed",
    "class_2022",
    "repeated_prior",
    "consumption_quintile",
    "household_education_expenditure",
]


def _find_file(tokens: tuple[str, ...]) -> Path | None:
    if not RAW_DIR.exists():
        return None
    suffixes = {".dta", ".sav", ".csv"}
    matches = [
        path
        for path in RAW_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in suffixes
        and all(token in path.name.lower() for token in tokens)
    ]
    return matches[0] if matches else None


def locate_sources() -> tuple[Path, Path]:
    person = _find_file(("person",))
    poverty = _find_file(("poverty",))
    if person is None or poverty is None:
        raise FileNotFoundError(
            "The EICV7 microdata is not in data/raw.\n"
            "The PDF named ddi-documentation-english-119.pdf is the codebook only.\n"
            "Download the data files after logging in at\n"
            "https://microdata.statistics.gov.rw/index.php/catalog/119/get_microdata\n"
            "Choose Stata or CSV, then put these two files in data/raw:\n"
            "  CS_S0_S1_S2_S3_S4_S6A_S6B_S6C_Person\n"
            "  CS_EICV7_poverty_file\n"
            "Then run: .\\.venv\\Scripts\\python -m src.train --epochs 1"
        )
    return person, poverty


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".dta":
        return pd.read_stata(path, convert_categoricals=False)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".sav":
        import pyreadstat

        frame, _ = pyreadstat.read_sav(path)
        return frame
    raise ValueError(f"Unsupported file type: {path.name}")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_frame(person: pd.DataFrame, poverty: pd.DataFrame) -> pd.DataFrame:
    needed = [
        "hhid",
        "pid",
        "s1q1",
        "s1q3y",
        "province",
        "district",
        "ur",
        "s4aq1",
        "s4aq2",
        "s4aq2a",
        "s4aq6a",
        "s4aq6d",
        "s4aq7",
        "s4aq8",
    ]
    missing = [name for name in needed if name not in person.columns]
    if missing:
        raise RuntimeError(f"Person file is missing columns: {missing}")
    for name in ("hhid", "quintile", "exp1"):
        if name not in poverty.columns:
            raise RuntimeError(f"Poverty file is missing column: {name}")

    people = person.loc[:, needed].copy()
    people["hhid"] = people["hhid"].astype(str)
    welfare = poverty.loc[:, ["hhid", "quintile", "exp1"]].copy()
    welfare["hhid"] = welfare["hhid"].astype(str)
    welfare = welfare.drop_duplicates("hhid")
    merged = people.merge(welfare, on="hhid", how="left")

    merged["age"] = _numeric(merged["s1q3y"])
    sample = merged[merged["age"].between(MIN_AGE, MAX_AGE)].copy()
    if sample.empty:
        raise RuntimeError("No people aged 12–17 were found in the person file.")

    attended = _numeric(sample["s4aq7"])
    level = _numeric(sample["s4aq8"])
    in_secondary = attended.eq(1) & level.isin(SECONDARY_LEVELS)
    # 1 = early-warning case: not in secondary school.
    sample["at_risk"] = (~in_secondary).astype(int)

    sample["sex"] = _numeric(sample["s1q1"])
    sample["province"] = _numeric(sample["province"])
    sample["district"] = _numeric(sample["district"])
    sample["urban_rural"] = _numeric(sample["ur"])
    sample["ever_attended"] = _numeric(sample["s4aq1"]).eq(1).astype(int)
    sample["highest_class"] = _numeric(sample["s4aq2"])
    sample["years_completed"] = _numeric(sample["s4aq2a"])
    sample["class_2022"] = _numeric(sample["s4aq6a"])
    sample["repeated_prior"] = _numeric(sample["s4aq6d"]).notna().astype(int)
    sample["consumption_quintile"] = _numeric(sample["quintile"])
    sample["household_education_expenditure"] = np.log1p(_numeric(sample["exp1"]).clip(lower=0))

    keep = ["hhid", "pid", "at_risk", *FEATURE_COLUMNS]
    return sample.loc[:, keep].reset_index(drop=True)


def load_frame() -> pd.DataFrame:
    person_path, poverty_path = locate_sources()
    return build_frame(read_table(person_path), read_table(poverty_path))


def save_frame(frame: pd.DataFrame) -> Path:
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PROCESSED_PATH, index=False)
    return PROCESSED_PATH
