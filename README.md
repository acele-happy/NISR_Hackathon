# Secondary school early-warning score

This project builds a simple risk score for Rwanda’s NISR 2026 Big Data Hackathon, Track 3 (education).

It looks at young people aged 12–17 in the EICV7 household survey and flags those who are not in secondary school. District education offices could use that list to follow up before a learner drops out for good.

The version in this repository is a pipeline check: the model is trained for one round only. The printed scores show that the code runs. They are not the final hackathon result.

## What you need

Python is already set up in the `.venv` folder inside this project. You also need two survey files from NISR. The PDF codebook is not enough.

1. Open the EICV7 catalog and log in: https://microdata.statistics.gov.rw/index.php/catalog/119
2. Open the **GET MICRODATA** tab.
3. Download the Stata or CSV version.
4. From the download, copy these two files into `data/raw`:
   - `CS_S0_S1_S2_S3_S4_S6A_S6B_S6C_Person` (one row per person, including schooling)
   - `CS_EICV7_poverty_file` (one row per household, including welfare)

File names can end in `.dta`, `.csv`, or `.sav`. Keep both files in `data/raw`. Do not commit them. NISR licenses this data, and the files contain household identifiers.

## How to run it

In PowerShell:

```powershell
cd E:\Projects\nisr-education-attendance
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m src.train --epochs 1
```

The first run installs the libraries. The second command builds the table, holds out 20% of people for checking, and fits one XGBoost round.

When it finishes, the terminal prints three numbers:

- **ROC-AUC** — how well the score separates at-risk learners from the others
- **PR-AUC** — the same idea, with extra attention to the at-risk group
- **F1** — a balance of how many flagged people are truly at risk

Then open the training curves:

```powershell
.\.venv\Scripts\tensorboard --logdir outputs\runs
```

Go to the address it prints, usually http://localhost:6006. You should see training loss, validation loss, AUC, and the model settings.

## What the score means

A person is marked **at risk** when they are aged 12–17 and are not attending lower or upper secondary school.

The model uses information a school office could know without already knowing today’s attendance result:

- age, sex, province, district, and whether the area is urban or rural
- whether they have ever been to school
- the highest class they completed, and how many years that took
- the class they attended in 2022/23
- whether they had repeated a class before
- their household’s consumption quintile
- how much the household spends on education

Today’s attendance answer and the reasons someone left school are used only to build the true label. They are not fed in as clues.

## Files worth opening

| File | What it is |
| --- | --- |
| `src/data.py` | Loads the two survey files and builds the 12–17 table |
| `src/model.py` | The XGBoost settings |
| `src/train.py` | Splits the data, trains one round, writes logs |
| `outputs/placeholder_metrics.json` | The summary numbers from the latest run |
| `README.md` | This guide |

`data/raw`, `data/processed`, TensorBoard logs, and the person-level score file stay on your computer. They are listed in `.gitignore`.
