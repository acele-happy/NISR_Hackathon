# Secondary progression early warning (EICV7)

Track 3. The score flags people aged 12–17 who are not attending secondary school.

## Data

The file `ddi-documentation-english-119.pdf` is the codebook. The model reads the microdata, which NISR releases after login:

https://microdata.statistics.gov.rw/index.php/catalog/119/get_microdata

Download Stata or CSV and place both files in `data/raw`:

- `CS_S0_S1_S2_S3_S4_S6A_S6B_S6C_Person`
- `CS_EICV7_poverty_file`

Target: `at_risk = 1` when the person is not attending lower or upper secondary (`s4aq8` codes 5 and 6) in the last 12 months (`s4aq7` = 1 means yes).

Inputs: age, sex, province, district, urban/rural, ever attended, highest class completed, years completed, class attended in 2022/23, a prior-repetition flag, consumption quintile, and household education expenditure. Current attendance and reasons for leaving school are not used as inputs.

## Run

```powershell
cd E:\Projects\nisr-education-attendance
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m src.train --epochs 1
.\.venv\Scripts\tensorboard --logdir outputs\runs
```

`--epochs 1` fits one boosting round. TensorBoard records train loss, validation loss, ROC-AUC, PR-AUC, and the hyperparameters.
