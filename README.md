# Cigarette and Marijuana Use Patterns — NSDUH 2012

Descriptive, inferential and predictive analysis of cigarette and marijuana use using the
2012 National Survey on Drug Use and Health (NSDUH) public-use file. Course project for
Data Analysis, Galatasaray University.

## Contents

| File | Description |
|---|---|
| `descriptive_figures.py` | Figures 1–8 (prevalence, initiation age, recent use, risk perception, dependence, mental health, demographics, acquisition) |
| `statistical_figures.py` | Figures 9–14 (statistical analysis: age, gender, risk, dependence, mental health, co-occurrence) |
| `regression_models.py` | Logistic regression models and Figures 15–18 |
| `rapport.tex` | Main report (IEEE format) |
| `appendix.tex` | Appendix: predictive modeling |
| `figures/` | Generated figures |

## Data

The data are **not included** in this repository. ICPSR's terms of use do not allow
redistribution. To reproduce the analysis:

1. Download study **ICPSR 34933** (NSDUH 2012) from
   [ICPSR](https://www.icpsr.umich.edu/web/ICPSR/studies/34933) and agree to its terms of use.
2. Place `34933-0001-Data.tsv` (from the `DS0001` folder) in the project root.

Data citation: United States Department of Health and Human Services. Substance Abuse and
Mental Health Services Administration. Center for Behavioral Health Statistics and Quality.
*National Survey on Drug Use and Health, 2012.* ICPSR34933-v1. Ann Arbor, MI:
Inter-university Consortium for Political and Social Research [distributor].

## Running

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python descriptive_figures.py
python statistical_figures.py
python regression_models.py
```

Compile the reports with `pdflatex rapport.tex` and `pdflatex appendix.tex` (run twice to
resolve cross-references).

## Notes

- All statistics are unweighted sample estimates; they are not nationally representative
  population estimates (NSDUH oversamples ages 12–25).
- The mental health items were asked only of adults, so analyses and models that use the
  mental distress score cover respondents aged 18 and older.
