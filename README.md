# NLP Semester Project — Urdu News Category Classification

**Course:** Introduction to NLP (UCP)

## Team

**Team Name:** Team Khabar
| Name | Roll No. |
|---|---|
| Rabi Sandhu | L1F22BSCS0359 |
| Asjad Aleem | L1F22BSCS0609 |
| Omais Rizvi | L1F22BSCS0749 |

## Topic

Classifying Urdu news articles into categories (Business & Economics, Science & Technology, Entertainment, Sports).

## Dataset

- **Name:** Urdu News Dataset 1M
- **Source:** Mendeley Data — https://data.mendeley.com/datasets/834vsxnb99
- **Licence:** CC BY 4.0
- **Size:** 1,038,340 articles in one CSV (~2 GB; 435 MB zipped)
- **Columns used:** `Headline`, `News Text`, `Category` (+ `Source`, `Date`, `URL` for exploration)
- **Our sample:** 40,000 articles, 10,000 per category, drawn with a fixed seed (42) by `src/load_data.py`
- **Citation:** Hussain, K. et al. (2024). *Urdu News Dataset 1M*, Mendeley Data, V4. doi:10.17632/834vsxnb99.4
- **Target:** news category (4 classes)

Raw data is NOT committed. See `data/README.md` for download steps.

## Project phases

- [ ] Phase 1 — Data acquisition & cleaning (`assignment-1`)
- [ ] Phase 2 — Word representations (`assignment-2`)
- [ ] Phase 3 — Classical ML
- [ ] Phase 4 — Deep learning

## Repository structure
```
src/load_data.py          # Task 1: balanced, reproducible 40k sample from the raw CSV
src/regex_patterns.py     # Task 3: regular expressions for our data
src/clean.py              # Task 4: clean_text() - the full cleaning pipeline
src/make_dataset.py       # Task 5: sample -> data/processed/cleaned.csv
notebooks/01_exploration.ipynb         # Task 2: statistics, Zipf, messy examples
notebooks/02_tokenization_regex.ipynb  # Task 3: tokenizers compared, regex demos
notebooks/03_cleaning_pipeline.ipynb   # Task 4: pipeline step by step, before/after vocab
```

## How to run (Phase 1)
```bash
git clone https://github.com/Rabisandhu/nlp-project-khabar.git
cd nlp-project-khabar
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt

# download the dataset (see data/README.md) and put the CSV in data/raw/
python src/load_data.py          # -> data/processed/sample_raw.csv  (40,000 rows)
python src/make_dataset.py       # -> data/processed/cleaned.csv     (clean_text, label)
jupyter notebook                 # open the notebooks in notebooks/
```
Phase 2 loads `data/processed/cleaned.csv` directly.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

## AI assistance

Parts of the code and documentation were drafted with help from Claude (Anthropic) and reviewed/understood by the team.
