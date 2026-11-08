# Data

Raw data is NOT committed (it is ~2 GB). To get it:

1. Open https://data.mendeley.com/datasets/834vsxnb99 and click **Download All** (435 MB zip).
2. Extract it and put `urdu-news-dataset-1M.csv` in `data/raw/`.
3. Run `python src/load_data.py` then `python src/make_dataset.py`.

Generated files (also not committed):
- `data/processed/sample_raw.csv` - balanced 40k sample
- `data/processed/cleaned.csv` - cleaned text + label (input for Phase 2)
