import pandas as pd

f = "data/raw/urdu-news-dataset-1M.csv"

# Step 1: First 500 rows read karo
df = pd.read_csv(f, nrows=500, encoding="utf-8")

print("Columns:", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))

# Step 2: Sample CSV save karo
df.to_csv("sample_500.csv", index=False, encoding="utf-8-sig")

# Step 3: Chhoti categories wale columns identify karo
label_cols = [c for c in df.columns if df[c].nunique() < 20]

counts = {c: {} for c in label_cols}
total = 0

# Step 4: Complete dataset chunks mein read karo
for chunk in pd.read_csv(f, chunksize=100_000, encoding="utf-8"):
    total += len(chunk)

    for c in label_cols:
        for k, v in chunk[c].value_counts().items():
            counts[c][k] = counts[c].get(k, 0) + int(v)

# Step 5: Results display karo
print("\nTotal rows:", total)

for c in label_cols:
    print(f"\nColumn: {c}")
    print(counts[c])

print("\nSample file saved successfully!")
