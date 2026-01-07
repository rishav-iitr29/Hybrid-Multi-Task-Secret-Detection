import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
INPUT_CSV = "final_dataset_span_fixed.csv"
OUT_DIR = "final_final_data"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42

# =========================
# LOAD
# =========================
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} samples")

# =========================
# TRAIN / TEMP SPLIT
# =========================
train_df, temp_df = train_test_split(
    df,
    test_size=(1 - TRAIN_RATIO),
    stratify=df["has_secret"],
    random_state=RANDOM_STATE
)

# =========================
# VAL / TEST SPLIT
# =========================
val_df, test_df = train_test_split(
    temp_df,
    test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)),
    stratify=temp_df["has_secret"],
    random_state=RANDOM_STATE
)

# =========================
# SAVE
# =========================
train_df.to_csv(f"{OUT_DIR}/train.csv", index=False)
val_df.to_csv(f"{OUT_DIR}/val.csv", index=False)
test_df.to_csv(f"{OUT_DIR}/test.csv", index=False)

print("\nSplit summary:")
print(f"Train: {len(train_df)}")
print(f"Val:   {len(val_df)}")
print(f"Test:  {len(test_df)}")

print("\nLabel distribution:")
for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(f"{name}:")
    print(split["has_secret"].value_counts(normalize=True).round(3))