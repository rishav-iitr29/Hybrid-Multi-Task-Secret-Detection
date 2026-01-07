import pandas as pd

# CONFIG

INPUT_CSV = "../data/train_negatives.csv"
OUTPUT_CSV = "../data/final/train_negatives_trimmed.csv"

MAX_CHARS = 800
MIN_CHARS = 20
MAX_LINES = 8

# LOAD
df = pd.read_csv(INPUT_CSV)

print("\nBEFORE")
print(df["code_snippet"].str.len().describe())

# CLEAN
def trim_snippet(snippet: str) -> str | None:
    if not isinstance(snippet, str):
        return None

    # Normalize whitespace
    snippet = snippet.replace("\r\n", "\n").strip()

    # Line-based trimming (keeps semantic locality)
    lines = snippet.split("\n")[:MAX_LINES]
    snippet = "\n".join(lines)

    # Length-based trimming
    if len(snippet) > MAX_CHARS:
        snippet = snippet[:MAX_CHARS]

    # Final sanity check
    if len(snippet) < MIN_CHARS:
        return None

    return snippet

df["code_snippet"] = df["code_snippet"].apply(trim_snippet)

# Drop rows that became invalid
before_rows = len(df)
df = df.dropna(subset=["code_snippet"])
after_rows = len(df)

# REPORT

print("\nAFTER")
print(df["code_snippet"].str.len().describe())

print("\nRows dropped:", before_rows - after_rows)
print("Final rows:", after_rows)

# SAVE

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nTrimmed dataset saved to {OUTPUT_CSV}")