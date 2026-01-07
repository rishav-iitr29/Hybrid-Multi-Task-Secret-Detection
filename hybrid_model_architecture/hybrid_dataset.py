import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast
from typing import Dict, Tuple


class SpanAwareDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        tokenizer: RobertaTokenizerFast,
        max_length: int = 512
    ):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Validate required columns
        required_cols = ['code_snippet', 'has_secret', 'secret_span_start', 'secret_span_end', 'entropy']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Statistics
        self.total_samples = len(self.df)
        self.positive_samples = (self.df['has_secret'] == 1).sum()
        self.negative_samples = (self.df['has_secret'] == 0).sum()

        # Track span availability
        self.spans_available = (
            (self.df['has_secret'] == 1) &
            (self.df['secret_span_start'] >= 0) &
            (self.df['secret_span_end'] >= 0)
        ).sum()

        print(f"  Loaded {self.total_samples} samples from {csv_path}")
        print(f"    Positive: {self.positive_samples}")
        print(f"    Negative: {self.negative_samples}")
        print(f"    With span annotations: {self.spans_available}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Extract data
        code_snippet = str(row['code_snippet'])
        has_secret = int(row['has_secret'])
        char_span_start = int(row['secret_span_start'])
        char_span_end = int(row['secret_span_end'])
        entropy = float(row['entropy'])
        entropy = min(max(entropy, 0.0), 6.0)

        # Tokenize with offset mapping
        encoding = self.tokenizer(
            code_snippet,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True  # CRITICAL for span mapping
        )

        # Extract offset mapping
        offset_mapping = encoding['offset_mapping'].squeeze(0)  # [seq_len, 2]

        # Convert character-level span to token-level span
        token_span_start, token_span_end, span_available = self._char_to_token_span(
            char_span_start,
            char_span_end,
            offset_mapping,
            has_secret
        )

        if span_available:
            assert token_span_start <= token_span_end
            assert token_span_start < self.max_length
            assert token_span_end < self.max_length

        return {
            'input_ids': encoding['input_ids'].squeeze(0).long(),
            'attention_mask': encoding['attention_mask'].squeeze(0).long(),
            'label': torch.tensor(has_secret, dtype=torch.long),
            'token_span_start': torch.tensor(token_span_start, dtype=torch.long),
            'token_span_end': torch.tensor(token_span_end, dtype=torch.long),
            'span_available': torch.tensor(span_available, dtype=torch.bool),
            'entropy': torch.tensor(entropy, dtype=torch.float)
        }

    def _char_to_token_span(
        self,
        char_start: int,
        char_end: int,
        offset_mapping: torch.Tensor,
        has_secret: int
    ) -> Tuple[int, int, bool]:

        # Case 1: No secret or invalid character span
        if has_secret == 0 or char_start < 0 or char_end < 0:
            return -1, -1, False

        # Case 2: Empty span
        if char_start >= char_end:
            return -1, -1, False

        # Find tokens that overlap with the character span
        token_start = -1
        token_end = -1

        for token_idx, (token_char_start, token_char_end) in enumerate(offset_mapping):
            # Skip special tokens (offset = (0, 0))
            if token_char_start == 0 and token_char_end == 0:
                continue

            # Find start token (first token that overlaps with char_start)
            if token_start == -1 and token_char_start <= char_start < token_char_end:
                token_start = token_idx

            # Find end token (last token that overlaps with char_end)
            if token_char_start < char_end <= token_char_end:
                token_end = token_idx
                break
            elif token_char_start < char_end:
                # Keep updating end until we find the final token
                token_end = token_idx

        # Case 3: Span not found (likely truncated)
        if token_start == -1 or token_end == -1:
            return -1, -1, False

        # Case 4: Invalid span (end before start)
        if token_end < token_start:
            return -1, -1, False

        # Case 5: Valid span found
        return token_start, token_end, True

    def get_statistics(self) -> Dict:
        return {
            'total_samples': self.total_samples,
            'positive_samples': int(self.positive_samples),
            'negative_samples': int(self.negative_samples),
            'spans_available': int(self.spans_available),
            'positive_ratio': float(self.positive_samples / self.total_samples),
            'span_availability_ratio': float(self.spans_available / max(self.positive_samples, 1))
        }


def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch]),
        'token_span_start': torch.stack([item['token_span_start'] for item in batch]),
        'token_span_end': torch.stack([item['token_span_end'] for item in batch]),
        'span_available': torch.stack([item['span_available'] for item in batch]),
        'entropy': torch.stack([item['entropy'] for item in batch])
    }


# PREPROCESSING: Add Token Spans to CSV
def preprocess_dataset_with_token_spans(
    input_csv: str,
    output_csv: str,
    tokenizer: RobertaTokenizerFast,
    max_length: int = 512
):
    print(f"Preprocessing dataset: {input_csv}")

    df = pd.read_csv(input_csv)

    # Add new columns
    token_span_starts = []
    token_span_ends = []
    span_available_flags = []

    truncated_count = 0

    for idx, row in df.iterrows():
        code_snippet = str(row['code_snippet'])
        has_secret = int(row['has_secret'])
        char_start = int(row['secret_span_start'])
        char_end = int(row['secret_span_end'])

        # Tokenize
        encoding = tokenizer(
            code_snippet,
            max_length=max_length,
            truncation=True,
            return_offsets_mapping=True
        )

        offset_mapping = encoding['offset_mapping']

        # Convert span
        token_start = -1
        token_end = -1
        span_avail = False

        if has_secret == 1 and char_start >= 0 and char_end >= 0 and char_start < char_end:
            for token_idx, (token_char_start, token_char_end) in enumerate(offset_mapping):
                if token_char_start == 0 and token_char_end == 0:
                    continue

                if token_start == -1 and token_char_start <= char_start < token_char_end:
                    token_start = token_idx

                if token_char_start < char_end <= token_char_end:
                    token_end = token_idx
                    break
                elif token_char_start < char_end:
                    token_end = token_idx

            if token_start != -1 and token_end != -1 and token_end >= token_start:
                span_avail = True
            else:
                truncated_count += 1

        token_span_starts.append(token_start)
        token_span_ends.append(token_end)
        span_available_flags.append(span_avail)

    # Add to dataframe
    df['token_span_start'] = token_span_starts
    df['token_span_end'] = token_span_ends
    df['span_available'] = span_available_flags

    # Save
    df.to_csv(output_csv, index=False)

    print(f" Preprocessed dataset saved to {output_csv}")
    print(f"  Samples: {len(df)}")
    print(f"  Truncated spans: {truncated_count}")
    print(f"  Span availability: {sum(span_available_flags)} / {(df['has_secret']==1).sum()}")


# TESTING
if __name__ == "__main__":
    print("TESTING SPAN-AWARE DATASET")

    # Create dummy CSV for testing
    import tempfile
    import os

    dummy_data = pd.DataFrame({
        'code_snippet': [
            'api_key = "AKIA1234567890"',
            'def hello(): return "world"',
            'secret = "ghp_1234567890abcdef"',
        ],
        'has_secret': [1, 0, 1],
        'secret_span_start': [11, -1, 10],
        'secret_span_end': [25, -1, 30],
        'entropy': [4.2, 0.0, 3.8]
    })

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        dummy_data.to_csv(f.name, index=False)
        temp_csv = f.name

    try:
        # Test dataset
        print("\n1. Loading tokenizer...")
        tokenizer = RobertaTokenizerFast.from_pretrained('microsoft/codebert-base')

        print("\n2. Creating dataset...")
        dataset = SpanAwareDataset(temp_csv, tokenizer, max_length=128)

        print("\n3. Testing sample retrieval...")
        sample = dataset[0]
        print(f"   Input IDs shape: {sample['input_ids'].shape}")
        print(f"   Label: {sample['label']}")
        print(f"   Token span: [{sample['token_span_start']}, {sample['token_span_end']}]")
        print(f"   Span available: {sample['span_available']}")

        print("\n4. Dataset statistics:")
        stats = dataset.get_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")

        print("\n5. Testing DataLoader...")
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

        batch = next(iter(loader))
        print(f"   Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"   Batch labels: {batch['label']}")
        print(f"   Batch span_available: {batch['span_available']}")

        print("\n  All tests passed!")

    finally:
        os.unlink(temp_csv)
