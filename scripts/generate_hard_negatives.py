import csv
import random
import string
import subprocess
import tempfile
import os
import math
from typing import List, Dict

#CONFIG
OUTPUT_CSV = "../data/final/hard_negatives.csv"
TARGET_COUNT = 1500


FORBIDDEN_SECRET_WORDS = [
    "api_key", "apikey", "secret", "access_token",
    "auth", "token", "private_key", "credential"
]

def is_semantically_safe(snippet: str) -> bool:
    s = snippet.lower()
    return not any(w in s for w in FORBIDDEN_SECRET_WORDS)


# UTILS
def sample_length():
    r = random.random()
    if r < 0.4:
        return 32
    elif r < 0.6:
        return 40
    elif r < 0.8:
        return 24
    else:
        return random.randint(16, 64)

def entropy(s: str) -> float:
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)

def rand_hex(n):
    return ''.join(random.choices("abcdef0123456789", k=n))

def rand_b64(n):
    return ''.join(random.choices(string.ascii_letters + string.digits + "+/", k=n))


# TEMPLATES
TEMPLATES = [
    'request_id = "{v}"',
    'trace_id = "{v}"',
    'cache_key = "{v}"',
    'get_user_by_id("{v}")',
    'log_request("{v}")',
    'fetch_by_hash("{v}")',
    '{{ "request_metadata": {{ "trace": "{v}" }} }}',
    'build: checksum: "{v}"'
]

STOPWORDS = [
    "about", "account", "activity", "adapter", "active",
    "action", "addon", "admin", "root", "014df517-39d1-4453-b7b3-9930c563627c",
    "abcdefghijklmnopqrstuvwxyz", "id", "uuid", "guid", "request_id", "correlation_id", "session_id", "trace_id",
    "hash", "md5", "sha1", "sha256", "checksum", "etag", "fingerprint", "version_hash", "build_number", "commit_sha",
    "deployment_id", "css_class", "element_id", "color_hex", "asset_name", "slug", "salt", "nonce", "iv",
    "build_id", "job_id", "cache_key", "object_id"
]

PLACEHOLDERS = [
    "YOUR_API_KEY", "example", "sample", "test", "dummy", "mock", "placeholder", "fake", "your_key_here", "REDACTED", "TODO",
    "REPLACE_ME",
    "INSERT_TOKEN_HERE",
    "<API_KEY>"
]

DEFAULT_CREDS = [
    "password123",
    "admin123",
    "root",
    "test123"
]


# HARD NEGATIVE GENERATORS
def gen_fake_entropy():
    n = sample_length()
    return random.choice([rand_hex(n), rand_b64(n)]), "fake_entropy"

def gen_hash():
    # hashes should look structured
    return rand_hex(32), "hash_or_id"

def gen_placeholder():
    return random.choice(PLACEHOLDERS), "placeholder"

def gen_stopword():
    return random.choice(STOPWORDS), "stopword"

def gen_default_cred():
    return random.choice(DEFAULT_CREDS), "default_credential"

GENERATORS = [
    gen_fake_entropy,
    gen_hash,
    gen_placeholder,
    gen_stopword,
    gen_default_cred
]


# MAIN LOOP
rows: List[Dict] = []

ATTEMPTS = 0
MAX_ATTEMPTS = TARGET_COUNT * 10

while len(rows) < TARGET_COUNT and ATTEMPTS < MAX_ATTEMPTS:
    ATTEMPTS += 1

    value, hn_type = random.choice(GENERATORS)()
    template = random.choice(TEMPLATES)
    snippet = template.format(v=value)

    if not is_semantically_safe(snippet):
        continue

    rows.append({
        "source": "hard_negative",
        "code_snippet": snippet,
        "secret_span_start": -1,
        "secret_span_end": -1,
        "file_path": "-1",
        "line_number": -1,
        "length": -1,
        "has_secret": 0,
        "hard_negative": 1,
        "entropy": round(entropy(value), 3),
        "rule": hn_type,
    })

print(f"✓ Generated {len(rows)} hard negatives")
print(f"  Attempts: {ATTEMPTS}")


# SAVE

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"✓ Saved to {OUTPUT_CSV}")