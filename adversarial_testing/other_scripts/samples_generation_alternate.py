import csv
import base64
import random
import math


# CONFIG
INPUT_SECRETS_FILE = "../secrets.txt"
OUTPUT_CSV = "../adversarial_data/adversarial_eval_with_negatives_alternate.csv"


random.seed(42)

# UTILS
def entropy(s: str) -> float:
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


# Seen Obfuscations (Used in training)
SEEN_OBFUSCATIONS = {
    "base64": lambda s: base64.b64encode(s.encode()).decode(),
    "hex": lambda s: s.encode().hex(),
    "concat_split": lambda s: f'"{s[:len(s)//2]}" + "{s[len(s)//2:]}"',
    "env_mock": lambda s: f'os.getenv("API_KEY", "{s}")',
    "commented": lambda s: f"// TODO remove key {s}",
}


# Unseen Obfuscations (adversarial benchmarking)
UNSEEN_OBFUSCATIONS = {
    "charcode_build": lambda s:
        "''.join([chr(x) for x in %s])" % [ord(c) for c in s],

    "byte_array": lambda s:
        "bytes(%s).decode()" % [ord(c) for c in s],

    "dict_lookup": lambda s:
        f'cfg = {{"token": "{s}"}}\napi_key = cfg["token"]',

    "list_join": lambda s:
        "''.join(%s)" % list(s),

    "lambda_return": lambda s:
        f'get_key = lambda: "{s}"\napi_key = get_key()',

    "default_arg": lambda s:
        f'def connect(key="{s}"):\n    pass',

    "ternary": lambda s:
        f'key = "{s}" if DEBUG else None',

    "try_except_fallback": lambda s:
        f'try:\n    key = load_key()\nexcept Exception:\n    key = "{s}"'
}


# Adversarial Negatives
NEGATIVE_GENERATORS = {
    "random_entropy": lambda: ''.join(random.choices(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=32
    )),

    "fake_base64": lambda: base64.b64encode(
        b"thisisnotasecret"
    ).decode(),

    "uuid_like": lambda: "550e8400-e29b-41d4-a716-446655440000",

    "placeholder": lambda: "YOUR_API_KEY_HERE",

    "hex_hash": lambda: "9e107d9d372bb6826bd81d3542a419d6",
}


# Templates
TEMPLATE_POOL = [
    'API_KEY = {v}',
    'SECRET = {v}',
    'ACCESS_TOKEN = {v}',
    'auth_token = {v}',
    'config = {{"api_key": {v}}}',
    'config["api_key"] = {v}',
    'settings["token"] = {v}',
    'def load_key():\n    return {v}',
    'client = Client({v})',
    'connect({v})',
    'if DEBUG:\n    api_key = {v}',
    'try:\n    secret = {v}\nexcept Exception:\n    pass',
    'os.environ["API_KEY"] = {v}',
    'headers = {{"Authorization": "Bearer " + {v}}}',
    'headers["X-API-KEY"] = {v}',
    'self.token = {v}',
    'const API_KEY = {v}',
    'module.exports = {{ key: {v} }}',
]
TEMPLATES_NEGATIVES = [
    'span_id = "{v}"',
    'traceparent = "{v}"',
    'correlation_id = "{v}"',
    'x_request_id = "{v}"',
    'otel_trace_id = "{v}"',
    'otel_span_id = "{v}"',
    'jaeger_trace = "{v}"',
    'datadog_span = "{v}"',
    'build_id = "{v}"',
    'job_id = "{v}"',
    'pipeline_run = "{v}"',
    'artifact_sha = "{v}"',
    'commit_hash = "{v}"',
    'image_digest = "{v}"',
    'release_fingerprint = "{v}"',
    'nonce = "{v}"',
    'salt = "{v}"',
    'iv = "{v}"',
    'signature = "{v}"',
    'digest = "{v}"',
    'checksum = "{v}"'
]

def wrap(value_expr):
    tpl = random.choice(TEMPLATE_POOL)
    return tpl.format(v=value_expr)

def wrap_negatives(value_expr):
    tpl = random.choice(TEMPLATES_NEGATIVES)
    return tpl.format(v=value_expr)


# LOAD SECRETS
with open(INPUT_SECRETS_FILE) as f:
    secrets = [l.strip() for l in f if l.strip()]

assert len(secrets) >= 10, "Need at least 10 secrets"


# GENERATION
rows = []

def add_negative(snippet, obf_type):
    rows.append({
        "code_snippet": snippet,
        "has_secret": 0,
        "secret_span_start": -1,
        "secret_span_end": -1,
        "entropy": round(entropy(snippet), 4),
        "obfuscation_type": f"negative:{obf_type}"
    })

def add_sample(snippet, secret, obf_type):
    try:
        start = snippet.index(secret)
        end = start + len(secret)
    except ValueError:
        return

    rows.append({
        "code_snippet": snippet,
        "has_secret": 1,
        "secret_span_start": start,
        "secret_span_end": end,
        "entropy": round(entropy(secret), 4),
        "obfuscation_type": obf_type
    })

for secret in secrets:
    # seen
    for name, fn in SEEN_OBFUSCATIONS.items():
        try:
            snippet = wrap(fn(secret))
            add_sample(snippet, secret, f"seen:{name}")
        except Exception:
            continue

    # unseen
    for name, fn in UNSEEN_OBFUSCATIONS.items():
        try:
            snippet = wrap(fn(secret))
            add_sample(snippet, secret, f"unseen:{name}")
        except Exception:
            continue

# adversarial negatives
for _ in range(len(rows)):
    name, gen = random.choice(list(NEGATIVE_GENERATORS.items()))
    fake_value = gen()
    snippet = wrap_negatives(fake_value)
    add_negative(snippet, name)


# SAVE
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "code_snippet",
            "has_secret",
            "secret_span_start",
            "secret_span_end",
            "entropy",
            "obfuscation_type"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} adversarial samples")
print(f"Saved to {OUTPUT_CSV}")