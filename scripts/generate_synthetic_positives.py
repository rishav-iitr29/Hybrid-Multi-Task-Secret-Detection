import csv
import random
import base64
import urllib.parse
import string
import math
from collections import Counter


# ============================================================
# CONFIG
# ============================================================

GITLEAKS_TOML = "../data/gitleaks.toml"
OUTPUT_CSV = "synthetic_positives.csv"

R = 2363   # Total synthetic positives

# RULE LIST

ALL_RULES = {
    # Here a lot of rules are same - because gitleaks.toml doesn't have a regex for that and hence I used from the available ones

    # Auth / Tokens
    "jwt", "generic-api-key", "github-oauth", "github-app-token",
    "linkedin-client-id", "linkedin-client-secret", "facebook-secret",
    "microsoft-teams-webhook", "generic-api-key",
    "grafana-service-account-token", "openshift-user-token",
    "gitlab-incoming-mail-token", "gitlab-runner-authentication-token",
    "npm-access-token",

    # Infra / DB
    "generic-api-key", "generic-api-key", "generic-api-key", "generic-api-key", "generic-api-key", "generic-api-key",
    "generic-api-key", "generic-api-key", "kubernetes-secret-yaml",
    "gcp-api-key", "generic-api-key", "private-key",

    # Vendor APIs
    "generic-api-key", "generic-api-key", "generic-api-key", "generic-api-key", "generic-api-key",
    "generic-api-key", "generic-api-key", "generic-api-key", "generic-api-key", "generic-api-key",
    "generic-api-key", "generic-api-key", "generic-api-key", "generic-api-key", "generic-api-key",
    "generic-api-key", "generic-api-key", "generic-api-key", "generic-api-key", "generic-api-key",
    "generic-api-key", "generic-api-key", "generic-api-key", "generic-api-key", "generic-api-key", "generic-api-key",
    "generic-api-key", "generic-api-key", "generic-api-key", "generic-api-key", "generic-api-key",
    "generic-api-key", "generic-api-key",
}

RULE_GROUP_MAP = {}

AUTH_RULES = {
    "jwt", "PaypalOauth", "GitHubOauth2", "github-app-token",
    "linkedin-client-id", "linkedin-client-secret", "facebook-secret",
    "microsoft-teams-webhook", "slack-webhook",
    "grafana-service-account-token", "openshift-user-token",
    "gitlab-incoming-mail-token", "gitlab-runner-authentication-token",
    "NpmToken",
}

INFRA_RULES = {
    "Postgres", "MongoDB", "SQLServer", "Redis", "RabbitMQ", "JDBC",
    "LDAP", "FTP", "kubernetes-secret-yaml",
    "GCPApplicationDefaultCredentials", "ocid", "private-key",
}

# Assign groups
for r in AUTH_RULES:
    RULE_GROUP_MAP[r] = "auth_token"

for r in INFRA_RULES:
    RULE_GROUP_MAP[r] = "infra_db"

# vendor_api (everything else)
for r in ALL_RULES:
    if r not in RULE_GROUP_MAP:
        RULE_GROUP_MAP[r] = "vendor_api"

# Explicit vendor cap
VENDOR_MAX_RATIO = 0.20   # ≤20% of synthetic data


# UTILITIES
def rand(n):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))

def fake_secret(rule):
    rule = rule.lower()

    # JWT
    if "jwt" in rule:
        return "eyJ" + rand(random.randint(30, 60))

    # Private keys / certs
    if "private-key" in rule:
        return (
            "-----BEGIN PRIVATE KEY-----\n"
            + rand(random.randint(48, 96))
            + "\n-----END PRIVATE KEY-----"
        )

    # Infra / DB
    if rule in {
        "postgres", "mongodb", "sqlserver", "redis",
        "rabbitmq", "jdbc", "ldap", "ftp", "ocid"
    }:
        return rand(random.randint(12, 32))

    # Auth tokens
    if rule in {
        "paypaloauth", "githuboauth2", "github-app-token",
        "npmtoken", "openshift-user-token",
        "gitlab-runner-authentication-token"
    }:
        return rand(random.randint(24, 64))

    # Vendor APIs
    return rand(random.randint(16, 48))


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())

# CODE TEMPLATES

TEMPLATES = [
    'API_KEY = "{value}"',
    'token = "{value}"',
    'secret = "{value}"',
    'config["secret"] = "{value}"',
    'headers = {{"Authorization": "Bearer {value}"}}',
    'conn_string = "postgres://user:{value}@localhost/db"',
    'export SECRET="{value}"',
    '// TODO: remove hardcoded key {value}',
    'os.environ["API_TOKEN"] = "{value}"',
    'tmp_val = "{value}"'
    'client.connect(tmp_val)',
    'client.authenticate(token="{value}", version="v2")'
]

# Obfuscations

def encode_base64(s): return base64.b64encode(s.encode()).decode()
def encode_hex(s): return s.encode().hex()
def encode_url(s): return urllib.parse.quote(s)

def concat_split(s):
    m = len(s)//2
    return f'"{s[:m]}" + "{s[m:]}"'

def reverse_string(s):
    return s[::-1] + "[::-1]"

def interleave_noise(s):
    return ''.join(c + random.choice("_-") for c in s)

def indirection(s):
    return f'tmp = "{s}"\nkey = tmp'

def env_mock(s):
    return f'os.getenv("API_KEY", "{s}")'

def commented(s):
    return f"// TODO remove key {s}"

ENCODING = [encode_base64, encode_hex, encode_url]
STRUCTURAL = [concat_split, reverse_string, interleave_noise]
POSITIONAL = [indirection, env_mock, commented]


# GENERATION
rows = []
vendor_count = 0
vendor_limit = int(R * VENDOR_MAX_RATIO)

BUCKETS = [
    ("clean", 0.25),
    ("encoding", 0.25),
    ("structural", 0.25),
    ("positional", 0.25),
]

for bucket, frac in BUCKETS:
    for _ in range(int(R * frac)):
        rule = random.choice(list(ALL_RULES))
        group = RULE_GROUP_MAP[rule]

        # Vendor downsampling
        if group == "vendor_api" and vendor_count >= vendor_limit:
            rule = random.choice([
                r for r in ALL_RULES
                if RULE_GROUP_MAP[r] != "vendor_api"
            ])
            group = RULE_GROUP_MAP[rule]

        if group == "vendor_api":
            vendor_count += 1

        secret = fake_secret(rule)

        # Select base template
        template = random.choice(TEMPLATES)

        # Apply obfuscation to secret only
        if bucket == "clean":
            obf_secret = secret
            obf = "clean"

        elif bucket == "encoding":
            fn = random.choice(ENCODING)
            obf_secret = fn(secret)
            obf = fn.__name__

        elif bucket == "structural":
            fn = random.choice(STRUCTURAL)
            obf_secret = fn(secret)
            obf = fn.__name__

        else:  # positional
            fn = random.choice(POSITIONAL)
            obf_secret = fn(secret)
            obf = fn.__name__

        # Build final snippet
        snippet = template.format(value=obf_secret)

        # Compute span
        if obf == "clean":
            secret_span_start = snippet.index(secret)
            secret_span_end = secret_span_start + len(secret)
        else:
            secret_span_start = -1
            secret_span_end = -1

        rows.append({
            "source": 'synthetic',
            "code_snippet": snippet,
            "secret": secret,
            "secret_span_start": secret_span_start,
            "secret_span_end": secret_span_end,
            "file_path": "-1",
            "line_number": -1,
            "length": len(secret),
            "entropy": round(shannon_entropy(secret), 4),
            "rule": rule,
            "has_secret": 1,
            "rule_group": group,
            "obfuscation_type": obf
        })

random.shuffle(rows)
rows = rows[:R]

# SAVE
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} synthetic positives → {OUTPUT_CSV}")

print("\nFinal distribution:")
print(Counter(r["rule_group"] for r in rows))