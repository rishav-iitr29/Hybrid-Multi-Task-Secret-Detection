import argparse
import os
import sys
import json
import torch
import re
from tqdm import tqdm
from transformers import RobertaTokenizerFast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from hybrid_model_architecture.hybrid_model import HybridMultiTaskModel
except ImportError:
    print("Error: Could not import HybridMultiTaskModel. Check your python path.")
    sys.exit(1)

MAX_LEN = 512


SECRET_REGEX = re.compile(
    r"""
    (?:
        ["']([A-Za-z0-9_\/+=\-]{16,})["'] |  # Group 1: Quoted
        =\s*([A-Za-z0-9_\/+=\-]{16,}) |      # Group 2: Assignment
        Bearer\s+([A-Za-z0-9_\/+=\-]{16,})   # Group 3: Bearer token
    )
    """,
    re.VERBOSE
)


# UTILS

def chunk_text_with_overlap(text, chunk_size=2000, overlap=400):
    if not text:
        return
    if len(text) <= chunk_size:
        yield text, 0
        return
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        yield text[i:i + chunk_size], i
        if i + chunk_size >= len(text):
            break


# LOADING MODEL

def load_model(model_path, device):
    model = HybridMultiTaskModel()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")
    return model, tokenizer


def run_model(model, tokenizer, snippet, device, top_k=5):
    enc = tokenizer(
        snippet,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
        return_offsets_mapping=True
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(enc["input_ids"], enc["attention_mask"])

        probs = torch.softmax(out["classification_logits"], dim=1)
        chunk_conf = probs[0, 1].item()
        entropy_pred = out["entropy_pred"].item()

        # Use Top-K to find multiple potential secrets in one chunk
        start_logits = out["span_start_logits"][0]
        end_logits = out["span_end_logits"][0]

        start_probs, start_indices = torch.topk(start_logits, top_k)
        end_probs, end_indices = torch.topk(end_logits, top_k)

        offsets = enc["offset_mapping"][0]

        candidates = []
        for i in range(top_k):
            for j in range(top_k):
                s_idx = start_indices[i].item()
                e_idx = end_indices[j].item()

                if e_idx < s_idx: continue
                if e_idx - s_idx > 150: continue  # Safety limit

                score = start_probs[i].item() + end_probs[j].item()
                candidates.append((score, s_idx, e_idx))

        candidates.sort(key=lambda x: x[0], reverse=True)

        final_spans = []
        for score, s, e in candidates:
            # NMS: Check overlap
            is_overlap = False
            for exist_s, exist_e in final_spans:
                if not (e < exist_s or s > exist_e):
                    is_overlap = True
                    break

            if not is_overlap:
                s_offset = offsets[s]
                e_offset = offsets[e]
                if s_offset[0] == 0 and s_offset[1] == 0: continue

                rel_start = s_offset[0].item()
                rel_end = e_offset[1].item()
                final_spans.append((rel_start, rel_end))

    return chunk_conf, final_spans, entropy_pred


def scan_file(path, model, tokenizer, device, threshold):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception:
        return None

    findings = []
    seen_starts = set()

    for chunk, chunk_offset in chunk_text_with_overlap(text):
        chunk_conf, spans, ent = run_model(model, tokenizer, chunk, device)

        if chunk_conf < threshold:
            continue

        for rel_start, rel_end in spans:
            abs_start = chunk_offset + rel_start
            abs_end = chunk_offset + rel_end

            raw_snippet = text[abs_start:abs_end].strip()

            if not raw_snippet or len(raw_snippet) < 3:
                continue

            m = SECRET_REGEX.search(raw_snippet)

            final_snippet = raw_snippet
            final_abs_start = abs_start

            if m:
                for i, g in enumerate(m.groups()):
                    if g:
                        clean_secret = g
                        match_start_index = m.start(i + 1)

                        final_snippet = clean_secret
                        final_abs_start = abs_start + match_start_index
                        break

            is_seen = False
            for seen_s in seen_starts:
                if abs(seen_s - final_abs_start) < 3:
                    is_seen = True
                    break
            if is_seen:
                continue
            seen_starts.add(final_abs_start)

            line_number = text.count("\n", 0, final_abs_start) + 1
            line_start_idx = text.rfind("\n", 0, final_abs_start)
            col_start = final_abs_start - (line_start_idx + 1 if line_start_idx != -1 else 0)

            findings.append({
                "confidence": round(chunk_conf, 3),
                "line": line_number,
                "column_span": [col_start, col_start + len(final_snippet)],
                "entropy": round(ent, 3),
                "snippet": final_snippet
            })

    findings.sort(key=lambda x: x['line'])
    return findings


# CLI

def scan_path(path, model, tokenizer, device, threshold):
    results = []
    files_to_scan = []

    if os.path.isfile(path):
        files_to_scan.append(path)
    else:
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith((".py", ".js", ".ts", ".env", ".yaml", ".json", ".txt")):
                    files_to_scan.append(os.path.join(root, f))

    for f_path in tqdm(files_to_scan, desc="Scanning files", unit="file"):
        findings = scan_file(f_path, model, tokenizer, device, threshold)
        if findings:
            results.append({"file": f_path, "findings": findings})

    return results


def main():
    parser = argparse.ArgumentParser(description="Hybrid Secret Scanner")
    parser.add_argument("scan", help="scan command", nargs="?")
    parser.add_argument("path", help="Path to file or directory")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--output", help="Save output to file")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[*] Loading model on {device}...")
    try:
        model, tokenizer = load_model(args.model, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"[*] Starting scan at: {args.path}")
    results = scan_path(args.path, model, tokenizer, device, args.threshold)

    if args.format == "json":
        out = json.dumps(results, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(out)
        else:
            print(out)
    else:
        if not results:
            print("\n[+] No secrets found.")
        else:
            for r in results:
                print(f"\n[!] {r['file']}")
                for i, f in enumerate(r["findings"], 1):
                    print(f"  ├─ Secret #{i} (Conf: {f['confidence']}, Ent: {f['entropy']})")
                    print(f"  └─ Line {f['line']}, Col {f['column_span'][0]}: {f['snippet']}")


if __name__ == "__main__":
    main()