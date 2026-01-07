import argparse
import csv
import random
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict
import os


# Utility: scanner-based check
def contains_secret(snippet: str) -> bool:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(snippet)
        tmp = f.name

    try:
        result = subprocess.run(
            ["gitleaks", "detect", "--source", tmp, "--no-git"],
            capture_output=True
        )
        return result.returncode == 1
    finally:
        os.remove(tmp)


# Negative Sampler
class NegativeCollector:

    CODE_EXTS = {
        ".py", ".js", ".ts", ".java", ".go", ".cpp",
        ".c", ".rs", ".php", ".rb"
    }

    SKIP_DIRS = {
        ".git", "node_modules", "venv", "env",
        "__pycache__", "dist", "build"
    }

    def __init__(self, positives_csv: str, repos_file: str):
        self.positives = self._load_csv(positives_csv)
        self.repos = self._load_repos(repos_file)
        self.temp_root = Path(tempfile.mkdtemp())

    # IO
    def _load_csv(self, path: str) -> List[Dict]:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _load_repos(self, path: str) -> List[str]:
        with open(path) as f:
            return [l.strip() for l in f if l.strip() and not l.startswith("#")]

    # Repo
    def clone_repo(self, repo_url: str) -> Path | None:
        repo_name = repo_url.rstrip("/").split("/")[-1]
        path = self.temp_root / repo_name

        if path.exists():
            shutil.rmtree(path)

        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(path)],
                check=True, timeout=300
            )
            return path
        except Exception:
            return None

    # Contextual Negatives
    def contextual_negatives(
        self,
        sample: Dict,
        repo_root: Path,
        window: int = 10,
        max_per_pos: int = 2
    ) -> List[Dict]:

        out = []

        try:
            line_no = int(sample["line_number"])
            file_path = repo_root / sample["file_path"]
        except Exception:
            return out

        if not file_path.exists():
            return out

        with open(file_path, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        start = max(0, line_no - window - 1)
        end = min(len(lines), line_no + window)

        candidates = [
            i for i in range(start, end)
            if i + 1 != line_no
        ]

        random.shuffle(candidates)

        for idx in candidates:
            if len(out) >= max_per_pos:
                break

            snippet = "".join(lines[idx:idx + 3]).strip()
            if not snippet:
                continue

            if contains_secret(snippet):
                continue

            out.append({
                "code_snippet": snippet,
                "label": 0,
                "secret": "",
                "secret_span_start": -1,
                "secret_span_end": -1,
                "file_path": sample["file_path"],
                "line_number": idx + 1,
                "entropy": 0.0,
                "length": 0,
                "source": "contextual_negative"
            })

        return out

    # Random Negatives
    def random_negatives(
        self,
        repo_root: Path,
        target: int
    ) -> List[Dict]:

        files = []
        for ext in self.CODE_EXTS:
            files.extend(repo_root.rglob(f"*{ext}"))

        files = [
            f for f in files
            if not any(d in f.parts for d in self.SKIP_DIRS)
        ]

        random.shuffle(files)
        out = []

        for fpath in files:
            if len(out) >= target:
                break

            try:
                with open(fpath, encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()

                if len(lines) < 5:
                    continue

                i = random.randint(0, len(lines) - 5)
                snippet = "".join(lines[i:i + 5]).strip()

                if not snippet:
                    continue

                if contains_secret(snippet):
                    continue

                out.append({
                    "code_snippet": snippet,
                    "label": 0,
                    "secret": "",
                    "secret_span_start": -1,
                    "secret_span_end": -1,
                    "file_path": str(fpath.relative_to(repo_root)),
                    "line_number": i + 1,
                    "entropy": 0.0,
                    "length": 0,
                    "source": "random_negative"
                })

            except Exception:
                continue

        return out


    def run(self, output_csv: str):
        negatives = []

        total_needed = len(self.positives)
        target_contextual = int(0.7 * total_needed)
        target_random = total_needed - target_contextual

        print(f"Target negatives: {total_needed}")
        print(f"  Contextual: {target_contextual}")
        print(f"  Random: {target_random}")


        # Clone repositories
        cloned_repos: Dict[str, Path] = {}
        counter = 1
        for repo_url in self.repos:
            repo_path = self.clone_repo(repo_url)
            if repo_path:
                cloned_repos[repo_url] = repo_path
            counter+=1
            print(f"{counter}/1050")

        if not cloned_repos:
            print("No repositories cloned successfully")
            return

        contextual_count = 0

        # Contextual negatives
        for repo_url, repo_path in cloned_repos.items():
            for p in self.positives:
                if contextual_count >= target_contextual:
                    break

                found = self.contextual_negatives(p, repo_path)
                contextual_count += len(found)
                negatives.extend(found)

            if contextual_count >= target_contextual:
                break

        # Random negatives
        per_repo = max(1, target_random // len(cloned_repos))

        for repo_url, repo_path in cloned_repos.items():
            negatives.extend(self.random_negatives(repo_path, per_repo))


        # Trim, shuffle, save
        negatives = negatives[:total_needed]
        random.shuffle(negatives)

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=negatives[0].keys())
            writer.writeheader()
            writer.writerows(negatives)

        print(f"Saved {len(negatives)} negatives to {output_csv}")


        # Cleanup
        for repo_path in cloned_repos.values():
            shutil.rmtree(repo_path, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positives", required=True, help="Positive CSV file")
    ap.add_argument("--repos", required=True, help="repos.txt")
    ap.add_argument("--output", required=True, help="Output negatives CSV")
    args = ap.parse_args()

    NegativeCollector(args.positives, args.repos).run(args.output)


if __name__ == "__main__":
    main()