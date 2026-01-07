import math
import subprocess
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
import tempfile


class SecretScanner:
    def __init__(self, output_dir: str = "scan_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp())

        # Check if tools are installed
        self._check_tools()

    def _check_tools(self):
        try:
            subprocess.run(["gitleaks", "version"],
                           capture_output=True, check=True)
            print("GitLeaks found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("GitLeaks not found")

        try:
            subprocess.run(["trufflehog", "--version"],
                           capture_output=True, check=True)
            print("TruffleHog found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("TruffleHog not found")

    def clone_repo(self, repo_url: str) -> Optional[Path]:
        try:
            # Extract repo name from URL
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            repo_path = self.temp_dir / repo_name

            # Remove if exists
            if repo_path.exists():
                shutil.rmtree(repo_path)

            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
                capture_output=True,
                check=True,
                timeout=300  # 5 minute timeout
            )

            return repo_path

        except subprocess.CalledProcessError as e:
            print(f"  Failed to clone: {e}")
            return None
        except Exception as e:
            print(f"  Error: {e}")
            return None

    def run_gitleaks(self, repo_path: Path) -> Optional[Path]:
        try:
            output_file = self.output_dir / f"gitleaks_{repo_path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            print(f"  Running GitLeaks...")

            # Run gitleaks detect
            result = subprocess.run(
                [
                    "gitleaks",
                    "detect",
                    "--source", str(repo_path),
                    "--report-format", "json",
                    "--report-path", str(output_file)
                ],
                capture_output=True,
                timeout=600  # 10 minute timeout
            )

            # GitLeaks returns exit code 1 if leaks found, which is what we want!
            if output_file.exists():
                print(f"  GitLeaks results saved to {output_file}")
                return output_file
            else:
                print(f"  GitLeaks completed (no leaks found)")
                return None

        except subprocess.TimeoutExpired:
            print(f"  GitLeaks timed out")
            return None
        except Exception as e:
            print(f"  GitLeaks error: {e}")
            return None

    def run_trufflehog(self, repo_path: Path) -> Optional[Path]:
        try:
            output_file = self.output_dir / f"trufflehog_{repo_path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            print(f"  Running TruffleHog...")

            # Run trufflehog
            result = subprocess.run(
                [
                    "trufflehog",
                    "filesystem",
                    str(repo_path),
                    "--json"
                ],
                capture_output=True,
                timeout=600,  # 10 minute timeout
                text=True
            )

            # Save output
            if result.stdout:
                with open(output_file, 'w') as f:
                    f.write(result.stdout)
                print(f"  TruffleHog results saved to {output_file}")
                return output_file
            else:
                print(f"  TruffleHog completed (no secrets found)")
                return None

        except subprocess.TimeoutExpired:
            print(f"  TruffleHog timed out")
            return None
        except Exception as e:
            print(f"  TruffleHog error: {e}")
            return None

    def cleanup_repo(self, repo_path: Path):
        # Delete cloned repository
        try:
            if repo_path.exists():
                shutil.rmtree(repo_path)
                print(f"  Cleaned up {repo_path}")
        except Exception as e:
            print(f"  Cleanup failed: {e}")


class DatasetParser:
    # Parse GitLeaks and TruffleHog outputs into dataset format

    @staticmethod
    def calculate_entropy(text: str) -> float:
        if not text:
            return 0.0

        # Count character frequencies
        from collections import Counter
        freq = Counter(text)

        # Calculate entropy
        entropy = 0.0
        text_len = len(text)
        for count in freq.values():
            prob = count / text_len
            entropy -= prob * math.log2(prob)

        return entropy  # Scale to ~0-5 range

    @staticmethod
    def extract_code_snippet(file_path: str, line_number: int, context: int = 3) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            # Get line range
            start = max(0, line_number - context - 1)
            end = min(len(lines), line_number + context)

            snippet = ''.join(lines[start:end])
            return snippet.strip()

        except Exception as e:
            print(f"    Warning: Could not read {file_path}: {e}")
            return ""

    def parse_gitleaks(self, json_file: Path, repo_path: Path) -> List[Dict]:

        # Parse GitLeaks JSON output
        secrets = []

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            if not isinstance(data, list):
                data = [data]

            for finding in data:
                secret_value = finding.get('Secret', finding.get('Match', ''))
                file_path = finding.get('File', '')
                line_number = finding.get('StartLine', 0)

                full_file_path = repo_path / file_path

                code_snippet = self.extract_code_snippet(
                    str(full_file_path),
                    line_number
                )

                secret_start = code_snippet.find(secret_value)
                secret_end = secret_start + len(secret_value) if secret_start != -1 else 0
                if secret_start == -1:
                    secret_start = None
                    secret_end = None

                entropy = self.calculate_entropy(secret_value)

                secrets.append({
                    'source': 'gitleaks',
                    'code_snippet': code_snippet,
                    'secret': secret_value,
                    'secret_span_start': secret_start,
                    'secret_span_end': secret_end,
                    'file_path': file_path,
                    'line_number': line_number,
                    'length': len(secret_value),
                    'entropy': entropy,
                    'rule': finding.get('RuleID', 'unknown'),
                    'label': 1,
                    'obfuscation_type': 'raw'
                })

        except Exception as e:
            print(f"    Error parsing GitLeaks output: {e}")

        return secrets

    def parse_trufflehog(self, json_file: Path, repo_path: Path) -> List[Dict]:

        # Parse TruffleHog JSON output
        secrets = []

        try:
            with open(json_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        finding = json.loads(line)

                        source_metadata = finding.get('SourceMetadata', {})
                        data = source_metadata.get('Data', {})

                        file_path = data.get('Filesystem', {}).get('file', '')
                        line_number = data.get('Filesystem', {}).get('line', 0)

                        raw_data = finding.get('Raw', '')

                        if len(raw_data) > 200:
                            continue

                        full_file_path = repo_path / file_path

                        code_snippet = self.extract_code_snippet(
                            str(full_file_path),
                            line_number
                        )

                        secret_start = code_snippet.find(raw_data)
                        secret_end = secret_start + len(raw_data) if secret_start != -1 else 0

                        entropy = self.calculate_entropy(raw_data)

                        secrets.append({
                            'source': 'trufflehog',
                            'code_snippet': code_snippet,
                            'secret': raw_data,
                            'secret_span_start': secret_start,
                            'secret_span_end': secret_end,
                            'file_path': file_path,
                            'line_number': line_number,
                            'length': len(raw_data),
                            'entropy': entropy,
                            'rule': finding.get('DetectorName', 'unknown'),
                            'label': 1,
                            'obfuscation_type': 'raw'
                        })

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            print(f"    Error parsing TruffleHog output: {e}")

        return secrets


class DatasetCollector:
    def __init__(self, output_csv: str = "dataset.csv"):
        self.scanner = SecretScanner()
        self.parser = DatasetParser()
        self.output_csv = output_csv
        self.all_secrets = []

    def read_repos(self, repos_file: str) -> List[str]:
        # Read repository URLs from text file
        try:
            with open(repos_file, 'r') as f:
                repos = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            print(f"Loaded {len(repos)} repositories from {repos_file}")
            return repos
        except Exception as e:
            print(f"Error reading repos file: {e}")
            return []

    def process_repo(self, repo_url: str):
        print(f"Processing: {repo_url}")

        # Clone repository
        repo_path = self.scanner.clone_repo(repo_url)
        if not repo_path:
            return

        # Run GitLeaks
        gitleaks_output = self.scanner.run_gitleaks(repo_path)
        if gitleaks_output:
            gitleaks_secrets = self.parser.parse_gitleaks(gitleaks_output, repo_path)
            self.all_secrets.extend(gitleaks_secrets)
            print(f"  Found {len(gitleaks_secrets)} secrets with GitLeaks")

        # Run TruffleHog
        trufflehog_output = self.scanner.run_trufflehog(repo_path)
        if trufflehog_output:
            trufflehog_secrets = self.parser.parse_trufflehog(trufflehog_output, repo_path)
            self.all_secrets.extend(trufflehog_secrets)
            print(f"  Found {len(trufflehog_secrets)} secrets with TruffleHog")

        # Cleanup
        self.scanner.cleanup_repo(repo_path)

    def save_to_csv(self):
        if not self.all_secrets:
            print("\nNo secrets found!")
            return

        # Remove duplicates based on code_snippet and secret
        seen = set()
        unique_secrets = []

        for secret in self.all_secrets:
            key = (secret['code_snippet'], secret['secret'])
            if key not in seen:
                seen.add(key)
                unique_secrets.append(secret)

        print(f"\n")
        print(f"DATASET SUMMARY")
        print(f"Total secrets found: {len(self.all_secrets)}")
        print(f"Unique secrets: {len(unique_secrets)}")
        print(f"From GitLeaks: {sum(1 for s in unique_secrets if s['source'] == 'gitleaks')}")
        print(f"From TruffleHog: {sum(1 for s in unique_secrets if s['source'] == 'trufflehog')}")

        # Save to CSV
        df = pd.DataFrame(unique_secrets)
        df.to_csv(self.output_csv, index=False)

        print(f"\nDataset saved to {self.output_csv}")


        # Statistics
        print(f"\nStatistics:")
        print(f"  Average secret length: {df['length'].mean():.2f}")
        print(f"  Average entropy: {df['entropy'].mean():.2f}")
        print(f"  Files with secrets: {df['file_path'].nunique()}")

    def run(self, repos_file: str):
        print("API KEY DETECTION - DATASET COLLECTOR")

        # Read repositories
        repos = self.read_repos(repos_file)
        if not repos:
            return

        # Process each repository
        for i, repo_url in enumerate(repos, 1):
            print(f"\n[{i}/{len(repos)}] Processing repository...")
            try:
                self.process_repo(repo_url)
            except Exception as e:
                print(f"  Error processing {repo_url}: {e}")
                continue

        # Save results
        self.save_to_csv()

        print("COLLECTION COMPLETE!")


def main():
    parser = argparse.ArgumentParser(
        description="Collect API key leakage dataset from GitHub repositories"
    )
    parser.add_argument(
        '--repos',
        type=str,
        required=True,
        help='Path to text file containing GitHub repo URLs (one per line)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='dataset.csv', # in my repository, it's in data folder
        help='Output CSV file path (default: dataset.csv)'
    )

    args = parser.parse_args()

    # Run collector
    collector = DatasetCollector(output_csv=args.output)
    collector.run(args.repos)


if __name__ == "__main__":
    main()