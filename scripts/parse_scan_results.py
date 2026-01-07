"""
parse_scan_results.py
====================
Parse existing GitLeaks and TruffleHog JSON outputs and save to CSV

This script:
1. Finds all JSON files in a directory (from GitLeaks/TruffleHog)
2. Parses each JSON file
3. Extracts positive examples with metadata
4. Saves to CSV file

Usage:
    python parse_scan_results.py --input scan_results/ --output dataset.csv --repo-path /path/to/repo

If you already ran the scans manually:
    gitleaks detect --source /path/to/repo --report-format json --report-path results/gitleaks.json
    trufflehog filesystem /path/to/repo --json > results/trufflehog.json
    python parse_scan_results.py --input results/ --output dataset.csv --repo-path /path/to/repo
"""

import json
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict
import math
import os


class DatasetParser:
    """Parse GitLeaks and TruffleHog JSON outputs into dataset format"""

    @staticmethod
    def calculate_entropy(text: str) -> float:
        """Calculate Shannon entropy of a string"""
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
            entropy -= prob * math.log2(prob)  # Simplified entropy

        return entropy# Scale to ~0-5 range

    @staticmethod
    def extract_code_snippet(file_path: str, line_number: int, context: int = 3) -> str:
        """
        Extract code snippet with context

        Args:
            file_path: Path to source file
            line_number: Line number of secret
            context: Number of lines before/after to include

        Returns:
            Code snippet as string
        """
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

    def parse_gitleaks(self, json_file: Path, repo_path: Path = None) -> List[Dict]:
        """
        Parse GitLeaks JSON output

        Args:
            json_file: Path to GitLeaks JSON output
            repo_path: Path to repository (for extracting code, optional)

        Returns:
            List of parsed secret dictionaries
        """
        secrets = []

        try:
            print(f"  Parsing GitLeaks: {json_file.name}")

            with open(json_file, 'r') as f:
                data = json.load(f)

            # GitLeaks format: list of findings
            if not isinstance(data, list):
                data = [data]

            for finding in data:
                # Extract secret information
                secret_value = finding.get('Secret', finding.get('Match', ''))
                file_path = finding.get('File', '')
                line_number = finding.get('StartLine', 0)

                # Extract code snippet if repo path provided
                code_snippet = ""
                if repo_path and file_path:
                    full_file_path = repo_path / file_path
                    code_snippet = self.extract_code_snippet(str(full_file_path), line_number)

                # If no snippet, use the match
                if not code_snippet:
                    code_snippet = finding.get('Match', secret_value)

                # Find secret position in snippet
                secret_start = code_snippet.find(secret_value)
                secret_end = secret_start + len(secret_value) if secret_start != -1 else 0

                # Calculate entropy
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
                    'has_secret': 1
                })

            print(f"    → Found {len(secrets)} secrets")

        except Exception as e:
            print(f"    ✗ Error parsing GitLeaks output: {e}")

        return secrets

    def parse_trufflehog(self, json_file: Path, repo_path: Path = None) -> List[Dict]:
        """
        Parse TruffleHog JSON output

        Args:
            json_file: Path to TruffleHog JSON output
            repo_path: Path to repository (optional)

        Returns:
            List of parsed secret dictionaries
        """
        secrets = []

        try:
            print(f"  Parsing TruffleHog: {json_file.name}")

            with open(json_file, 'r') as f:
                # TruffleHog outputs one JSON object per line
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        finding = json.loads(line)

                        # TruffleHog v3 format
                        source_metadata = finding.get('SourceMetadata', {})
                        data = source_metadata.get('Data', {})

                        # Get file path
                        filesystem_data = data.get('Filesystem', {})
                        file_path = filesystem_data.get('file', '')
                        line_number = filesystem_data.get('line', 0)

                        # Get the secret (try multiple fields)
                        raw_data = finding.get('Raw', '')
                        if not raw_data:
                            raw_data = finding.get('RawV2', '')
                        if not raw_data:
                            raw_data = finding.get('Redacted', '')

                        if not raw_data:
                            continue

                        # Extract code snippet if repo path provided
                        code_snippet = ""
                        if repo_path and file_path:
                            full_file_path = repo_path / file_path
                            code_snippet = self.extract_code_snippet(str(full_file_path), line_number)

                        # If no snippet, use just the secret
                        if not code_snippet:
                            code_snippet = raw_data

                        # Find secret in snippet
                        secret_start = code_snippet.find(raw_data)
                        if secret_start == -1:
                            secret_start = 0
                            secret_end = len(raw_data)
                        else:
                            secret_end = secret_start + len(raw_data)

                        # Calculate entropy
                        entropy = self.calculate_entropy(raw_data)

                        # Get detector name
                        detector_name = finding.get('DetectorName', 'unknown')

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
                            'rule': detector_name,
                            'has_secret': 1
                        })

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"    Warning: Error parsing line {line_num}: {e}")
                        continue

            print(f"    → Found {len(secrets)} secrets")

        except Exception as e:
            print(f"    ✗ Error parsing TruffleHog output: {e}")

        return secrets


class JSONResultsParser:
    """Parse all JSON files in a directory and save to CSV"""

    def __init__(self, output_csv: str):
        self.parser = DatasetParser()
        self.output_csv = output_csv
        self.all_secrets = []

    def find_json_files(self, input_dir: Path) -> Dict[str, List[Path]]:
        """
        Find all GitLeaks and TruffleHog JSON files

        Args:
            input_dir: Directory containing JSON files

        Returns:
            Dict with 'gitleaks' and 'trufflehog' lists of file paths
        """
        gitleaks_files = []
        trufflehog_files = []

        # Find files by naming convention
        for json_file in input_dir.glob("*.json"):
            name_lower = json_file.name.lower()

            if 'gitleaks' in name_lower:
                gitleaks_files.append(json_file)
            elif 'trufflehog' in name_lower or 'truffle' in name_lower:
                trufflehog_files.append(json_file)
            else:
                # Try to guess by content
                try:
                    with open(json_file, 'r') as f:
                        first_line = f.readline()

                    # Check if it looks like GitLeaks (array format)
                    if first_line.strip().startswith('['):
                        gitleaks_files.append(json_file)
                    # Check if it looks like TruffleHog (line-delimited JSON)
                    elif 'DetectorName' in first_line or 'SourceMetadata' in first_line:
                        trufflehog_files.append(json_file)
                except:
                    pass

        return {
            'gitleaks': gitleaks_files,
            'trufflehog': trufflehog_files
        }

    def parse_directory(self, input_dir: Path, repo_path: Path = None):
        """
        Parse all JSON files in directory

        Args:
            input_dir: Directory containing JSON files
            repo_path: Optional path to repository for code extraction
        """
        print("=" * 80)
        print(f"PARSING JSON FILES FROM: {input_dir}")
        print("=" * 80)

        # Find JSON files
        json_files = self.find_json_files(input_dir)

        print(f"\nFound:")
        print(f"  GitLeaks files: {len(json_files['gitleaks'])}")
        print(f"  TruffleHog files: {len(json_files['trufflehog'])}")

        if not json_files['gitleaks'] and not json_files['trufflehog']:
            print("\n✗ No JSON files found!")
            print("  Expected files with 'gitleaks' or 'trufflehog' in the name")
            return

        # Parse GitLeaks files
        if json_files['gitleaks']:
            print("\n" + "-" * 80)
            print("PARSING GITLEAKS FILES")
            print("-" * 80)
            for json_file in json_files['gitleaks']:
                secrets = self.parser.parse_gitleaks(json_file, repo_path)
                self.all_secrets.extend(secrets)

        # Parse TruffleHog files
        if json_files['trufflehog']:
            print("\n" + "-" * 80)
            print("PARSING TRUFFLEHOG FILES")
            print("-" * 80)
            for json_file in json_files['trufflehog']:
                secrets = self.parser.parse_trufflehog(json_file, repo_path)
                self.all_secrets.extend(secrets)

    def save_to_csv(self):
        """Save collected secrets to CSV"""
        if not self.all_secrets:
            print("\n✗ No secrets found in JSON files!")
            return

        # Remove duplicates based on code_snippet and secret
        seen = set()
        unique_secrets = []

        for secret in self.all_secrets:
            key = (secret['code_snippet'], secret['secret'])
            if key not in seen:
                seen.add(key)
                unique_secrets.append(secret)

        print("\n" + "=" * 80)
        print("DATASET SUMMARY")
        print("=" * 80)
        print(f"Total secrets found: {len(self.all_secrets)}")
        print(f"Unique secrets: {len(unique_secrets)}")
        print(f"From GitLeaks: {sum(1 for s in unique_secrets if s['source'] == 'gitleaks')}")
        print(f"From TruffleHog: {sum(1 for s in unique_secrets if s['source'] == 'trufflehog')}")

        # Save to CSV
        df = pd.DataFrame(unique_secrets)
        df.to_csv(self.output_csv, index=False)

        print(f"\n✓ Dataset saved to {self.output_csv}")

        # Show sample
        print(f"\nSample entries:")
        print(df.head(3).to_string())

        # Statistics
        print(f"\nStatistics:")
        print(f"  Average secret length: {df['length'].mean():.2f}")
        print(f"  Average entropy: {df['entropy'].mean():.2f}")
        print(f"  Files with secrets: {df['file_path'].nunique()}")
        print(f"  Unique rules triggered: {df['rule'].nunique()}")

    def run(self, input_dir: str, repo_path: str = None):
        """
        Run the complete parsing pipeline

        Args:
            input_dir: Directory containing JSON files
            repo_path: Optional path to repository for code extraction
        """
        print("=" * 80)
        print("API KEY DETECTION - JSON RESULTS PARSER")
        print("=" * 80)

        # Validate input directory
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"✗ Error: Input directory not found: {input_dir}")
            return

        # Validate repo path if provided
        repo_path_obj = None
        if repo_path:
            repo_path_obj = Path(repo_path)
            if not repo_path_obj.exists():
                print(f"⚠ Warning: Repository path not found: {repo_path}")
                print("  Will parse without code snippet extraction")
                repo_path_obj = None

        # Parse all JSON files
        self.parse_directory(input_path, repo_path_obj)

        # Save results
        self.save_to_csv()

        print("\n" + "=" * 80)
        print("PARSING COMPLETE!")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Parse GitLeaks and TruffleHog JSON outputs to CSV dataset"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Directory containing JSON files from GitLeaks/TruffleHog'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='dataset.csv',
        help='Output CSV file path (default: dataset.csv)'
    )
    parser.add_argument(
        '--repo-path',
        type=str,
        default=None,
        help='Optional: Path to repository for code snippet extraction'
    )

    args = parser.parse_args()

    # Run parser
    json_parser = JSONResultsParser(output_csv=args.output)
    json_parser.run(args.input, args.repo_path)


if __name__ == "__main__":
    main()