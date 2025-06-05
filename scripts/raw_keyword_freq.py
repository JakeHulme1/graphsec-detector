import json
import os
import re
from collections import Counter
import chardet

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, '..', 'datasets', 'processed', 'clean_commits.jsonl')
OUTPUT_FILE = os.path.join(BASE_DIR, '..', 'datasets', 'processed', 'raw_keyword_freq.json')

# Detect encoding
def detect_encoding(file_path, sample_size=100000):
    with open(file_path, 'rb') as f:
        raw_data = f.read(sample_size)
    return chardet.detect(raw_data)['encoding'] or 'utf-8'

def main():
    encoding = detect_encoding(INPUT_FILE)
    counter = Counter()

    with open(INPUT_FILE, 'r', encoding=encoding) as f:
        for line in f:
            try:
                record = json.loads(line)
                raw = record.get('keyword', '').strip().lower()
                if raw:
                    counter[raw] += 1
            except json.JSONDecodeError:
                continue

    # Save full raw keyword frequencies to JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        json.dump(counter.most_common(), out, indent=2)

    print(f"✅ Saved {len(counter)} raw keyword frequencies to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
