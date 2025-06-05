import json
import os
import re
from collections import Counter
import chardet

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, '..', 'datasets', 'processed', 'clean_commits.jsonl')
RAW_OUTPUT_FILE = os.path.join(BASE_DIR, '..', 'datasets', 'processed', 'raw_keyword_freq.json')
NORMALIZED_OUTPUT_FILE = os.path.join(BASE_DIR, '..', 'datasets', 'processed', 'normalised_keyword_freq.json')


# Detect encoding
def detect_encoding(file_path, sample_size=100000):
    with open(file_path, 'rb') as f:
        raw_data = f.read(sample_size)
    return chardet.detect(raw_data)['encoding'] or 'utf-8'

# Normalise a keyword string for CWE mapping purposes
# This function lowercases text, removes punctuation, and strips noise words such as 'fix', 'update' etc., that don't define the vulnerability type
def normalise_keyword(keyword: str) -> str:
    keyword = keyword.lower().strip() # Convert to lowercase, trim whitespace
    keyword = re.sub(r'[^a-z0-9\s._-]', '', keyword)  # remove special characters except dots/underscores
    keyword = re.sub(
         r'\b(fix|check|update|improve|issue|correct|change|prevent|protect|resolve|patch|handle|secure)\b',
        '',
        keyword
    ) # Remove non-semantic action words (keywords are like: 'pickle improve' in VUDENC)
    keyword = re.sub(r'\s+', ' ', keyword).strip()  # collapse multiple spaces and trim
    return keyword

def main():
    encoding = detect_encoding(INPUT_FILE)
    raw_counter = Counter()
    normalized_counter = Counter()

    with open(INPUT_FILE, 'r', encoding=encoding) as f:
        for line in f:
            try:
                record = json.loads(line)
                raw = record.get('keyword', '').strip().lower()
                if raw:
                    raw_counter[raw] += 1
                    norm = normalise_keyword(raw)
                    if norm:
                        normalized_counter[norm] += 1
            except json.JSONDecodeError:
                continue

    # Save raw keyword frequencies
    with open(RAW_OUTPUT_FILE, 'w', encoding='utf-8') as out:
        json.dump(raw_counter.most_common(), out, indent=2)

    # Save normalized keyword frequencies
    with open(NORMALIZED_OUTPUT_FILE, 'w', encoding='utf-8') as out:
        json.dump(normalized_counter.most_common(), out, indent=2)

    print(f"Saved {len(raw_counter)} raw keyword frequencies to {RAW_OUTPUT_FILE}")
    print(f"Saved {len(normalized_counter)} normalised keyword frequencies to {NORMALIZED_OUTPUT_FILE}")

if __name__ == '__main__':
    main()
