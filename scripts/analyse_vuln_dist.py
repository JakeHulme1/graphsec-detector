import json
import os
from collections import Counter
import chardet

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, '..', 'datasets', 'processed', 'clean_commits.jsonl')

def detect_encoding(file_path, sample_size=100000):
    with open(file_path, 'rb') as f:
        raw_data = f.read(sample_size)
    return chardet.detect(raw_data)['encoding'] or 'utf-8'

# Normalise keywords
def normalise(text):
    return text.strip().lower()

def main():
    encoding = detect_encoding(INPUT_FILE)
    keyword_counter = Counter()

    with open(INPUT_FILE, 'r', encoding=encoding) as f:
        for line in f:
            try:
                record = json.loads(line)
                keyword = normalise(record.get('keyword', ''))
                if keyword:
                    keyword_counter[keyword] += 1
            except json.JSONDecoder:
                continue

    print("Top 30 keywords:")
    for keyword, count in keyword_counter.most_common(30):
        print(f"{keyword:40} {count}")

if __name__ == '__main__':
    main()