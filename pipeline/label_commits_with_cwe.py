import json
import os
import re
import chardet

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "..", 'datasets', 'processed', 'clean_commits.jsonl')
OUTPUT_FILE = os.path.join(BASE_DIR, "..", 'datasets', 'processed', 'labelled_commits.jsonl')
ALIAS_FILE = os.path.join(BASE_DIR, "keyword_aliases.json")
CWE_MAPPING_FILE = os.path.join(BASE_DIR, "cwe_mapping.json")

# Detect encoding
def detect_encoding(file_path, sample_size=100000):
    with open(file_path, 'rb') as f:
        raw_data = f.read(sample_size)
    return chardet.detect(raw_data)['encoding'] or 'utf-8'

def normalise_keyword(keyword: str) -> str:
    keyword = keyword.lower().strip() # Convert to lowercase, trim whitespace
    keyword = re.sub(r'[^a-z0-9\s._-]', '', keyword)  # remove special characters except dots/underscores
    keyword = re.sub(
         r'\b(fix|check|update|improve|issue|correct|change|prevent|protect|resolve|patch|handle|secure|insecure|vulnerability|vulnerable|malicious|attack|exploit|unsafe|exposure)\b',
        '',
        keyword
    ) # Remove non-semantic action words (keywords are like: 'pickle improve' in VUDENC)
    keyword = re.sub(r'\s+', ' ', keyword).strip()  # collapse multiple spaces and trim
    return keyword

with open(ALIAS_FILE, "r") as f:
    keyword_aliases = json.load(f)

def resolve_alias(keyword: str) -> str:
    return keyword_aliases.get(keyword, keyword)

def map_to_cwe(keyword, cwe_map):
    # Return the id for the keyword
    return cwe_map.get(keyword, {}).get("id", "CWE-Other")

def main():
    encoding = detect_encoding(INPUT_FILE)

    # Open cwe map
    with open(CWE_MAPPING_FILE, "r") as f:
        cwe_map = json.load(f)

    # Open input file
    with open(INPUT_FILE, "r", encoding=encoding) as fin, open(OUTPUT_FILE, "w", encoding='utf-8') as fout:
        for i, line in enumerate(fin, 1):
            try:
                record = json.loads(line)
                keyword = record.get("keyword", "").strip().lower()
                if keyword:
                    norm = normalise_keyword(keyword)
                    norm = resolve_alias(norm)
                    record["cwe_id"] = map_to_cwe(norm, cwe_map)
                else:
                    record["cwe_id"] = "CWE-Other"
                fout.write(json.dumps(record) + "\n")

                if i % 10000 == 0:
                    print(f"{i} lines processed...")

            except json.JSONDecodeError:
                continue

if __name__ == '__main__':
    main()
