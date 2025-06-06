import json 
import os
from collections import Counter, defaultdict

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NORMALISED_KEYWORDS_FILE = os.path.join(BASE_DIR, "..", "datasets", "processed", "normalised_keyword_freq.json")
CWE_MAPPING_FILE = os.path.join(BASE_DIR, "cwe_mapping.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "datasets", "processed", "cwe_freq.json")

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def map_to_cwe(keyword, cwe_map):
    # Return value of the keyword, else return the placeholder 
    return cwe_map.get(keyword, {"id": "CWE-Other", "name": "Unmpapped/Unknown Vulnerability"})

def main():
    # Load resources
    keyword_freq = dict(load_json(NORMALISED_KEYWORDS_FILE)) # Convert to dict to easily access name and count
    cwe_map = load_json(CWE_MAPPING_FILE)

    cwe_freq = defaultdict(int) # Defaults value to 0 for new keys (don't need to check if cwe_id exists)
    cwe_label_map = {} # Stores CWE name for each CWE ID

    for keyword, count in keyword_freq.items():
        cwe_entry = map_to_cwe(keyword, cwe_map)
        cwe_id =  cwe_entry["id"]
        cwe_freq[cwe_id] += count
        cwe_label_map[cwe_id] = cwe_entry["name"]

    # Save output in descending order of frequency
    result = [{"cwe_id": cwe_id, 
               "cwe_name": cwe_label_map[cwe_id], 
               "count": count} 
               for cwe_id, count in sorted(cwe_freq.items(), key=lambda x: -x[1])]
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"CWE Frequencies written to {OUTPUT_FILE}")

    unmapped = [
    keyword for keyword in keyword_freq
    if map_to_cwe(keyword, cwe_map)["id"] == "CWE-Other"
    ]

    print(f"\n[!] Unmapped keywords: {len(unmapped)}")
    for kw in sorted(unmapped):
        print("-", kw)

if __name__ == "__main__":
    main()