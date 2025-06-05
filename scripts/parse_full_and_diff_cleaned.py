import ijson # Streaming JSON parser for large files
import json
import re # For detecting comment lines
import logging
import os
import chardet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths relative to script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, '..', 'datasets', 'raw', 'PyCommitsWithDiffs.jsonl')
OUTPUT_FILE = os.path.join(BASE_DIR, '..', 'datasets', 'processed', 'clean_commits.jsonl')

def detect_encoding(file_path, sample_size=100000):
    with open(file_path, 'rb') as f:
        raw_data = f.read(sample_size)
    result = chardet.detect(raw_data)
    return result['encoding'] or 'utf-8'

def is_comment_line(line):
    """
    Detects single-line comments in common formats
    """
    return bool(re.match(r'^\s*(#|//|/\*).+', line))

def strip_multiline_comments(lines):
    """
    Removes multi-line comment blocks like from a list of lines.

    Returns:
        A list of lines with all multi-line comment blocks removed.
    """
    stripped = []
    in_block = False
    block_delimiter = None

    for line in lines:
        # Start or end of multi-line block
        if not in_block and (line.strip().startswith('"""') or line.strip().startswith('"""')):
            in_block = True
            block_delimiter = line.strip()[:3] # Store block_delimiter
            # If its a single line docstring, skip it entirely
            if line.strip().count(block_delimiter) == 2:
                in_block = False
                continue
            continue

        if in_block:
            # End of block
            if block_delimiter and block_delimiter in line:
                in_block = False
            continue

        # Outside block, keep line
        stripped.append(line)

    return stripped

def parse_diff(diff_text):
    """
    Parses a unified diff string, removing both inline and block comments, and separates:
        - Full pre-fix code ("old_lines")
        - Full post-fix code ("new_lines")
        - Lines that were removed ("old_changed")
        - Lines that were added ("new_changed")

    Returns:
        old_lines (str), new_lines(str)
        old_changed (str), new_changed (str)
    """
    old_lines = []
    new_lines = []
    old_changed = []
    new_changed = []

    raw_old = []
    raw_new = []

    for line in diff_text.splitlines(): # Each line if diff in VUDENC is separated by '\n'
        line = line.strip('\n')
        if line.startswith(('diff ', 'index ', '---', '+++')): # Skip irrelevant data
            continue

        elif line.startswith('-') and not line.startswith('--'):
            content = line[1:].strip()
            if not is_comment_line(content):
                raw_old.append(content)
                old_changed.append(content)

        elif line.startswith('+') and not line.startswith('++'):
            content = line[1:].strip()
            if not is_comment_line(content):
                raw_new.append(content)
                new_changed.append(content)

        elif not line.startswith(('+', '-')):
            content = line.strip()
            if not is_comment_line(content):
                raw_old.append(content)
                raw_new.append(content)


    # Remove multiline comment blocks from collected raw lines
    old_lines = strip_multiline_comments(raw_old)
    new_lines = strip_multiline_comments(raw_new)

    return (
        '\n'.join(old_lines),
        '\n'.join(new_lines),
        '\n'.join(old_changed),
        '\n'.join(new_changed)
    )

def process_commits():
    """
    Streams and processes each commit from a large JSON file, extracting relevant fields,
    removing all types of comments from the code, and writing structured, cleaned records
    to an output JSONL file. This format is suitable for downstream ML processing.
    """

    logging.info("Starting commit processing...")
    count_total = 0
    count_written = 0
    count_skipped = 0

    # Detect the encoding of the input .jsonl
    encoding = detect_encoding(INPUT_FILE)
    logging.info(f"Detected encoding: {encoding}")

    with open(INPUT_FILE, 'r', encoding=encoding) as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            outer = json.loads(line)
            commit = list(outer.values())[0]  # Get the actual commit object
            count_total += 1

            # Extract metadata and diff string
            sha = commit.get('sha')
            repo = commit.get('repo')
            keyword = commit.get('keyword', '')
            diff = commit.get('diff', '')

            # Parse the diff to separate old/new full code and the changed lines
            old_code, new_code, old_diff, new_diff = parse_diff(diff)

            # Skip commits with no meaningful code after cleaning
            if not (old_diff.strip() or new_diff.strip()):
                logging.warning(f"Skipping commit {sha} — no code extracted. Keyword: {keyword}\nFirst 300 chars of diff:\n{diff[:300]}")
                count_skipped += 1
                continue


            # Structure the clean commit record for output
            record = {
                'sha': sha,
                'repo': repo,
                'keyword': keyword,
                'old_code_full': old_code,
                'new_code_full': new_code,
                'old_changed_lines': old_diff,
                'new_changed_lines': new_diff
            }

            # Write the structured JSON record to the output file
            f_out.write(json.dumps(record) + '\n')
            count_written += 1

            # Log progress every 100 records
            if count_total % 100 == 0:
                logging.info(f"Processed: {count_total} commits, Written: {count_written}, Skipped: {count_skipped}")

    # Final summary log
    logging.info(f"Finished processing. Total: {count_total}, Written: {count_written}, Skipped: {count_skipped}")

if __name__ == '__main__':
    process_commits()