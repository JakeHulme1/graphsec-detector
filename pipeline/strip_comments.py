# Script usage:
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
strip_comments.py

Removes single-line comments (`# ...`) and multi-line comments/docstrings from `source` and `new_code` fields in VUDENC samples, and writes a cleaned output file.

Author: Jake Hulme
Date: 09/06/2025
"""

import json
import re
from pathlib import Path
import logging
import io
import tokenize

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def strip_comments(code: str) -> str:
    """
    Removes all comments and docstrings from Python code using the `tokenize` module.

    Args:
        code (str): Raw Python code.

    Returns:
        str: Cleaned code with comments removed, or empyt if tokenization fails.
    """

    # Make sure code is always a string
    if code is None:
        return ""

    output = []
    prev_toktype = None # Track previous token type to detect blocks of doc-strings
    last_lineno = -1    # Line numbers start at 1
    last_col = 0        # Col numbers start at 0

    # Convert code string to a byte stream (to tokenize strings, must encode into bytes)
    bytes_io = io.BytesIO(code.encode('utf-8'))

    try:
        # Tokenize the code using byte stream's readline method
        tokens = tokenize.tokenize(bytes_io.readline)

        # Loop through all tokens in the source code
        for tok in tokens:
            toktype = tok.type      # Type of token
            tokval = tok.string     # The string of content
            srow, scol = tok.start  # Starting line and column of the token
            erow, ecol = tok.end    # Ending line and column of the token

            # Skip all comments (inline or full-line)
            if toktype == tokenize.COMMENT:
                continue

            # Skip block-level docstrings
            # If the current token is a STRING and the previous token was an INDENT or NEWLINE,
            # it is likely a docstring (module, class, or function-level)
            if toktype == tokenize.STRING and prev_toktype in {tokenize.INDENT, tokenize.NEWLINE}:
                continue

            # If token starts on a new line, reset column tracking
            if srow > last_lineno:
                last_col = 0

            # If there's a space between this token and the previous, preserve spacing
            if scol > last_col:
                output.append(" " * (scol - last_col)) # Output required number of spaces

            # Append the content token to the output
            output.append(tokval)

            # Update previous token type and cursor poitions
            prev_toktype = toktype
            last_lineno = erow
            last_col = ecol

    except tokenize.TokenError as e:
        # If code has tokenisation error, log and return empty
        logging.warning(f"Tokenization error: {e}")
        return ""
    
    # Join all tokens and return the stripped code
    return "".join(output).strip()

def clean_dataset(input_path: Path, output_path: Path) -> None:
    """
    Processes each entry into a JSONL dataset, remving comments from source and new_code.

    Args:
        input_path (Path): Path to input JSONL
        output_path (Path): Path to output JSONL
    """
    count = 0
    skipped = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            entry = json.loads(line)

            # Strip comments from both fields (track if stripping failed)
            stripped_source = strip_comments(entry.get("source", ""))
            stripped_new_code = strip_comments(entry.get("new_code", ""))

            # If both were stripped to empty, probable a tokenizer failure
            if not stripped_source and entry.get("source"):
                skipped += 1
            if not stripped_new_code and entry.get("new_code"):
                skipped += 1

            entry["source"] = stripped_source
            entry["new_code"] = stripped_new_code

            # Write stripped code to output
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    logging.info(f"Cleaned {count} entries and saved to {output_path}")
    logging.info(f"Skipped {skipped} fields due to tokenization errors")

if __name__ == "__main__":
    input_file = Path("datasets/vudenc/raw/vudenc_patched.jsonl")
    output_file = Path("datasets/vudenc/raw/vudenc_stripped.jsonl")

    clean_dataset(input_file, output_file)