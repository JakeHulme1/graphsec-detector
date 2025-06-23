import json
import argparse
import tokenize
from io import StringIO
from tqdm import tqdm
from pathlib import Path
from translation.parser.DFG import DFG_python
import tree_sitter_python as tspython 
from tree_sitter import Language, Parser
from translation.parser.utils import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token)
from typing import List, Dict, Any, Iterator, Tuple
from transformers import AutoTokenizer

LANG_SO = Path("translation/parser/my-languages.so")
PY_LANG = Language(tspython.language())

Point = Tuple[int, int]

def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """
    Iterate through json.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def write_jsonl(records: Iterator[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

def init_tokenizer():
    """
    Use the GraphCOdeBERT tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    return tokenizer

def tokenize_code(code:str, tokenizer):
    batch_encoding = tokenizer(code, return_offsets_mapping=True, add_special_tokens=True)
    return batch_encoding["offset_mapping"][0]

def init_dfg_parser() -> Parser:
    """
    Instantiates a Tree-Sitter Parser for Python using
    the pypi-installed tree_sitter_python grammar.
    """
    lang = Language(tspython.language())
    parser = Parser(lang)
    return parser

def parse_code_to_ast(code:str, parser: Parser):
    try:
        clean = remove_comments_and_docstrings(code, "python")
    except tokenize.TokenError:
        clean = code # fall back to original
    tree = parser.parse(clean.encode("utf-8"))
    return tree.root_node

def compute_token_maps(code: str, root_node) -> Dict[Tuple[Point, Point], Tuple[int, str]]:
    # break into lines so index_to_code_token can slice correctly
    code_lines = code.splitlines()

    # get back a dict: token_id -> (start_point, end_point)
    raw_index = tree_to_token_index(root_node)

    # build the reverse map: span -> (token_id, token_str)
    index_to_code: dict[tuple[tuple[int,int],tuple[int,int]], tuple[int,str]] = {}
    for token_id, span in enumerate(raw_index):
        start_pt, end_pt = span
        token_str = index_to_code_token(span, code_lines)
        index_to_code[span] = (token_id, token_str)

    return index_to_code

def main():

    # CLI setup
    p = argparse.ArgumentParser(
        description="DFG extractor")
    p.add_argument("input", help="path to .jsonl being split")
    p.add_argument("out_dir", help="where the DFGs will go, ready for training")
    p.add_argument("out_file_name", help="name of the output file")
    args = p.parse_args()

    out_path = Path(args.out_dir) / args.out_file_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    
    # Setup
    tokenizer = init_tokenizer()
    parser = init_dfg_parser()

    # Stream the .jsonl and iterate through each block
    def gen():

        for i, rec in enumerate(iter_jsonl(Path(args.input)), start=1):

            if i % 100 == 0:
                print(f"Processed {i} records")

            try:

                # Tokenize code (only get offsets)
                offsets = tokenize_code(rec["code"], tokenizer=tokenizer)
                rec["offset_mapping"] = offsets

                # parse to AST (may throw tokenize.TokenError)
                root = parse_code_to_ast(rec["code"], parser)

                # compute token-span map
                index_map = compute_token_maps(rec["code"], root)

                # print("TOKENS:")
                # for span, (tok_idx, tok_str) in sorted(index_map.items(), key=lambda x: x[1][0]):
                #     print(f"  [{tok_idx:2d}] '{tok_str}' at lines {span}")

                # run the extractor (may throw KeyError)
                nodes, _ = DFG_python(root, index_map, {})

                # flatten edges
                edges = [
                    (src, dst)
                    for (_, dst, _, _, src_idxs) in nodes
                    for src in src_idxs
                ]

            except (tokenize.TokenError, KeyError) as e:
                print(f"  Skipping record {i} due to parse/DFG error: {e}")
                yield None
                continue

            # # TESTING
            # print("TOKENS:", sorted(index_map.items(), key=lambda x: x[1][0]))
            # print("NODES:", nodes)
            # print("EDGES:", edges)

            rec["graph_nodes"] = nodes
            rec["graph_edges"] = edges
            yield rec

    skipped = 0
    with out_path.open("w", encoding="utf-8") as f:
        for rec in gen():
            if rec is None:
                skipped += 1
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"DFGs extracted! Skipped {skipped} malformed snippets.")


if __name__ == "__main__":
    main()