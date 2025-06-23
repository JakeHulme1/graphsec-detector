import json
import argparse
from tqdm import tqdm
from pathlib import Path
from translation.parser.DFG import DFG_Python
from tree_sitter import Language, Parser
from translation.parser.utils import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token)
from typing import List, Dict, Any, Iterator, Tuple
from transformers import AutoTokenizer

LANG_SO = Path("translation/parser/my-languages.so")
PY_LANG = Language(str(LANG_SO), "python")

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
    batch_encoding = tokenizer(code, return_offsets_mapping=True)
    return(
        batch_encoding["input_ids"][0],
        batch_encoding["attention_mask"][0],
        batch_encoding["token_type_ids"][0],
        batch_encoding["offset_mapping"][0],

    )

def init_dfg_parser(lib_path: str, language: str) -> Parser:
    """
    Load the compiled Tree-Sitter grammar (.so) and return a Parser
    bound to the passed language (Python in this project).
    """
    lang = Language(lib_path, language)
    parser = Parser()
    parser.set_language(lang)
    return parser

def parse_code_to_ast(code:str, parser: Parser):
    # strip comments, alreafy done in vudenc, this is just a failsafe
    clean_code = remove_comments_and_docstrings(code)
    # feed into tree-sitter
    tree = parser.parse(bytes(clean_code, "utf-8"))
    # return the root
    return tree.root_node

def compute_token_maps(code: str, root_node) -> dict[tuple[int, int], tuple[int, str]]:
    clean_code = remove_comments_and_docstrings(code)
    raw_index = tree_to_token_index(root_node, clean_code)

    index_to_code = {}
    for (start_pt, end_pt), idx in raw_index.items():
        token_str = index_to_code_token(start_pt, end_pt, clean_code)
        index_to_code[(start_pt, end_pt)] = (idx, token_str)

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
    parser = init_dfg_parser(str(LANG_SO), "python")

    # Stream the .jsonl and iterate through each block
    def gen():
        for i, rec in enumerate(iter_jsonl(Path(args.input)), start=1):

            if i % 100 == 0:
                print(f"Processed {i} records")

            # Tokenize code (only get offsets)
            offsets = tokenize_code(rec["code"], tokenizer=tokenizer)
            rec["offset_mapping"] = offsets.tolist()

            # Generate AST + token map
            root = parse_code_to_ast(rec["code"], parser)
            index_map = compute_token_maps(rec["code"], root)

            # DFG extraction
            nodes, _ = DFG_Python(root, index_map, {})
            rec["graph_nodes"] = nodes
            rec["graph_edges"] = [
                (src, dst)
                for (_, dst, _, _, src_idxs) in nodes
                for src in src_idxs
            ]
            yield rec

    write_jsonl(gen(), out_path)
    print("DFGs extracted!")

if __name__ == "__main__":
    main()