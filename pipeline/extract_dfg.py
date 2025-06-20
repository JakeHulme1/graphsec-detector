import json
from pathlib import Path
from translation.parser.DFG import DFG_Python
from tree_sitter import Language, Parser
from translation.parser.utils import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from typing import List, Dict, Any, Iterator, Tuple, Node, Edge
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

def init_tokenizer(model_name: str):
    """
    Use the GraphCOdeBERT tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    return tokenizer

def tokenize_code(code:str, tokenizer) -> Tuple[List[int], List[int], List[int], List[Tuple[int, int]]]:
    return tokenizer(code, return_tensors='pt')

def init_dfg_parser(lib_path: str, language: str) -> Parser:
    """
    Load the compiled Tree-Sitter grammar (.so) and return a Parser
    bound to the passed language (Python in this project).
    """
    LANG = Language(lib_path, language)
    parser = Parser()
    parser.set_language(LANG)
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
        index_to_code[(start_pt, end_pt)]

    return index_to_code

def init_dfg_builder():
    parser = Parser()
    parser.set_langauge(PY_LANG)
    return parser