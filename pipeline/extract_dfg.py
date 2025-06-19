import json
from pathlib import Path
from translation.parser.DFG import DFG_Python
from tree_sitter import Language, Parser
from translation.parser.utils import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from typing import List, Dict, Any, Iterator, Tuple, Node, Edge

def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """
    Iterate through json instead of reading.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

# def parse_code_to_ast(code:str) -> 
#
# def compute_token_maps(code:str, root_node)
# 
# def run_dfg(root_node, index_to_code) -> (Nodes, Edges)
# 
# def extract_graph(code:str) -> (Nodes, Edges)

def main():
    dfg_builder = DFG_Python