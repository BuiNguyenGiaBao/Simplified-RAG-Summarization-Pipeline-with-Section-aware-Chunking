import os
import json
import argparse
import random
from typing import Any, Dict, List, Optional
from datasets import load_from_disk
from rulebase_chunkforpdf import process_document
from retrivel_tokenizer import DenseEncoder, MMRDenseRetriever, Document
from sumarized import T5Summarizer

DEFAULT_QUERY_TEMPLATES = [
    "Summarize the main contribution of this paper",
    "Summarize the proposed method and main findings of this paper",
    "What problem does this paper address and how does it solve it?",]


def ensur_dir(path:str):
    os.makedirs(path,exist_ok=True)

def write_jsonl(path: str, record: list[dict[str,Any]]):
    with open(path,'w',encoding='utf-8')as f:
        for r in record :
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
def append_jsonl(path: str, records: List[Dict[str, Any]]):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def clean_text_field(x: Any, join_with:str= '\n')->str:
    """
    the text can be word or string 
    """
    if x is None:
        return""
    if isinstance(x, list):
        parts = [str(i).strip() for i in x if str(i).strip()]
        return join_with.join(parts).strip()
    text = str(x)
    text = text.replace("/n", "\n")
    return text.strip()

    