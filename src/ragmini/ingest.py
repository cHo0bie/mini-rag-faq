from __future__ import annotations
import pathlib, re
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

@dataclass
class Doc:
    id: str
    title: str
    url: str
    text: str

def read_docs(root_dir: str) -> List[Doc]:
    root = pathlib.Path(root_dir)
    docs: List[Doc] = []
    for p in sorted(root.rglob("*.md")):
        title = p.stem.replace("_"," ").strip().title()
        url   = f"file://{p.as_posix()}"
        text  = p.read_text(encoding="utf-8")
        docs.append(Doc(id=p.name, title=title, url=url, text=text))
    return docs

def split_into_chunks(text: str, size: int = 400, overlap: int = 60) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    i, chunks = 0, []
    while i < len(text):
        chunk = text[i:i+size]
        chunks.append(chunk)
        i += size - overlap
    return chunks

def build_corpus(docs: List[Doc]) -> Tuple[list, list]:
    corpus, meta = [], []
    for d in docs:
        for chunk in split_into_chunks(d.text):
            corpus.append(chunk)
            meta.append({"doc_id": d.id, "title": d.title, "url": d.url})
    return corpus, meta

def build_tfidf_index(corpus: list):
    vect = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    mat  = vect.fit_transform(corpus)
    return vect, mat
