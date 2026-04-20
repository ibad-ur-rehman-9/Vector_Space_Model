import os
import re
import math
import pickle
from collections import defaultdict
from preprocessing import preprocess

CORPUS_PATH = r"C:\Users\Ibad Ur Rehman\Desktop\Sem 6\IR\VSM_IR\corpus\Trump Speechs\Trump Speechs"
INDEX_DIR = "index"

def load_corpus():
    # read all docs and make a dict of doc_id -> tokens
    docs = {}

    files = sorted(os.listdir(CORPUS_PATH))
    for fname in files:
        if not fname.endswith(".txt"):
            continue  

        # filename se number nikalo - "trump1.txt" -> 1
        match = re.search(r"\d+", fname)
        if not match:
            continue  #skip if number not found

        doc_id = int(match.group())
        fpath = os.path.join(CORPUS_PATH, fname)

        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        docs[doc_id] = preprocess(raw)

    print(f"[+] {len(docs)} documents load ho gaye")
    return docs

def build_tfidf(docs):
    N = len(docs)  # total documents

    #count df
    df = defaultdict(int)
    for tokens in docs.values():
        for term in set(tokens):  # set taake ek doc mein ek baar hi count ho
            df[term] += 1

    # vocabulary
    vocab = sorted(df.keys())
    term_index = {term: i for i, term in enumerate(vocab)}

    #now making tf-idf vector for each doc
    tfidf_matrix = {}  # doc_id -> {term: tfidf_score}

    for doc_id, tokens in docs.items():
        # tf: term frequency in this document
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1

        vec = {}
        for term, count in tf.items():
            idf = math.log(N / df[term]) 
            vec[term] = count * idf       

        tfidf_matrix[doc_id] = vec

    return tfidf_matrix, vocab, df

def save_index(tfidf_matrix, vocab, df):
    os.makedirs(INDEX_DIR, exist_ok=True)

    with open(os.path.join(INDEX_DIR, "tfidf_matrix.pkl"), "wb") as f:
        pickle.dump(tfidf_matrix, f)

    with open(os.path.join(INDEX_DIR, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    with open(os.path.join(INDEX_DIR, "df.pkl"), "wb") as f:
        pickle.dump(df, f)

    print("[+] Index save ho gaya - index/ folder mein dekho")

def load_index():
    with open(os.path.join(INDEX_DIR, "tfidf_matrix.pkl"), "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open(os.path.join(INDEX_DIR, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)

    with open(os.path.join(INDEX_DIR, "df.pkl"), "rb") as f:
        df = pickle.load(f)

    print("[+] Index load ho gaya")
    return tfidf_matrix, vocab, df

if __name__ == "__main__":
    docs = load_corpus()
    tfidf_matrix, vocab, df = build_tfidf(docs)
    save_index(tfidf_matrix, vocab, df)
    print(f"[+] Vocabulary size: {len(vocab)} terms")