import math
from collections import defaultdict
from preprocessing import preprocess
from indexer import load_index

def cosine_similarity(query_vec, doc_vec):
    #dot prod b/w query and doc
    dot = 0.0
    for term, q_weight in query_vec.items():
        if term in doc_vec:
            dot += q_weight * doc_vec[term]

    # magnitude of both vectors
    mag_q = math.sqrt(sum(w ** 2 for w in query_vec.values()))
    mag_d = math.sqrt(sum(w ** 2 for w in doc_vec.values()))

    #avoiding from zero divison
    if mag_q == 0 or mag_d == 0:
        return 0.0

    return dot / (mag_q * mag_d)

def build_query_vector(query_tokens, tfidf_matrix, df, N):
    #tf-idf vector for query
    tf = defaultdict(int)
    for t in query_tokens:
        tf[t] += 1

    query_vec = {}
    for term, count in tf.items():
        if term in df:  #if term not in corpus -> skip!
            idf = math.log(N / df[term])
            query_vec[term] = count * idf
        #if term not in vocab-> just ignore..no score will be given

    return query_vec

def search(query_text, alpha=0.005):
    # index load karo
    tfidf_matrix, vocab, df = load_index()
    N = len(tfidf_matrix)

    # query preprocess - SAME pipeline
    tokens = preprocess(query_text)

    if not tokens:
        print("[-] Query mein koi useful term nahi mila after preprocessing")
        return []

    # query vector
    query_vec = build_query_vector(tokens, tfidf_matrix, df, N)

    if not query_vec:
        print("[-] Query ke terms corpus ki vocabulary mein nahi hain")
        return []

    #cosine similarity against every doc
    scores = {}
    for doc_id, doc_vec in tfidf_matrix.items():
        score = cosine_similarity(query_vec, doc_vec)
        if score > alpha:  
            scores[doc_id] = score

    # sort by highest score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked