from flask import Flask, render_template, request, jsonify
import os
import re

# 1. New Imports required for the weights calculation
from vsm_retrieval import search, build_query_vector
from preprocessing import preprocess
from indexer import load_index

app = Flask(__name__)

# Load index once at startup
tfidf_matrix, vocab, df = load_index()
N = len(tfidf_matrix)

CORPUS_PATH = r"C:\Users\Ibad Ur Rehman\Desktop\Sem 6\IR\VSM_IR\corpus"

def get_filename(doc_id):
    for fname in os.listdir(CORPUS_PATH):
        if fname.endswith(".txt"):
            match = re.search(r"\d+", fname)
            if match and int(match.group()) == doc_id:
                return fname
    return f"Document {doc_id}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search_route():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query empty hai bhai"}), 400

    # Get the actual search results
    results = search(query)

    # --- START OF TEACHER'S REQUESTED CHANGE ---
    
    # 2. Preprocess the query to get individual tokens
    tokens = preprocess(query)
    
    # 3. Calculate the weights for each term in the query
    query_vec = build_query_vector(tokens, tfidf_matrix, df, N)
    
    # 4. Sort terms by weight (highest first) and format for frontend
    query_terms = [
        {"term": t, "weight": round(w, 4)}
        for t, w in sorted(query_vec.items(), key=lambda x: x[1], reverse=True)
    ]
    
    # --- END OF TEACHER'S REQUESTED CHANGE ---

    formatted = []
    for rank, (doc_id, score) in enumerate(results, start=1):
        formatted.append({
            "rank": rank,
            "doc_id": doc_id,
            "filename": get_filename(doc_id),
            "score": round(score, 6)
        })

    # 5. Add query_terms to the final JSON return
    return jsonify({
        "query": query,
        "total": len(formatted),
        "results": formatted,
        "query_terms": query_terms  # This allows the frontend to show the sidebar
    })

if __name__ == "__main__":
    app.run(debug=True)