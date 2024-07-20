"""
Script to show how retrieved works
Techniques used:
Query expansion
BM25
Hybrid retrieval: re-rank candidates using DPR
"""
import argparse
import json
import nltk
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from rank_bm25 import BM25Okapi

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Ensure required NLTK data is downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')

def expand_query(query):
    synonyms = set()
    for word in query.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    expanded_query = list(synonyms)
    return expanded_query

def main(question, db_path):
    model_st = SentenceTransformer('all-MiniLM-L6-v2')

    expanded_query = expand_query(question)
    expanded_query.append(question)

    milvus_client = MilvusClient(uri=db_path)

    collection_name = "BF_collection"

    query_res = milvus_client.query(
        collection_name=collection_name,
        filter="",  
        output_fields=["text"],
        limit=10
    )

    all_texts = [entity['text'] for entity in query_res]
    bm25 = BM25Okapi([text.split() for text in all_texts])
    bm25_scores = bm25.get_scores(question.split())

    bm25_top_n_indices = bm25.get_top_n(question.split(), all_texts, n=10)

    candidate_embeddings = [model_st.encode(text).tolist() for text in bm25_top_n_indices]

    expanded_query_embeddings = [model_st.encode(term).tolist() for term in expanded_query]

    search_results = []
    for embedding in expanded_query_embeddings:
        search_res = milvus_client.search(
            collection_name=collection_name,
            data=[embedding],
            limit=3, 
            search_params={"metric_type": "IP", "params": {}},  
            output_fields=["text", "title"], 
        )
        search_results.extend(search_res[0])

    retrieved_lines_with_distances = [
        {
            "text": res["entity"]["text"],
            "title": res["entity"]["title"],
            "distance": res["distance"]
        }
        for res in search_results
    ]

    unique_results = list({(result['text'], result['title'], result['distance']): result for result in retrieved_lines_with_distances}.values())

    sorted_results = sorted(unique_results, key=lambda x: x["distance"], reverse=True)

    top_results = sorted_results[:3]

    print(json.dumps(top_results, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a query against the Milvus database with BM25 and DPR re-ranking.")
    parser.add_argument("question", type=str, help="The question to query.")
    parser.add_argument("db_path", type=str, help="The path to the Milvus database.")

    args = parser.parse_args()

    main(args.question, args.db_path)
