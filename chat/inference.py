import nltk
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from rank_bm25 import BM25Okapi
import json

model_st = SentenceTransformer('all-MiniLM-L6-v2')

question = "who is harry potter"


milvus_client = MilvusClient(uri="BookRaptor-QA/Raptor/milvus_chat.db")


collection_name = "rag_collection"


query_res = milvus_client.query(
    collection_name=collection_name,
    filter="",  # Empty filter to retrieve all entities
    output_fields=["text"],
    limit=1000  # Return only the 'text' field
)


all_texts = [entity['text'] for entity in query_res]

# print(f"Total texts retrieved: {len(all_texts)}")
# print(f"Sample texts: {all_texts[:5]}")


bm25 = BM25Okapi([text.split() for text in all_texts])


bm25_scores = bm25.get_scores(question.split())
bm25_top_n_indices = bm25.get_top_n(question.split(), all_texts, n=10)

# print(f"BM25 Top N Results: {bm25_top_n_indices}")


candidate_embeddings = [model_st.encode(text).tolist() for text in bm25_top_n_indices]


question_embedding = model_st.encode(question).tolist()


search_results = []
search_res = milvus_client.search(
    collection_name=collection_name,
    data=[question_embedding],
    limit=3,  # Return top 5 results
    search_params={"metric_type": "IP", "params": {}},  # Inner product distance
    output_fields=["text", "title", "page_number"],  # Return the text, title, and page number fields
)
search_results.extend(search_res[0])

print(f"Total search results: {len(search_results)}")


retrieved_lines_with_distances = [
    {
        "text": res["entity"]["text"],
        "title": res["entity"]["title"],
        "page_number": res["entity"]["page_number"],
        "distance": res["distance"]
    }
    for res in search_results
]


print(f"Retrieved Lines with Distances (Before Deduplication): {json.dumps(retrieved_lines_with_distances, indent=4)}")


unique_results = list({(result['text'], result['title'], result['page_number'], result['distance']): result for result in retrieved_lines_with_distances}.values())

print(f"Total unique results: {len(unique_results)}")


print(f"Unique Results: {json.dumps(unique_results, indent=4)}")


sorted_results = sorted(unique_results, key=lambda x: x["distance"], reverse=True)


top_results = sorted_results[:3]


print(json.dumps(top_results, indent=4))
