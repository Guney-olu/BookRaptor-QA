"""
Helper func
"""

import tiktoken

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_token_count(texts):
    return [num_tokens_from_string(t) for t in texts]


# Milvus DB data save
from pymilvus import MilvusClient
from embeddings import model_emb_st
import tqdm

def prepare_data(combined_texts):
    data = []
    for i, text in enumerate(tqdm.tqdm(combined_texts, desc="Preparing data")):
        embedding = model_emb_st.encode(text).tolist()
        title = f"Textbook Title {i+1}"
        page_number = i + 1
        data.append({
            "id": i,
            "vector": embedding,
            "text": text,
            "title": title,
            "page_number": page_number
        })
    return data

def save_to_milvus(combined_texts):
    test_embedding = model_emb_st.encode("This is a test")
    embedding_dim = len(test_embedding)
    milvus_client = MilvusClient(uri="/BookRaptor-QA/Raptor/milvus_chat.db")
    collection_name = "rag_collection"
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)
    
    milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",  # Inner product distance
    consistency_level="Strong",  # Strong consistency level
    )
    data = prepare_data(combined_texts)
    milvus_client.insert(collection_name=collection_name, data=data)