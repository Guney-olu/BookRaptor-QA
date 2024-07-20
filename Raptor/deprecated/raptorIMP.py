from typing import Optional
import numpy as np
import umap
import tqdm
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.mixture import GaussianMixture
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Loading the embedding model and summarizing model
model_emb_st = SentenceTransformer('all-MiniLM-L6-v2')

def t5_summary(text):
    tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-v1_1-base")
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_new_tokens=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


#loading the txt files from the dir 
loader = DirectoryLoader('/Users/aryanrajpurohit/BookRaptor-QA/corpus/test', glob="**/*.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

docs = text_splitter.split_documents(docs)
texts = [doc.page_content for doc in docs]

global_embeddings = [model_emb_st.encode(txt) for txt in tqdm.tqdm(texts, desc="Generating embeddings")]

def reduce_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)

dim = 2
global_embeddings_reduced = reduce_cluster_embeddings(global_embeddings, dim)

def get_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 50, random_state: int = 1234):
    max_clusters = min(max_clusters, len(embeddings))
    bics = [GaussianMixture(n_components=n, random_state=random_state).fit(embeddings).bic(embeddings)
            for n in range(1, max_clusters)]
    return np.argmin(bics) + 1

def gmm_clustering(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state).fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


labels, _ = gmm_clustering(global_embeddings_reduced, threshold=0.5)

simple_labels = [label[0] if len(label) > 0 else -1 for label in labels]

df = pd.DataFrame({
    'Text': texts,
    'Embedding': list(global_embeddings_reduced),
    'Cluster': simple_labels
})

def format_cluster_texts(df):
    clustered_texts = {}
    for cluster in df['Cluster'].unique():
        cluster_texts = df[df['Cluster'] == cluster]['Text'].tolist()
        clustered_texts[cluster] = " --- ".join(cluster_texts)
    return clustered_texts

clustered_texts = format_cluster_texts(df)

summaries = {}
for cluster, text in tqdm.tqdm(clustered_texts.items(), desc="Generating summaries"):
    summary = t5_summary(text)
    summaries[cluster] = summary


embedded_summaries = [model_emb_st.encode(summary) for summary in summaries.values()]
embedded_summaries_np = np.array(embedded_summaries)
labels, _ = gmm_clustering(embedded_summaries_np, threshold=0.5)
simple_labels = [label[0] if len(label) > 0 else -1 for label in labels]


clustered_summaries = {}
for i, label in enumerate(simple_labels):
    if label not in clustered_summaries:
        clustered_summaries[label] = []
    clustered_summaries[label].append(list(summaries.values())[i])

final_summaries = {}
for cluster, texts in clustered_summaries.items():
    combined_text = ' '.join(texts)
    summary = t5_summary(combined_text)
    final_summaries[cluster] = summary

    
texts_from_df = df['Text'].tolist()
texts_from_clustered_texts = list(clustered_texts.values())
texts_from_final_summaries = list(final_summaries.values())

combined_texts = texts_from_df + texts_from_clustered_texts + texts_from_final_summaries

# Saving the combined data in Milvus DB
from pymilvus import MilvusClient
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

save_to_milvus(combined_texts)