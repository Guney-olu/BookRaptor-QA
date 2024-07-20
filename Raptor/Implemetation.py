"""
RAG with RAPTOR indexing Implemetation
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import umap
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

RANDOM_SEED=104

def reduce_dimensionality_globally(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Performing global dimensionality reduction on embeddings using UMAP
    STEP-->.
    """
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def reduce_dimensionality_locally(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    """
    Performing local dimensionality reduction on embeddings using UMAP.
    """
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def find_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    """
    Determining the optimal number of clusters using the Bayesian Information Criterion (BIC).
    """
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    return n_clusters[np.argmin(bics)]


def cluster_with_GMM(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    """
    Clustering embeddings using Gaussian Mixture Models (GMM) and return cluster labels.
    """
    n_clusters = find_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def hierarchical_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
) -> List[np.ndarray]:
    """
    Performing hierarchical clustering on embeddings, combining global and local dimensionality reduction and clustering.
    """
    if len(embeddings) <= dim + 1:
        # Avoid clustering when there's insufficient data
        return [np.array([0]) for _ in range(len(embeddings))]

    reduced_embeddings_global = reduce_dimensionality_globally(embeddings, dim)
    global_clusters, n_global_clusters = cluster_with_GMM(
        reduced_embeddings_global, threshold
    )

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0
    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]

        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = reduce_dimensionality_locally(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = cluster_with_GMM(
                reduced_embeddings_local, threshold
            )
        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    return all_local_clusters


def generate_embeddings(texts):
    """
    Generating embeddings for a list of texts.
    """
    text_embeddings = model_emb_st.embed_documents(tqdm(texts, desc="Generating Embeddings"))
    text_embeddings_np = np.array(text_embeddings)
    return text_embeddings_np


def embed_and_cluster_texts(texts: List[str], titles: List[str]) -> pd.DataFrame:
    """
    Embedding texts, perform hierarchical clustering, and return a DataFrame with texts, embeddings, titles, and cluster labels.
    """
    text_embeddings_np = generate_embeddings(texts)
    cluster_labels = hierarchical_clustering(
        text_embeddings_np, 10, 0.1
    ) 
    df = pd.DataFrame() 
    df["text"] = texts 
    df["embd"] = list(text_embeddings_np)
    df["cluster"] = cluster_labels 
    return df


def format_text_for_summary(df: pd.DataFrame) -> str:
    """
    Formating text from DataFrame for summarization.
    """
    unique_txt = df["text"].tolist()
    return "--- --- \n --- --- ".join(unique_txt)


def embed_cluster_and_summarize_texts(
    texts: List[str], level: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Embed, cluster, and summarize texts, returning DataFrames with clusters and summaries.
    """
    df_clusters = embed_and_cluster_texts(texts)
    expanded_list = []

    for index, row in df_clusters.iterrows():
        for cluster in row["cluster"]:
            expanded_list.append(
                {"text": row["text"], "embd": row["embd"], "cluster": cluster}
            )
    expanded_df = pd.DataFrame(expanded_list)

    all_clusters = expanded_df["cluster"].unique()

    print(f"--Generated {len(all_clusters)} clusters--")

    template = """
    Summarize the given text below
    {context}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()

    summaries = []
    for i in tqdm(all_clusters, desc="Generating Summaries"):
        df_cluster = expanded_df[expanded_df["cluster"] == i]
        formatted_txt = format_text_for_summary(df_cluster)
        summaries.append(chain.invoke({"context": formatted_txt}))

    df_summary = pd.DataFrame(
        {
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        }
    )

    return df_clusters, df_summary


def recursive_embed_cluster_summarize(
    texts: List[str], level: int = 1, n_levels: int = 3
) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    print("recursive_embed_cluster_summarize")

    results = {} 

    df_clusters, df_summary = embed_cluster_and_summarize_texts(texts, level)

    results[level] = (df_clusters, df_summary)

    unique_clusters = df_summary["cluster"].nunique()
    if level < n_levels and unique_clusters > 1:
        new_texts = df_summary["summaries"].tolist()
        next_level_results = recursive_embed_cluster_summarize(
            new_texts, level + 1, n_levels
        )

        results.update(next_level_results)

    return results

if __name__ == "__main__":
    #Setting up the embedding params
    # check Raptor/embeddings.py to Know More
    from embeddings import hf_embedding_load
    model_emb_st = hf_embedding_load()

    # Setting up the model to use 
    # check Raptor/summarizer.py to Know More
    
    from summarizer import TextSummarizer
    summarizer = TextSummarizer(openai_api_key='your_api_key', openai_org_key='your_org_key')
    model = summarizer.openai_summarize()
    
    from extraction import PDFDocumentProcessor
    processor = PDFDocumentProcessor("Path to the file", title="book title")
    docs_with_metadata = processor.load_and_split()
    
    
    leaf_texts = docs_with_metadata.copy()
    title= "XYZ"
    results = recursive_embed_cluster_summarize(leaf_texts,title=title, level=1, n_levels=3)
    

