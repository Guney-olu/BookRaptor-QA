"""
This py file have diffrently loaded embedding 
BC OF BUGS SOMETIMES
"""
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# Directly using sentence_tranformer
from sentence_transformers import SentenceTransformer

model_emb_st = SentenceTransformer('all-MiniLM-L6-v2')



## More reliant Method (Very Fast)
from langchain_community.embeddings import HuggingFaceEmbeddings

def hf_embedding_load():
    """
    Loading same model but using hf pineline
    """
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        #model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  
        )



# TODO Add more models
#Using hugging-face [MAX POOLING ->]
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import tqdm

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer_st = AutoTokenizer.from_pretrained(model_name)
model_st = AutoModel.from_pretrained(model_name)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# DEDICATED function for raptor (SHAPE IS KNOWN)
def get_global_embeddings(texts):
    embeddings = []
    for text in tqdm.tqdm(texts, desc="Embedding texts"):
        encoded_input = tokenizer_st(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            model_output = model_st(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)  # Normalize embeddings
        embeddings.append(sentence_embeddings.squeeze().tolist())
    return embeddings

def get_summary_embeddings_(summaries):
    embeddings = []
    for summary in tqdm.tqdm(summaries.values(), desc="Embedding texts"):
        encoded_input = tokenizer_st(summary, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            model_output = model_st(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)  # Normalize embeddings
        embeddings.append(sentence_embeddings.squeeze().tolist())
    return embeddings