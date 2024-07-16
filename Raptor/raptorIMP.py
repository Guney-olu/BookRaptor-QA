import tiktoken
from langchain_community.document_loaders import DirectoryLoader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embeddings import model_emb_st
from helper import get_token_count
#loading the txt files 
loader = DirectoryLoader('PATH TO THE DIR', glob="**/*.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

docs = text_splitter.split_documents(docs)
texts = [doc.page_content for doc in docs]

#count = get_token_count(texts) 

d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]
)

global_embeddings = [model_emb_st.encode(txt) for txt in texts]




