"""
Streamlit GUI for chatting 
TODO Add on spot processing and langchain for more chat model option

"""
import streamlit as st
import nltk
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from rank_bm25 import BM25Okapi
from openai import OpenAI
import json
from template.customtemplate import css, bot_template, user_template

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

client = OpenAI(api_key="", organization="")

model_st = SentenceTransformer('all-MiniLM-L6-v2')

def expand_query(query):
    synonyms = set()
    for word in query.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    expanded_query = list(synonyms)
    return expanded_query

def process_question(question):
    expanded_query = expand_query(question)
    expanded_query.append(question) 
    milvus_client = MilvusClient(uri="/Users/aryanrajpurohit/BookRaptor-QA/demo_db/bookfusion.db")
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


    # Hybrid retrieval: re-rank candidates using DPR and Milvus
    search_results = []
    for embedding in expanded_query_embeddings:
        search_res = milvus_client.search(
            collection_name=collection_name,
            data=[embedding],
            limit=1,  # Return top 3 results
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

    top_results = sorted_results[:2]

    return top_results

def query_gpt3(question, context):
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer according to the context:"
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    max_tokens=100,
    messages=[{"role": "user", "content": prompt}])
    print(response.choices[0].message.content)
    return response.choices[0].message.content


st.set_page_config(page_title="RAG Chat Application",page_icon=":books:")
st.header("Chat with Mr Mars")

question = st.text_input("Enter your question!!!")

if st.button("Submit"):
    if question:
        results = process_question(question)
        context = " ".join([result["text"] for result in results])
        title = " & ".join([result["title"] for result in results])  # Combine titles if there are multiple results
        answer = query_gpt3(question, context)
        rendered_template = bot_template.replace("{{TITLE}}", title).replace("{{MSG}}", answer)
        st.write(rendered_template, unsafe_allow_html=True)

    else:
        st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
