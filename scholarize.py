import os
import requests
import openai
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from serpapi import GoogleSearch

nltk.download('punkt')
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

semantic_scholar_api_key = os.environ["semantic_scholar_api_key"]
serpapi_api_key = os.environ["serpapi_api_key"]
openai.api_key = os.environ["openai_api_key"]

semantic_scholar_headers = {
    "x-api-key": semantic_scholar_api_key
}

def remove_stopwords(text):
    text_tokens = word_tokenize(text)
    text_tokens = [w for w in text_tokens if not w in stop_words]
    return " ".join(text_tokens)

summarization_prompt = """
State accurately what (if anything) the paper below says about the question "{question}".

Paper: "{text}"

Question: What (if anything) does the paper say about the question "{question}"?
Answer: The paper""".strip()

def summarize(text, question):
    prompt = summarization_prompt.format(text=text, question=question)
    completion_result = openai.Completion.create(
        engine="davinci-instruct-beta-v2",
        prompt=prompt,
        max_tokens=128,
        temperature=0
    )    
    return completion_result["choices"][0]["text"].strip()

def main():
    question = st.text_input("Research question", help="For example: How does poverty affect the brain?")
    google_query = remove_stopwords(question)
    # st.write(f"Question without stop words: {google_query}")

    if not google_query:
        return
    
    params = {
        "engine": "google_scholar",
        "q": google_query,
        "api_key": serpapi_api_key,
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    scholar_results = results["organic_results"]
    
    # st.json(organic_results)
    
    if not scholar_results:
        return

    bar = st.progress(0.0)
    
    for (i, scholar_result) in enumerate(scholar_results[:10]):

        bar.progress((i + 1) / len(scholar_results))
        
        title = scholar_result.get("title")
        # main_author = organic_results[0]["publication_info"]["authors"][0]["name"]
        
        if not title:
            continue
        
        params = {
            "query": title
        }
        
        response = requests.get("https://api.semanticscholar.org/graph/v1/paper/search", params=params, headers=semantic_scholar_headers)
        response_json = response.json()

        data = response_json.get("data")
        
        if not data:
            continue
        
        paper_id = data[0].get("paperId")

        if not paper_id:
            continue
        
        params = {
            "fields": "title,abstract,authors,citationCount,url,year"
        }
        
        paper_detail_response = requests.get(f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}", params=params, headers=semantic_scholar_headers)
        
        paper_detail = paper_detail_response.json()
        
        abstract = paper_detail.get("abstract")
        if not abstract:
            continue

        summary = summarize(text=abstract, question=question)

        st.markdown(f"#### {paper_detail.get('title')}")
        st.write(summary.capitalize())
        authors = " ".join([author["name"] for author in paper_detail.get("authors")])
        st.write(f"{paper_detail.get('citationCount')} citations - {paper_detail.get('year')} - {authors} - [url]({paper_detail.get('url')})")
        
    bar.empty()


main()        
