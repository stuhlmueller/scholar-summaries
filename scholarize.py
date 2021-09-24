import nltk
import openai
import os
import requests
import streamlit as st

from dataclasses import dataclass
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from serpapi import GoogleSearch
from typing import Any


nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

semantic_scholar_api_key = os.environ["semantic_scholar_api_key"]
serpapi_api_key = os.environ["serpapi_api_key"]
openai.api_key = os.environ["openai_api_key"]

semantic_scholar_headers = {"x-api-key": semantic_scholar_api_key}


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
        engine="davinci-instruct-beta-v2", prompt=prompt, max_tokens=128, temperature=0
    )
    return completion_result["choices"][0]["text"].strip()


conclusions_prompt = """
List the main conclusions of the following study

Study abstract: "{text}"

Study conclusions:
-""".strip()


def list_conclusions(text):
    prompt = conclusions_prompt.format(text=text)
    completion_result = openai.Completion.create(
        engine="davinci-instruct-beta-v2", prompt=prompt, max_tokens=256, temperature=0
    )
    result_text = completion_result["choices"][0]["text"].strip()
    return [line.strip("- ") for line in result_text.split("\n")]


@dataclass(order=False)
class Claim:
    source: Any
    text: str

    def __lt__(self, other):
        if isinstance(other, Claim):
            return self.text < other.text
        return True

    def __gt__(self, other):
        if isinstance(other, Claim):
            return self.text > other.text
        return False


def show_sorted_results(question, claims):
    documents = [claim.text for claim in claims]
    results = openai.Engine(id="babbage-search-index-v1").search(
        documents=documents, query=question, version="alpha"
    )
    scored_claims = sorted(
        [(datum["score"], claim) for (datum, claim) in zip(results["data"], claims)],
        reverse=True,
    )
    for (score, claim) in scored_claims:
        with st.expander(claim.text):
            source = claim.source
            st.markdown(f"*{source.get('title')}*")
            st.write(source.get("abstract"))
            authors = " ".join([author["name"] for author in source.get("authors")])
            st.write(
                f"{source.get('citationCount')} citations - {source.get('year')} - {authors} - [url]({source.get('url')})"
            )


def main():
    question = st.text_input(
        "Research question", help="For example: How does creatine affect cognition?"
    )
    google_query = remove_stopwords(question)

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

    if not scholar_results:
        return

    bar = st.progress(0.0)
    progress_text = st.empty()

    claims = []

    for (i, scholar_result) in enumerate(scholar_results):

        bar.progress((i + 1) / len(scholar_results))

        title = scholar_result.get("title")

        progress_text.write(f"Parsing {title}... ({i+1}/{len(scholar_results)})")

        if not title:
            continue

        params = {"query": title}

        response = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params,
            headers=semantic_scholar_headers,
        )
        response_json = response.json()

        data = response_json.get("data")

        if not data:
            continue

        paper_id = data[0].get("paperId")

        if not paper_id:
            continue

        params = {"fields": "title,abstract,authors,citationCount,url,year"}

        paper_detail_response = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}",
            params=params,
            headers=semantic_scholar_headers,
        )

        paper_detail = paper_detail_response.json()

        abstract = paper_detail.get("abstract")
        if not abstract:
            continue

        conclusions = list_conclusions(text=abstract)

        for conclusion in conclusions:
            claims.append(Claim(text=conclusion, source=paper_detail))

    bar.empty()
    progress_text.write("")

    show_sorted_results(question, claims)


main()
