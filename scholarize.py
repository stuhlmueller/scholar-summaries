import time
import asyncio
import aiohttp
import openai
import os
import requests
import streamlit as st
import itertools

from dataclasses import dataclass
from serpapi import GoogleSearch
from typing import Any


semantic_scholar_api_key = os.environ["semantic_scholar_api_key"]
serpapi_api_key = os.environ["serpapi_api_key"]
openai_api_key = os.environ["openai_api_key"]

openai.api_key = openai_api_key

semantic_scholar_headers = {"x-api-key": semantic_scholar_api_key}


conclusions_prompt = """
Accurately list all of the conclusions of the following study:

Abstract of the study: "{text}"

Conclusions of the study (one sentence each):
-""".strip()


async def list_conclusions(session, text):
    prompt = conclusions_prompt.format(text=text[:4000])
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}",
    }
    engine = "davinci-instruct-beta-v2"
    data = {
        "prompt": prompt,
        "max_tokens": 256,
        "temperature": 0
    }
    response = await session.post(
        f"https://api.openai.com/v1/engines/{engine}/completions",
        json=data,
        headers=headers,
    )
    completion_result = await response.json()
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
            st.markdown(f"[{source.get('title')}]({source.get('url')})")
            st.write(source.get("abstract"))
            authors = " ".join([author["name"] for author in source.get("authors")])
            st.write(
                f"{source.get('citationCount')} citations - {source.get('year')} - {authors}"
            )


async def scholar_result_to_claims(session, scholar_result):
    title = scholar_result.get("title")
    if not title:
        return [], "No title found"
    params = {"query": title}
    response = await session.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params=params,
        headers=semantic_scholar_headers,
    )
    response_json = await response.json()
    data = response_json.get("data")
    if not data:
        return [], title
    paper_id = data[0].get("paperId")
    if not paper_id:
        return [], title
    params = {"fields": "title,abstract,authors,citationCount,url,year"}
    paper_detail_response = await session.get(
        f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}",
        params=params,
        headers=semantic_scholar_headers,
    )
    paper_detail = await paper_detail_response.json()
    abstract = paper_detail.get("abstract")
    if not abstract:
        return [], title
    conclusions = await list_conclusions(session, text=abstract)
    claims = []
    for conclusion in conclusions:
        claims.append(Claim(text=conclusion, source=paper_detail))
    return claims, title


async def scholar_results_to_claims(scholar_results, set_progress):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for scholar_result in scholar_results:
            task = asyncio.create_task(
                scholar_result_to_claims(session, scholar_result)
            )
            task.title = scholar_result.get("title")
            tasks.append(task)

        i = 0
        claims = []
        for task in asyncio.as_completed(tasks):
            task_claims, title = await task
            claims += task_claims
            i += 1
            set_progress(
                i / len(tasks), f"Extracted claims from '{title}' ({i}/{len(tasks)})"
            )
        return claims


async def main():
    question = st.text_input(
        "Research question", help="For example: How does creatine affect cognition?"
    )
    google_query = question

    if not google_query:
        return

    params = {
        "engine": "google_scholar",
        "q": google_query,
        "api_key": serpapi_api_key,
        "num": 20,
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    scholar_results = results["organic_results"]

    if not scholar_results:
        return

    bar = st.progress(0.0)
    progress_text = st.empty()

    def set_progress(perc, text):
        bar.progress(perc)
        progress_text.write(text)

    claims = await scholar_results_to_claims(scholar_results, set_progress)

    unique_claims = []
    seen_claim_texts = set()
    for claim in claims:
        if not isinstance(claim, Claim):
            # Encountered exception
            print(claim)
            continue
        if claim.text in seen_claim_texts:
            continue
        unique_claims.append(claim)
        seen_claim_texts.add(claim.text)

    bar.progress(1.0)

    progress_text.write("Ranking claims...")

    show_sorted_results(question, unique_claims)

    bar.empty()
    progress_text.write("")


asyncio.run(main())
