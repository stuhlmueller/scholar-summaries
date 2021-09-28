import asyncio
import aiohttp
import openai
import os
import streamlit as st
import nest_asyncio

from dataclasses import dataclass
from serpapi import GoogleSearch
from typing import Any
from sentence_transformers import CrossEncoder


semantic_scholar_api_key = os.environ["semantic_scholar_api_key"]
serpapi_api_key = os.environ["serpapi_api_key"]
openai_api_key = os.environ["openai_api_key"]

openai.api_key = openai_api_key

semantic_scholar_headers = {"x-api-key": semantic_scholar_api_key}

@st.cache(hash_funcs={CrossEncoder: (lambda _: None)})
def get_msmarco_encoder():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', max_length=512)

msmarco_encoder = get_msmarco_encoder()

conclusions_prompt = """
Accurately list all of the conclusions of the following studies:

Abstract of study 1: "Background and aims: Creatine is a supplement used by sportsmen to increase athletic performance by improving energy supply to muscle tissues. It is also an essential brain compound and some hypothesize that it aids cognition by improving energy supply and neuroprotection. The aim of this systematic review is to investigate the effects of oral creatine administration on cognitive function in healthy individuals. Methods: A search of multiple electronic databases was performed for the identification of randomized clinical trials (RCTs) examining the cognitive effects of oral creatine supplementation in healthy individuals. Results: Six studies (281 individuals) met our inclusion criteria. Generally, there was evidence that short term memory and intelligence/reasoning may be improved by creatine administration. Regarding other cognitive domains, such as long‐term memory, spatial memory, memory scanning, attention, executive function, response inhibition, word fluency, reaction time and mental fatigue, the results were conflicting. Performance on cognitive tasks stayed unchanged in young individuals. Vegetarians responded better than meat‐eaters in memory tasks but for other cognitive domains no differences were observed. Conclusions: Oral creatine administration may improve short‐term memory and intelligence/reasoning of healthy individuals but its effect on other cognitive domains remains unclear. Findings suggest potential benefit for aging and stressed individuals. Since creatine is safe, future studies should include larger sample sizes. It is imperative that creatine should be tested on patients with dementias or cognitive impairment. HIGHLIGHTSOral creatine supplementation improves memory of healthy adults.Findings suggest potential benefit for aging and stressed individuals.Future trials should investigate the effect of creatine administration on individuals with dementia or mild cognitive impairment."

Conclusions of study 1 (one sentence each):
- Short term memory and intelligence/reasoning may be improved by creatine administration
- Regarding other cognitive domains, such as long‐term memory, spatial memory, memory scanning, attention, executive function, response inhibition, word fluency, reaction time and mental fatigue, the results of creatine administration were conflicting
- Performance on cognitive tasks stayed unchanged in young individuals after creatine administration
- Vegetarians responded better than meat‐eaters in memory tasks but for other cognitive domains no differences were observed after creatine administration
- Oral creatine administration may improve short‐term memory and intelligence/reasoning of healthy individuals but its effect on other cognitive domains remains unclear
- Creatine has potential benefits for aging and stressed individuals
- Creatine should be tested on patients with dementias or cognitive impairment
- Oral creatine supplementation improves memory of healthy adults
- Findings suggest potential benefit of creatine administration for aging and stressed individuals
- Future trials should investigate the effect of creatine administration on individuals with dementia or mild cognitive impairment

Abstract of study 2: "{text}"

Conclusions of study 2 (one sentence each):
-""".strip()


async def list_conclusions(session, text):
    prompt = conclusions_prompt.format(text=text[-2000:])
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}",
    }
    engine = "davinci-instruct-beta-v2"
    data = {
        "prompt": prompt,
        "max_tokens": 600,
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


@st.cache(persist=True, show_spinner=False)
def score_claims_openai(question, claims):
    documents = [claim.text for claim in claims]
    results = openai.Engine(id="babbage-search-index-v1").search(
        documents=documents, query=question, version="alpha"
    )
    scored_claims = sorted(
        [(datum["score"], claim) for (datum, claim) in zip(results["data"], claims)],
        reverse=True,
    )
    return scored_claims


@st.cache(persist=True, show_spinner=False)
def score_claims_msmarco(question, claims):
    scores = msmarco_encoder.predict(
        [(question, claim.text) for claim in claims])
    scored_claims = sorted(zip(scores, claims), reverse=True)
    return scored_claims


def show_sorted_results(question, claims):
    scored_claims = score_claims_msmarco(question, claims)
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


async def async_scholar_results_to_claims(scholar_results, set_progress):
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


def get_event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # no event loop running:
        loop = asyncio.new_event_loop()
    else:
        nest_asyncio.apply()
    return loop


@st.cache(suppress_st_warning=True, persist=True, show_spinner=False)
def scholar_results_to_claims(scholar_results, set_progress):
    loop = get_event_loop()    
    result = loop.run_until_complete(async_scholar_results_to_claims(scholar_results, set_progress))
    return result


def main():
    app_state = st.experimental_get_query_params()
    url_question = app_state.get("q", [""])[0]
    question = st.text_input(
        "Research question", value=url_question, help="For example: How does creatine affect cognition?"
    )
    st.experimental_set_query_params(q=question)
    if not question:
        return

    params = {
        "engine": "google_scholar",
        "q": question,
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

    claims = scholar_results_to_claims(scholar_results, set_progress)

    unique_claims = []
    seen_claim_texts = set()
    for claim in claims:
        if claim.text in seen_claim_texts:
            continue
        unique_claims.append(claim)
        seen_claim_texts.add(claim.text)

    bar.progress(1.0)

    progress_text.write("Ranking claims...")

    show_sorted_results(question, unique_claims)

    bar.empty()
    progress_text.write("")


main()
