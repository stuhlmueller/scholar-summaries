import asyncio
import aiohttp
import openai
import os
import streamlit as st
import nest_asyncio
import sentence_transformers

from serpapi import GoogleSearch
from sentence_transformers import CrossEncoder
from claim import Claim


semantic_scholar_api_key = os.environ["semantic_scholar_api_key"]
serpapi_api_key = os.environ["serpapi_api_key"]
openai_api_key = os.environ["openai_api_key"]

openai.api_key = openai_api_key

semantic_scholar_headers = {"x-api-key": semantic_scholar_api_key}


@st.cache(hash_funcs={CrossEncoder: (lambda _: ("msmarco", None))})
def get_msmarco_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512)


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
    data = {"prompt": prompt, "max_tokens": 600, "temperature": 0}
    response = await session.post(
        f"https://api.openai.com/v1/engines/{engine}/completions",
        json=data,
        headers=headers,
    )
    completion_result = await response.json()
    result_text = completion_result["choices"][0]["text"].strip()
    return [line.strip("- ") for line in result_text.split("\n")]


@st.cache(persist=True, show_spinner=False, allow_output_mutation=True)
def score_claims_openai(question, claims):
    documents = [claim.text for claim in claims]
    results = openai.Engine(id="babbage-search-index-v1").search(
        documents=documents, query=question, version="alpha"
    )
    return [(datum["score"], claim) for (datum, claim) in zip(results["data"], claims)]


def score_claims_msmarco(question, claims):
    scores = msmarco_encoder.predict([(question, claim.text) for claim in claims])
    return zip(scores, claims)


def sort_score_claims(question, claims):
    scored_claims = score_claims_msmarco(question, claims)
    return sorted(scored_claims, reverse=True)


async def scholar_result_to_claims(session, scholar_result):

    cache_key = scholar_result.get("link", scholar_result["title"])
    if cache_key in st.session_state.claims:
        claims = st.session_state.claims[cache_key]
        return claims

    def cache(*values):
        st.session_state.claims[cache_key] = values
        return values

    title = scholar_result.get("title")
    if not title:
        return cache([], "No title found")
    params = {"query": title}
    response = await session.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params=params,
        headers=semantic_scholar_headers,
    )
    response_json = await response.json()
    data = response_json.get("data")
    if not data:
        return cache([], title)
    paper_id = data[0].get("paperId")
    if not paper_id:
        return cache([], title)
    params = {"fields": "title,abstract,authors,citationCount,url,year"}
    paper_detail_response = await session.get(
        f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}",
        params=params,
        headers=semantic_scholar_headers,
    )
    paper_detail = await paper_detail_response.json()
    abstract = paper_detail.get("abstract")
    if not abstract:
        return cache([], title)
    conclusions = await list_conclusions(session, text=abstract)
    claims = []
    for conclusion in conclusions:
        claims.append(Claim(text=conclusion, source=paper_detail))
    return cache(claims, title)


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


def scholar_results_to_claims(scholar_results, set_progress):
    loop = get_event_loop()
    result = loop.run_until_complete(
        async_scholar_results_to_claims(scholar_results, set_progress)
    )
    return result


def filter_scholar_results(scholar_results, min_citations):
    filtered = []
    for result in scholar_results:
        inline_links = result.get("inline_links")
        if not inline_links:
            continue
        cited_by = inline_links.get("cited_by")
        if not cited_by:
            continue
        total = int(cited_by.get("total", 0))
        if total > min_citations:
            filtered.append(result)
    return filtered


@st.cache(suppress_st_warning=True, persist=True, show_spinner=False)
def get_scholar_results(query, min_year):
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": serpapi_api_key,
        "num": 20,
        "as_ylo": min_year,
    }
    search = GoogleSearch(params)
    data = search.get_dict()
    return data["organic_results"]


def main():

    if "claims" not in st.session_state:
        st.session_state["claims"] = {}

    app_state = st.experimental_get_query_params()
    url_question = app_state.get("q", [""])[0]
    question = st.text_input(
        "Research question",
        value=url_question,
        help="For example: How does creatine affect cognition?",
    )
    st.experimental_set_query_params(q=question)

    with st.expander("Options"):
        col1, col2 = st.columns(2)
        with col1:
            min_year = st.number_input(
                "Year at least", min_value=1950, max_value=2021, value=2011
            )
        with col2:
            min_citations = st.number_input("Citations at least", min_value=0, value=10)

    if not question:
        return

    bar = st.progress(0.0)
    progress_text = st.empty()

    progress_text.write("Retrieving papers...")

    raw_scholar_results = get_scholar_results(question, min_year)

    scholar_results = filter_scholar_results(raw_scholar_results, min_citations)

    if not scholar_results:
        return

    progress_text.write("Extracting claims...")

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

    sorted_scored_claims = sort_score_claims(question, unique_claims)

    for (score, claim) in sorted_scored_claims:
        with st.expander(claim.text):
            source = claim.source
            st.markdown(f"[{source.get('title')}]({source.get('url')})")
            st.write(source.get("abstract"))
            authors = " ".join([author["name"] for author in source.get("authors")])
            st.write(
                f"{source.get('citationCount')} citations - {source.get('year')} - {authors}"
            )

    bar.empty()
    progress_text.write("")


main()
