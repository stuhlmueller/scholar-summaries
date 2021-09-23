# scholar-summaries

How to run:

1. Check out the repository: `git clone git@github.com:stuhlmueller/scholar-summaries.git` 
2. `cd scholar-summaries`
2. Make a file `secrets` with contents like this:
```
export semantic_scholar_api_key="..."
export serpapi_api_key="..."
export openai_api_key="..."
```
3. Install [Poetry](https://python-poetry.org/docs/) if you don’t have it
4. Run `poetry install`
5. Run `poetry shell`
6. Run `source ./secrets`
7. Run `streamlit run scholarize.py` which will open a web browser
