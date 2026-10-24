# Apify Reddit data quality check

A quick check of whether data from the Apify Reddit actor (`harshmaur/reddit-scraper`) is good enough to answer questions like "how do people feel about Claude AI lately" or "which brands come up next to Nike". You run a test question, read the quality report and some sample rows, and score it by hand. It's not a product, just a notebook plus a small `helpers.py`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and put in your own Apify token. Add `ANTHROPIC_API_KEY` too if you want to use the optional Claude judge cell.

Register the kernel so Jupyter can find the venv:

```bash
python -m ipykernel install --user --name apify-dq --display-name "Apify DQ"
```

Set up nbstripout once per clone, so outputs get stripped before each commit. Heads up, this edits your git config:

```bash
nbstripout --install
```

If you don't have `.env` set up yet, the notebook will just ask for the token when it starts.

## Running a question

1. Open `apify_data_quality_check.ipynb` with the "Apify DQ" kernel.
2. Run the setup cell and the `TESTS` cell.
3. Set `QID` to one question (for example `q2_claude_sentiment`) and run that cell. This is the one that calls Apify.
4. Run the report and sample cells, then the question specific cell that matches.
5. Fill in a row in the scoring sheet.

Do one question at a time. Don't loop over all of them.

## Cost warning

The actor charges per result. Keep `maxPostsCount` and `maxCommentsCount` small while testing. Each run is cached in `apify_cache/` by actor + input, so rerunning the same cell is free. Change the input or set `use_cache=False` and you pay again. The real charge shows on the run page in Apify Console.

## Team notes

* Everyone uses their own Apify token.
* Never commit `.env`. It's already in `.gitignore`.
* If a token leaks (pasted in chat, committed, in a screenshot), rotate it in Apify Console right away.
* Scraped data stays out of the repo. `apify_cache/` and `data_quality_scores.csv` are ignored.
* Notebook outputs get stripped before commit (nbstripout), so no data or tokens end up in git.

## Tests

The helpers can be tested with fake data, no Apify calls:

```bash
python test_helpers.py
```
