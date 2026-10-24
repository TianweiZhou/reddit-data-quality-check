"""Helpers for the Apify Reddit data quality notebook.

Kept in a plain module so they can be tested without running any actor.
"""
import hashlib
import json
import os
import re
import time
from collections import Counter
from decimal import Decimal

import pandas as pd

REMOVED_BODIES = {"[removed]", "[deleted]"}


def _to_dict(o):
    if isinstance(o, dict):
        return o
    for fn in ("model_dump", "dict"):
        if hasattr(o, fn):
            try:
                return getattr(o, fn)(by_alias=True)
            except TypeError:
                return getattr(o, fn)()
    return dict(o)


def run_actor(client, actor_id, run_input, use_cache=True, cache_dir="apify_cache",
              max_items=None, max_total_charge_usd=None):
    """Run an Apify actor and return (DataFrame, meta). Cached on disk by actor + input.

    max_items and max_total_charge_usd are hard caps that Apify enforces on its side.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5(json.dumps([actor_id, run_input], sort_keys=True).encode()).hexdigest()[:12]
    path = os.path.join(cache_dir, f"{key}.json")

    if use_cache and os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        print(f"cache hit: {path}")
        return pd.DataFrame(d["items"]), d["meta"]

    t0 = time.time()
    caps = {}
    if max_items is not None:
        caps["max_items"] = max_items
    if max_total_charge_usd is not None:
        caps["max_total_charge_usd"] = Decimal(str(max_total_charge_usd))
    run = _to_dict(client.actor(actor_id).call(run_input=run_input, **caps))
    dataset_id = run.get("defaultDatasetId") or run.get("default_dataset_id")
    items = list(client.dataset(dataset_id).iterate_items())

    meta = {
        "actor": actor_id,
        "input": run_input,
        "dataset_id": dataset_id,
        "run_id": run.get("id"),
        "seconds": round(time.time() - t0, 1),
        "usage_usd": run.get("usageTotalUsd"),
        "n_items": len(items),
        "note": "For pay-per-result actors the real charge is shown on the run page in Apify Console.",
    }
    with open(path, "w") as f:
        json.dump({"items": items, "meta": meta}, f, default=str)
    print(f"ran {actor_id}: {len(items)} items in {meta['seconds']}s (usage_usd={meta['usage_usd']})")
    return pd.DataFrame(items), meta


def text_of(df, text_cols):
    """Join the given text columns into one lowercase string per row."""
    cols = [c for c in text_cols if c in df.columns]
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    return df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()


def _is_filled(v):
    if v is None:
        return False
    if isinstance(v, float) and pd.isna(v):
        return False
    if isinstance(v, (str, list, dict)) and len(v) == 0:
        return False
    return True


def keyword_hits(df, text_cols, keywords):
    """Boolean Series: does the row mention any of the keywords."""
    t = text_of(df, text_cols)
    if not keywords:
        return pd.Series([True] * len(df), index=df.index)
    pat = "|".join(re.escape(k.lower()) for k in keywords)
    return t.str.contains(pat, regex=True)


def quality_report(df, text_cols, date_col, keywords, needed_fields, id_col="id"):
    """Print the basic quality checks for one result set."""
    print("=" * 70)
    n = len(df)
    print(f"rows: {n}")
    if n == 0:
        print("EMPTY RESULT. Check input, filters, or the actor run summary.")
        return

    if "dataType" in df.columns:
        print("\nrows by dataType:")
        print(df["dataType"].value_counts().to_string())

    if id_col in df.columns:
        dup = df.duplicated(subset=[c for c in ["dataType", id_col] if c in df.columns]).sum()
        print(f"\nduplicate rows: {dup}")

    if date_col in df.columns:
        d = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        print(f"\ndate range ({date_col}): {d.min()}  ->  {d.max()}")
        print(f"missing dates: {d.isna().sum()}")
    else:
        print(f"\n(no '{date_col}' column found)")

    print("\nneeded fields, share of rows that are filled:")
    for f in needed_fields:
        if f in df.columns:
            filled = df[f].apply(_is_filled).mean()
            print(f"  {f:<22} {filled:6.1%}")
        else:
            print(f"  {f:<22} MISSING COLUMN")

    t = text_of(df, text_cols)
    empty = (t.str.strip() == "").mean()
    print(f"\nrows with empty text: {empty:.1%}")
    if keywords:
        hit = keyword_hits(df, text_cols, keywords).mean()
        print(f"rows mentioning any of {keywords}: {hit:.1%}")
        print("(low number means the search is returning loosely related stuff)")


def _fields_for(data_type, needed_fields):
    """Posts and comments have different columns. Swap in the comment names."""
    if data_type != "comment":
        return list(needed_fields)
    out = []
    for f in needed_fields:
        if f == "title":
            continue  # comments have no title
        out.append("commentCreatedAt" if f == "createdAt" else f)
    return out


def reddit_quality_report(df, text_cols, keywords, needed_fields):
    """Reddit checks. Splits by dataType, since posts and comments have different columns."""
    print("#" * 70)
    print(f"total rows: {len(df)}")
    if len(df) == 0:
        print("EMPTY RESULT. Check input, filters, or the actor run summary.")
        return
    if "dataType" not in df.columns:
        print("No dataType column, running the plain report on everything.")
        quality_report(df, text_cols, "createdAt", keywords, needed_fields)
        return

    print(df["dataType"].value_counts().to_string())

    for dt in ["post", "comment"]:
        # drop columns that only belong to the other dataType, so missing ones show as MISSING
        sub = df[df["dataType"] == dt].dropna(axis=1, how="all")
        if len(sub) == 0:
            continue
        print(f"\n\n>>> {dt.upper()}S")
        date_col = "commentCreatedAt" if dt == "comment" else "createdAt"
        quality_report(sub, text_cols, date_col, keywords, _fields_for(dt, needed_fields))
        _reddit_extras(sub, text_cols, keywords, dt)

    others = df[~df["dataType"].isin(["post", "comment"])]
    if len(others):
        print(f"\n(also {len(others)} other rows: {others['dataType'].value_counts().to_dict()}, not checked)")

    pc = df[df["dataType"].isin(["post", "comment"])]
    if "communityName" in pc.columns:
        print("\n\ntop 15 communities (posts + comments):")
        print(pc["communityName"].value_counts().head(15).to_string())


def _reddit_extras(sub, text_cols, keywords, dt):
    print("\nreddit checks:")
    n = len(sub)

    if dt == "post":
        if "removalStatus" in sub.columns:
            s = sub["removalStatus"]
            known = s.notna()
            not_active = (s[known].astype(str).str.lower() != "active").mean() if known.any() else float("nan")
            print(f"  posts with removalStatus not active: {not_active:.1%} (no value on {(~known).sum()} rows)")
        else:
            print("  removalStatus: MISSING COLUMN")

    if "body" in sub.columns:
        b = sub["body"].fillna("").astype(str).str.strip()
        print(f"  empty body: {(b == '').mean():.1%}")
        print(f"  [removed] or [deleted] body: {b.str.lower().isin(REMOVED_BODIES).mean():.1%}")
    else:
        print("  body: MISSING COLUMN")

    if "sentimentLabel" in sub.columns:
        unc = (sub["sentimentLabel"].astype(str).str.lower() == "uncertain").mean()
        print(f"  sentimentLabel == uncertain (bot or mod boilerplate): {unc:.1%}")
    else:
        print("  sentimentLabel: not there (needs paid plan + aiAnalysis)")

    if "relevanceScore" in sub.columns:
        r = pd.to_numeric(sub["relevanceScore"], errors="coerce")
        print(f"  relevanceScore: min {r.min():.2f}, median {r.median():.2f}, max {r.max():.2f}, missing {r.isna().sum()}")
        bins = pd.cut(r, [0, 0.25, 0.5, 0.75, 1.0], include_lowest=True).value_counts(sort=False)
        print("    " + bins.to_string().replace("\n", "\n    "))
        print(f"  relevanceScore below 0.5: {(r < 0.5).sum() / n:.1%}")
    else:
        print("  relevanceScore: not there (needs paid plan + aiAnalysis)")

    if "searchTerm" in sub.columns:
        print("\n  by searchTerm (which keywords bring noise):")
        print("    " + search_term_table(sub, text_cols, keywords).to_string().replace("\n", "\n    "))


def search_term_table(df, text_cols, keywords):
    """One row per searchTerm: count, keyword hit share, and AI scores when present."""
    tmp = pd.DataFrame({"searchTerm": df["searchTerm"].fillna("(none)"),
                        "kw_hit": keyword_hits(df, text_cols, keywords)})
    if "relevanceScore" in df.columns:
        tmp["low_relevance"] = pd.to_numeric(df["relevanceScore"], errors="coerce") < 0.5
    if "sentimentLabel" in df.columns:
        tmp["uncertain"] = df["sentimentLabel"].astype(str).str.lower() == "uncertain"
    g = tmp.groupby("searchTerm")
    out = g.size().to_frame("rows")
    for c in tmp.columns.drop("searchTerm"):
        out[c] = g[c].mean().round(2)
    return out.sort_values("rows", ascending=False)


def flatten_ai_flags(df):
    """Turn the aiFlags dict column into aiFlags.<key> columns. Returns a new DataFrame."""
    if "aiFlags" not in df.columns:
        return df
    flags = df["aiFlags"].apply(lambda v: v if isinstance(v, dict) else {})
    flat = pd.json_normalize(flags.tolist()).add_prefix("aiFlags.")
    flat.index = df.index
    return pd.concat([df.drop(columns=["aiFlags"]), flat], axis=1)


def top_values(df, col, n=25):
    """Count items in a list column. Handles lists of strings or lists of dicts."""
    c = Counter()
    if col not in df.columns:
        return pd.DataFrame(columns=[col, "count"])
    for v in df[col].dropna():
        if not isinstance(v, list):
            v = [v]
        for x in v:
            if isinstance(x, dict):
                x = x.get("name") or x.get("text") or x.get("entity") or json.dumps(x, sort_keys=True)
            c[str(x).strip().lower()] += 1
    return pd.DataFrame(c.most_common(n), columns=[col, "count"])


def show_sample(df, cols, n=20, seed=1):
    """Random rows to read by hand. This is the most important check."""
    cols = [c for c in cols if c in df.columns]
    return df.sample(min(n, len(df)), random_state=seed)[cols].reset_index(drop=True)
