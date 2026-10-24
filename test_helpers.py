"""Tests for helpers.py using a small fake Reddit result. No Apify calls.

Run with: python test_helpers.py
"""
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import pandas as pd

import helpers as h


def fake_reddit_df():
    rows = [
        {"dataType": "post", "id": "p1", "title": "Nike vs Adidas for running", "body": "Adidas feels softer",
         "communityName": "running", "createdAt": "2026-09-01T10:00:00Z", "removalStatus": "active",
         "searchTerm": "Nike vs", "sentimentLabel": "positive", "relevanceScore": 0.9,
         "entities": ["Nike", "Adidas"], "aiFlags": {"competitor": True, "question": False}},
        {"dataType": "post", "id": "p2", "title": "Best Nike alternative?", "body": "Looking at Hoka",
         "communityName": "running", "createdAt": "2026-09-05T10:00:00Z", "removalStatus": "active",
         "searchTerm": "Nike alternative", "sentimentLabel": "neutral", "relevanceScore": 0.7,
         "entities": [{"name": "Hoka"}, {"name": "Nike"}], "aiFlags": {"competitor": True}},
        {"dataType": "post", "id": "p3", "title": "gone", "body": "[removed]",
         "communityName": "Sneakers", "createdAt": "2026-09-07T10:00:00Z", "removalStatus": "removed_by_moderator",
         "searchTerm": "Nike vs", "sentimentLabel": "uncertain", "relevanceScore": 0.2,
         "entities": [], "aiFlags": None},
        {"dataType": "comment", "id": "c1", "body": "Nike runs small compared to New Balance",
         "communityName": "Sneakers", "commentCreatedAt": "2026-09-02T10:00:00Z",
         "searchTerm": "Nike compared to", "sentimentLabel": "negative", "relevanceScore": 0.8,
         "entities": ["Nike", "New Balance"]},
        {"dataType": "comment", "id": "c2", "body": "[deleted]",
         "communityName": "running", "commentCreatedAt": "2026-09-03T10:00:00Z",
         "searchTerm": "Nike vs", "sentimentLabel": "neutral", "relevanceScore": 0.4},
        {"dataType": "community", "id": "r1", "communityName": "running"},
    ]
    return pd.DataFrame(rows)


def capture(fn, *a, **kw):
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*a, **kw)
    return buf.getvalue()


class TestHelpers(unittest.TestCase):
    def setUp(self):
        self.df = fake_reddit_df()
        self.args = (["title", "body"], ["nike"], ["title", "body", "entities", "aiFlags", "createdAt"])

    def test_report_splits_by_datatype(self):
        out = capture(h.reddit_quality_report, self.df, *self.args)
        self.assertIn(">>> POSTS", out)
        self.assertIn(">>> COMMENTS", out)
        self.assertIn("1 other rows", out)
        # comments use commentCreatedAt, and have no title or aiFlags
        comments = out.split(">>> COMMENTS")[1]
        self.assertIn("date range (commentCreatedAt)", comments)
        self.assertIn("aiFlags                MISSING COLUMN", comments)
        self.assertNotIn("title  ", comments.split("reddit checks")[0])

    def test_removed_and_uncertain_shares(self):
        out = capture(h.reddit_quality_report, self.df, *self.args)
        posts = out.split(">>> POSTS")[1].split(">>> COMMENTS")[0]
        self.assertIn("removalStatus not active: 33.3%", posts)
        self.assertIn("[removed] or [deleted] body: 33.3%", posts)
        self.assertIn("uncertain (bot or mod boilerplate): 33.3%", posts)
        self.assertIn("relevanceScore below 0.5: 33.3%", posts)
        comments = out.split(">>> COMMENTS")[1]
        self.assertIn("[removed] or [deleted] body: 50.0%", comments)

    def test_missing_ai_columns_do_not_crash(self):
        df = self.df.drop(columns=["sentimentLabel", "relevanceScore", "entities", "aiFlags", "removalStatus"])
        out = capture(h.reddit_quality_report, df, *self.args)
        self.assertIn("sentimentLabel: not there", out)
        self.assertIn("relevanceScore: not there", out)
        self.assertIn("removalStatus: MISSING COLUMN", out)
        self.assertIs(h.flatten_ai_flags(df), df)

    def test_empty_and_no_datatype(self):
        self.assertIn("EMPTY RESULT", capture(h.reddit_quality_report, pd.DataFrame(), *self.args))
        out = capture(h.reddit_quality_report, self.df.drop(columns=["dataType"]), *self.args)
        self.assertIn("No dataType column", out)

    def test_search_term_table(self):
        posts = self.df[self.df["dataType"] == "post"]
        tbl = h.search_term_table(posts, ["title", "body"], ["nike"])
        self.assertEqual(tbl.loc["Nike vs", "rows"], 2)
        self.assertEqual(tbl.loc["Nike vs", "kw_hit"], 0.5)
        self.assertEqual(tbl.loc["Nike vs", "uncertain"], 0.5)

    def test_flatten_ai_flags(self):
        flat = h.flatten_ai_flags(self.df)
        self.assertIn("aiFlags.competitor", flat.columns)
        self.assertNotIn("aiFlags", flat.columns)
        self.assertEqual(len(flat), len(self.df))
        self.assertEqual(flat["aiFlags.competitor"].fillna(False).astype(bool).sum(), 2)

    def test_top_values_handles_str_and_dict(self):
        tv = h.top_values(self.df, "entities").set_index("entities")["count"]
        self.assertEqual(tv["nike"], 3)
        self.assertEqual(tv["hoka"], 1)
        self.assertEqual(len(h.top_values(self.df, "nope")), 0)

    def test_run_actor_uses_cache(self):
        with tempfile.TemporaryDirectory() as d:
            inp = {"searchTerms": ["x"], "maxPostsCount": 1}
            key = h.hashlib.md5(json.dumps(["a/b", inp], sort_keys=True).encode()).hexdigest()[:12]
            with open(os.path.join(d, f"{key}.json"), "w") as f:
                json.dump({"items": [{"id": 1}], "meta": {"n_items": 1}}, f)
            buf = io.StringIO()
            with redirect_stdout(buf):
                df, meta = h.run_actor(client=None, actor_id="a/b", run_input=inp, cache_dir=d)
            self.assertEqual(len(df), 1)
            self.assertIn("cache hit", buf.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
