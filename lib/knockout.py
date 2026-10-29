"""Knockout metrics: m_* = log-prob margin at the slot, f_* = flip (from the study's knockout_summary.py)."""
import json
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent.parent / "data"
classes = {int(k): v for k, v in json.loads((H / "spec/classes.json").read_text())["classes"].items()}


def lse(v):
    v = np.asarray(v)
    return v.max() + np.log(np.exp(v - v.max()).sum())

def _idx(r):
    return {c: i for i, c in enumerate(r["cands"])}

def m_c(r, row):
    idx, g = _idx(r), classes[r["cls"]]["content"]
    if r["content"] != g:
        return row[idx[g]] - row[idx[r["content"]]]
    return row[idx[g]] - max(v for c, v in zip(r["cands"], row) if c != g)

def f_c(r, row):
    g = classes[r["cls"]]["content"]
    top = r["cands"][int(np.argmax(row))]
    return top == g if r["content"] != g else top != g

def branch(c):
    return [v["item"] for v in classes.values() if v["content"] == c]

def pc(r):
    return r["prefix"].split('"content": "')[1].split('"')[0]

def m_i(r, row):
    idx, g = _idx(r), classes[r["cls"]]["item"]
    b = branch(pc(r))
    if g in b:  # control: the gold item is on the prefixed branch; margin over its siblings
        return row[idx[g]] - max(row[idx[x]] for x in b if x != g)
    return row[idx[g]] - lse([row[idx[x]] for x in b])

def f_i(r, row):
    idx, g = _idx(r), classes[r["cls"]]["item"]
    cand = list(dict.fromkeys([g] + branch(pc(r))))
    top = cand[int(np.argmax([row[idx[x]] for x in cand]))]
    base_top = cand[int(np.argmax([r["lp"][0][idx[x]] for x in cand]))]
    return top != base_top
