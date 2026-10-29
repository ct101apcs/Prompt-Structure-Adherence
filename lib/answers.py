"""Class spec and answer tables of the image runs (the model-free part of the study's slots.py)."""
import json
from pathlib import Path

import pandas as pd

H = Path(__file__).resolve().parent.parent / "data"
classes = {int(k): v for k, v in json.loads((H / "spec/classes.json").read_text())["classes"].items()}
IMAGE_ROOT = H / "photos"  # GTSRB photos; only the ones the notebook shows are included
leaf_of = {(v["shape"], v["content"], v["item"]): c for c, v in classes.items()}
item_leaf = {v["item"]: c for c, v in classes.items()}
CONTENTS = sorted({v["content"] for v in classes.values()})
SHAPE_CONTENTS = {s: sorted({v["content"] for v in classes.values() if v["shape"] == s}) for s in ("round", "triangle")}
ITEMS = [classes[c]["item"] for c in sorted(classes)]


def load_answers(fmt):
    rows = []
    for s in range(5):
        labels = {int(k): v for k, v in json.loads((H / f"rulesets/h3L{s}/labels.json").read_text())["labels"].items()}
        items = pd.concat([pd.read_csv(H / f"rulesets/h3L{s}/items_image_o{o}.csv") for o in "12"])
        items = items.drop_duplicates("Path").set_index("Path")
        for o in "12":
            for line in (H / f"runs/instruct-h3L{s}-image/{fmt}_o{o}.jsonl").read_text().splitlines():
                r = json.loads(line)
                a = {k.strip().lower(): str(v).strip() for k, v in (r.get("attrs") or {}).items()}
                rep = (a.get("shape"), a.get("content"), a.get("item"))
                rc = leaf_of.get(rep)
                rows.append({"fmt": fmt, "seed": s, "order": f"o{o}", "path": r["path"], "raw": r["raw"],
                             "cls": int(items.at[r["path"], "ClassId"]), "pred": r["pred"], "label": r["label"],
                             "ok": r["pred"] == r["label"], "adh": rc is not None and labels[rc] == r["pred"],
                             "shape": rep[0], "content": rep[1], "item": rep[2]})
    return pd.DataFrame(rows)
