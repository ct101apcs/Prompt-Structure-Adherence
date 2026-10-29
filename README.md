# How representations affect rule adherence in Vison-language model: A mech-interp walkthrough

A vision-language model (Qwen3-VL-8B-Instruct) reads a traffic-sign photo and applies a decision tree given in the
prompt. Four ways of writing the same tree reach the same accuracy, but lettered steps ("Step A … go to Step E")
lags behind the others on rule adherence. The notebook traces this into the model's internals.
- Most off-tree answers are shape jumps: a late retrieval of a content name from the other shape's rule lines.
- A single attention head (L28H0) flags these answers in every format.
- The outline format anchors the right list through its parent line; the step format's pointer route goes unused.

## Quick start

```bash
git lfs install            # the residual streams (data/runs/instruct-h3-interp/hidden, ~410 MB) are stored with LFS
pip install -r requirements.txt
jupyter lab interp_walkthrough.ipynb   # run from the repo root
```

The notebook needs a CPU only and runs in about a minute. Every model output is precomputed.

## Layout

| path | content |
|---|---|
| `interp_walkthrough.ipynb` | the walkthrough, executed |
| `lib/` | helpers: answer tables and adherence (`answers.py`), knockout metrics (`knockout.py`), probes (`probes.py`) |
| `data/spec/` | the 22 classes and their three attributes (shape → content → item) |
| `data/rulesets/h3L0…4/` | five random label assignments (`labels.json`), photo lists (`items_image_o*.csv`), the prompts (`prompts/image/<format>_o<order>.txt`) |
| `data/runs/instruct-h3L<k>-image/` | the model's logged answers, per format and rule order |
| `data/runs/instruct-h3-interp/` | interpretability results: slot read-outs and residual streams (`hidden/`), attention knockouts, head attribution, the detector scan over all 56,164 answers (L28H0 only), the route knockout |
| `data/photos/` | the GTSRB photos the notebook shows |

Formats: `steps` = P1 lettered steps, `outline` = P2 indented outline, `table` = P3 decision table, `flat` = P4 flat
rules. Orders `o1`/`o2` list the rules in two different orders.

## Provenance

- **Model:** Qwen3-VL-8B-Instruct, greedy decoding, bf16. Each photo is resized to 448×448 and placed after the
  rules.
- **Photos:** German Traffic Sign Recognition Benchmark (Stallkamp et al., 2012), filtered to 14,041 legible photos of
  22 classes. The photos themselves are not redistributed beyond the few shown in the notebook.
- **Interpretability runs:** batch 1, teacher-forced on each logged answer, plain `transformers` hooks. The GPU
  pipeline that produced `data/runs/instruct-h3-interp/` is not part of this repo.