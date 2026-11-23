"""Manual check of the Task 2 labels: build a local HTML viewer, then apply what was changed.

    python experiments/task2_aspect/labeling/review_viewer.py build
        -> data/interim/task2_labeling/review_viewer.html (open it in a browser; data inside)
    python experiments/task2_aspect/labeling/review_viewer.py apply <exported.json>
        -> data/processed/task2_labels_checked.jsonl + agreement numbers for the paper

The viewer shows one review at a time (full text + every sentence with its labels), test
games first. Progress is kept in the browser (localStorage) and exported as JSON:
{"reviewed": [review_id, ...], "labels": {sentence_id: [[aspect, polarity], ...]}}, where
"labels" holds only the sentences whose labels were changed.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_agents import Run  # noqa: E402

from steam_review_translator.core.config import load_yaml, project_root  # noqa: E402
from steam_review_translator.core.schemas import Aspect, Polarity  # noqa: E402

LABELS = "data/processed/task2_labels.jsonl"
OUT_HTML = "data/interim/task2_labeling/review_viewer.html"
OUT_CHECKED = "data/processed/task2_labels_checked.jsonl"
SPLIT_ORDER = {"test": 0, "val": 1, "train": 2}


def _rows() -> list[dict]:
    path = project_root() / LABELS
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def build() -> None:
    rows = {r["sentence_id"]: r for r in _rows()}
    run = Run(load_yaml("configs/task2/labeling.yaml"))
    reviews = []
    for b in run.batches():
        for rv in b.reviews:
            reviews.append(
                {
                    "id": rv.review_id,
                    "game": b.game_name,
                    "split": b.split,
                    "rec": rv.recommended,
                    "text": rv.text,
                    "s": [
                        [
                            s.sentence_id,
                            s.sentence,
                            [[x["aspect"], x["polarity"]] for x in rows[s.sentence_id]["labels"]],
                        ]
                        for s in rv.sentences
                    ],  # fmt: skip
                }
            )
    reviews.sort(key=lambda r: (SPLIT_ORDER[r["split"]], r["game"], r["id"]))
    data = {
        "aspects": [a.value for a in Aspect],
        "polarities": [p.value for p in Polarity],
        "reviews": reviews,
    }
    html = TEMPLATE.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False))
    out = project_root() / OUT_HTML
    out.write_text(html, encoding="utf-8")
    n = sum(len(r["s"]) for r in reviews)
    print(f"{len(reviews)} reviews, {n} sentences -> {out}")


def apply(export_path: str) -> None:
    exp = json.loads(Path(export_path).read_text(encoding="utf-8"))
    reviewed = {int(x) for x in exp["reviewed"]}
    changed = {k: [{"aspect": a, "polarity": p} for a, p in v] for k, v in exp["labels"].items()}
    rows = _rows()
    out, stats = [], Counter()
    gold_pairs = pred_pairs = both = 0
    for r in rows:
        checked = r["review_id"] in reviewed
        llm = {(x["aspect"], x["polarity"]) for x in r["labels"]}
        human = changed.get(r["sentence_id"])
        new = r | {"llm_labels": r["labels"], "human_checked": checked}
        if human is not None:
            new["labels"] = human
        out.append(new)
        if checked:
            final = {(x["aspect"], x["polarity"]) for x in new["labels"]}
            stats[f"checked_{r['split']}"] += 1
            stats["checked"] += 1
            stats["changed"] += final != llm
            stats[f"changed_{r['split']}"] += final != llm
            gold_pairs += len(final)
            pred_pairs += len(llm)
            both += len(final & llm)
    path = project_root() / OUT_CHECKED
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), "utf-8")
    p = both / pred_pairs if pred_pairs else 0.0
    rc = both / gold_pairs if gold_pairs else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    print(f"checked sentences: {stats['checked']} of {len(rows)} "
          f"(test {stats['checked_test']}, val {stats['checked_val']}, "
          f"train {stats['checked_train']})")  # fmt: skip
    print(f"sentences changed: {stats['changed']} "
          f"({stats['changed'] / max(stats['checked'], 1):.1%}); test {stats['changed_test']}, "
          f"val {stats['changed_val']}, train {stats['changed_train']}")  # fmt: skip
    print(f"LLM vs human (checked sentences): pair P {p:.3f} R {rc:.3f} F1 {f1:.3f}")
    print(f"-> {path}")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Task 2 label check</title>
<style>
:root{--bg:#f7f7f8;--card:#fff;--ink:#1f2328;--mute:#6b7280;--line:#e5e7eb;--pos:#15803d;
--neg:#b91c1c;--neu:#6b7280;--acc:#2563eb;--chg:#fff7ed}
@media (prefers-color-scheme:dark){:root{--bg:#111316;--card:#1b1e23;--ink:#e6e6e6;
--mute:#9aa1ab;--line:#2c3038;--chg:#2a2112}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.45 system-ui,Segoe UI,sans-serif}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);
padding:8px 16px;display:flex;gap:12px;flex-wrap:wrap;align-items:center;z-index:5}
header b{font-size:15px}.bar{flex:1;min-width:160px;height:8px;background:var(--line);
border-radius:4px;overflow:hidden}.bar i{display:block;height:100%;background:var(--acc)}
button,select{font:inherit;padding:4px 10px;border:1px solid var(--line);border-radius:6px;
background:var(--card);color:var(--ink);cursor:pointer}button.primary{background:var(--acc);
color:#fff;border-color:var(--acc)}main{max-width:1000px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.meta{color:var(--mute);font-size:13px;margin-bottom:6px}.full{white-space:pre-wrap;
font-size:13.5px;color:var(--mute);max-height:160px;overflow:auto;border-left:3px solid
var(--line);padding-left:10px;margin:6px 0 12px}table{width:100%;border-collapse:collapse}
td{border-top:1px solid var(--line);padding:7px 6px;vertical-align:top}td.n{width:28px;
color:var(--mute);font-size:12px}td.l{width:42%}tr.chg{background:var(--chg)}
.chip{display:inline-block;margin:2px 4px 2px 0;padding:1px 8px;border-radius:12px;
border:1px solid currentColor;font-size:12.5px;cursor:pointer;user-select:none}
.positive{color:var(--pos)}.negative{color:var(--neg)}.neutral{color:var(--neu)}
.add{font-size:12.5px;padding:1px 6px}.empty{color:var(--mute);font-size:12.5px}
.done{color:var(--pos);font-weight:600}.help{color:var(--mute);font-size:12.5px;margin-top:10px}
</style></head><body>
<header>
 <b>Task 2 label check</b>
 <span id="pos"></span><div class="bar"><i id="prog"></i></div><span id="stat"></span>
 <select id="split"><option value="">all splits</option><option>test</option><option>val</option>
 <option>train</option></select>
 <label><input type="checkbox" id="todo"> unchecked only</label>
 <button id="prev">&larr;</button><button id="next">&rarr;</button>
 <button id="export">Export JSON</button>
</header>
<main><div class="card" id="card"></div>
<div class="help">Click a label to cycle positive &rarr; negative &rarr; neutral &rarr; remove.
"+" adds an aspect. <b>Enter</b> = mark this review checked and go to the next one;
<b>&larr; / &rarr;</b> = move without marking. Progress is saved in this browser; press
<b>Export JSON</b> regularly and at the end.</div></main>
<script>
const D = /*DATA*/null;
const KEY = "task2-label-check-v1";
let st = {reviewed: {}, labels: {}, i: 0};
try { const s = JSON.parse(localStorage.getItem(KEY)); if (s) st = s; } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} };
const POL = D.polarities;
const view = () => {
  const sp = document.getElementById("split").value, todo = document.getElementById("todo").checked;
  return D.reviews.map((r, i) => i).filter(i => (!sp || D.reviews[i].split === sp) &&
    (!todo || !st.reviewed[D.reviews[i].id] || i === st.i));
};
const labelsOf = s => st.labels[s[0]] || s[2];
const same = (a, b) => JSON.stringify([...a].sort()) === JSON.stringify([...b].sort());
function setLabels(s, labs) {
  if (same(labs, s[2])) delete st.labels[s[0]]; else st.labels[s[0]] = labs;
  save(); render();
}
function render() {
  const r = D.reviews[st.i], card = document.getElementById("card");
  const nRev = Object.keys(st.reviewed).length, nAll = D.reviews.length;
  const v = view(), k = v.indexOf(st.i);
  document.getElementById("pos").textContent = `review ${k + 1} / ${v.length}`;
  document.getElementById("prog").style.width = (100 * nRev / nAll) + "%";
  document.getElementById("stat").textContent =
    `checked ${nRev}/${nAll} reviews, changed ${Object.keys(st.labels).length} sentences`;
  card.innerHTML = "";
  const meta = document.createElement("div"); meta.className = "meta";
  meta.innerHTML = `<b>${r.game}</b> · ${r.split} · review ${r.id} · ` +
    `${r.rec ? "recommended" : "not recommended"}` +
    (st.reviewed[r.id] ? ' · <span class="done">checked</span>' : "");
  const full = document.createElement("div"); full.className = "full"; full.textContent = r.text;
  const tb = document.createElement("table");
  r.s.forEach((s, j) => {
    const tr = tb.insertRow(); const labs = labelsOf(s);
    if (st.labels[s[0]]) tr.className = "chg";
    tr.insertCell().outerHTML = `<td class="n">${j + 1}</td>`;
    const t = tr.insertCell(); t.textContent = s[1];
    const c = tr.insertCell(); c.className = "l";
    if (!labs.length) c.insertAdjacentHTML("beforeend", '<span class="empty">no aspect </span>');
    labs.forEach(([a, p], m) => {
      const ch = document.createElement("span"); ch.className = "chip " + p;
      ch.textContent = `${a}/${p}`;
      ch.onclick = () => {
        const nx = POL.indexOf(p) + 1, out = labs.map(x => [...x]);
        if (nx >= POL.length) out.splice(m, 1); else out[m][1] = POL[nx];
        setLabels(s, out);
      };
      c.appendChild(ch);
    });
    const sel = document.createElement("select"); sel.className = "add";
    sel.innerHTML = '<option value="">+</option>' + D.aspects
      .filter(a => !labs.some(x => x[0] === a)).map(a => `<option>${a}</option>`).join("");
    sel.onchange = () => {
      if (sel.value) setLabels(s, [...labs.map(x => [...x]), [sel.value, "positive"]]);
    };
    c.appendChild(sel);
  });
  card.append(meta, full, tb);
}
function go(d) {
  const v = view(); let k = v.indexOf(st.i);
  if (k < 0) k = 0; else k = Math.min(Math.max(k + d, 0), v.length - 1);
  if (v.length) st.i = v[k]; save(); render(); window.scrollTo(0, 0);
}
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT") return;
  if (e.key === "Enter") { st.reviewed[D.reviews[st.i].id] = 1; go(1); e.preventDefault(); }
  if (e.key === "ArrowRight") go(1);
  if (e.key === "ArrowLeft") go(-1);
});
document.getElementById("next").onclick = () => go(1);
document.getElementById("prev").onclick = () => go(-1);
document.getElementById("split").onchange = () => go(0);
document.getElementById("todo").onchange = () => go(0);
document.getElementById("export").onclick = () => {
  const blob = new Blob([JSON.stringify({reviewed: Object.keys(st.reviewed).map(Number),
    labels: st.labels}, null, 1)], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-");
  a.download = `task2_label_check_${stamp}.json`;
  a.click();
};
render();
</script></body></html>
"""


def main() -> int:
    if sys.argv[1:2] == ["build"]:
        build()
    elif sys.argv[1:2] == ["apply"] and len(sys.argv) == 3:
        apply(sys.argv[2])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
