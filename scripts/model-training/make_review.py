"""
Render the same five held-out broadcasts for any model, in one shared format.

  python make_review.py --predictions build/pred_V2_w3072_test.jsonl \
      --title "V2_w3072 - greedy" --out review/V2_w3072_greedy.html

The transcript is rendered ONCE with your breaks in the left rail and the
model's in the right. That halves the DOM versus two side-by-side panes, needs
no scroll syncing (nothing to keep in step), and makes alignment exact by
construction rather than approximate.

The five broadcasts are fixed across every model (build/_picks.json) so the
pages are directly comparable — same text, same order, only the right rail
changes.
"""
import argparse
import html
import json
import re

import config as C
import build_dataset as B
import evaluate as E

CSS = """
:root{--ground:#171A21;--ground-2:#1E222B;--line:#2C313D;--paper:#FBFAF8;
 --ink:#1A1D24;--ink-2:#5B6270;--human:#B45309;--human-soft:#F5E3D0;
 --model:#0F766E;--model-soft:#D6EBE8;--dim:#8A92A3}
@media (prefers-color-scheme:light){:root{--ground:#EDEAE4;--ground-2:#E3DFD7;
 --line:#D2CDC3;--dim:#6B7280}}
:root[data-theme="light"]{--ground:#EDEAE4;--ground-2:#E3DFD7;--line:#D2CDC3;--dim:#6B7280}
:root[data-theme="dark"]{--ground:#171A21;--ground-2:#1E222B;--line:#2C313D;--dim:#8A92A3}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--paper);
 font:15px/1.6 ui-monospace,"SF Mono",Menlo,Consolas,monospace;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 80px}
h1{font:600 20px/1.3 ui-monospace,monospace;margin:0 0 4px;letter-spacing:.02em}
.sub{margin:0 0 8px;color:var(--dim);font-size:13px;max-width:74ch}
.sub b{color:var(--human);font-weight:600}.sub i{color:var(--model);font-style:normal;font-weight:600}
.tot{margin:0 0 20px;font-size:12px;color:var(--dim);font-variant-numeric:tabular-nums}
.tot b{color:var(--paper);font-weight:600}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:18px}
.tab{flex:1 1 170px;text-align:left;cursor:pointer;background:var(--ground-2);
 border:1px solid var(--line);border-radius:3px;padding:9px 11px;color:var(--dim);
 font:inherit;font-size:12px;display:grid;gap:2px}
.tab:hover{border-color:var(--dim)}
.tab:focus-visible{outline:2px solid var(--model);outline-offset:2px}
.tab.on{background:var(--paper);border-color:var(--paper);color:var(--ink)}
.t-out{font-weight:700;letter-spacing:.08em;font-size:10px;text-transform:uppercase}
.tab.on .t-out{color:var(--human)}
.t-show{font-size:12px}.t-tag{font-size:11px;opacity:.7}
.panel{display:none}.panel.on{display:block}
.phead{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;
 flex-wrap:wrap;padding:0 0 12px;border-bottom:1px solid var(--line)}
.phead h2{margin:0;font-size:15px;font-weight:600}
.meta{margin:3px 0 0;font-size:12px;color:var(--dim)}
.chips{display:flex;gap:6px}
.chip{font-size:11px;padding:3px 8px;border-radius:2px;border:1px solid var(--line);
 color:var(--dim);font-variant-numeric:tabular-nums}
.c-ok{color:var(--model);border-color:var(--model)}
.c-miss{color:var(--human);border-color:var(--human)}
.script{background:var(--paper);color:var(--ink);border-radius:0 0 3px 3px;padding:30px 0}
.script p{max-width:62ch;margin:0 auto 1.05em;padding:0 26px;
 font:16px/1.72 "Iowan Old Style",Georgia,"Times New Roman",serif}
.script p:last-child{margin-bottom:0}
.brk{display:grid;grid-template-columns:1fr minmax(auto,62ch) 1fr;align-items:center;
 gap:10px;margin:26px 0;padding:0 12px}
.rail{font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;white-space:nowrap}
.rail-l{text-align:right}.rail-r{text-align:left}
.tick{display:inline-block;padding:2px 7px;border-radius:2px;font-variant-numeric:tabular-nums}
.tick b{font-weight:700}
.tick-h{background:var(--human-soft);color:var(--human)}
.tick-m{background:var(--model-soft);color:var(--model)}
.rule{position:relative;height:1px;background:var(--ink);opacity:.22}
.note{position:absolute;left:50%;top:-9px;transform:translateX(-50%);background:var(--paper);
 padding:0 10px;font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--ink-2)}
.brk-match .rule{background:var(--model);opacity:.5;height:2px}
.brk-match .note{color:var(--model)}
.brk-missed .rule{background:var(--human);opacity:.45}
.brk-missed .note{color:var(--human)}
.brk-extra .rule{opacity:.18;background-image:repeating-linear-gradient(90deg,
 var(--ink) 0 5px,transparent 5px 10px)}
.legend{display:flex;gap:18px;flex-wrap:wrap;margin:14px 0 0;font-size:11.5px;color:var(--dim)}
.legend span{display:flex;align-items:center;gap:7px}
.sw{width:22px;height:2px;display:inline-block}
.sw-m{background:var(--model)}.sw-h{background:var(--human)}
.sw-e{background-image:repeating-linear-gradient(90deg,var(--dim) 0 5px,transparent 5px 10px);height:1px}
@media (max-width:820px){.brk{grid-template-columns:1fr;gap:5px;padding:0 20px}
 .rail-l,.rail-r{text-align:left}.rule{order:3}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

JS = """
var tabs=document.querySelectorAll('.tab'),panels=document.querySelectorAll('.panel');
for(var i=0;i<tabs.length;i++){tabs[i].addEventListener('click',function(){
 var n=this.dataset.i;
 for(var j=0;j<tabs.length;j++){var on=tabs[j].dataset.i===n;
  tabs[j].classList.toggle('on',on);tabs[j].setAttribute('aria-selected',on?'true':'false');
  panels[j].classList.toggle('on',panels[j].dataset.i===n);}
 window.scrollTo(0,0);});}
"""


def paragraphs(words):
    """Transcript bodies run speaker labels together; split on NAME: to read."""
    t = " ".join(words)
    t = re.sub(r'(?<=[a-z.\)\]"])([A-Z][A-Z .\'-]{2,25}(?:, [A-Z][A-Za-z .]+)?:)',
               r"\n\1", t)
    return [p.strip() for p in t.split("\n") if p.strip()]


def marker(ev):
    e = html.escape
    if ev["kind"] == "match":
        L = f'<span class="tick tick-h">yours <b>{ev["gold"]:,}</b></span>'
        R = f'<span class="tick tick-m">model <b>{ev["model"]:,}</b></span>'
        note = f'agreed &middot; {ev["off"]} word{"" if ev["off"] == 1 else "s"} apart'
    elif ev["kind"] == "missed":
        L = f'<span class="tick tick-h">yours <b>{ev["gold"]:,}</b></span>'
        R, note = "", "model missed this"
    else:
        L = ""
        R = f'<span class="tick tick-m">model <b>{ev["model"]:,}</b></span>'
        note = "model added this"
    return (f'<div class="brk brk-{ev["kind"]}"><div class="rail rail-l">{L}</div>'
            f'<div class="rule"><span class="note">{note}</span></div>'
            f'<div class="rail rail-r">{R}</div></div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    e = html.escape
    tr, ann = B.load_sources()
    ann = B.validate(tr, ann)
    gold = {a["record_id"]: sorted(a["breaks"]) for a in ann}
    meta = {a["record_id"]: a for a in ann}
    preds = {p["record_id"]: sorted(p["pred_breaks"])
             for p in B.load_jsonl(args.predictions)}
    picks = json.load(open(C.BUILD_DIR / "_picks.json"))

    tabs, panels = [], []
    TP = FN = FP = 0
    for i, pk in enumerate(picks):
        rid = pk["rid"]
        a, g, p = meta[rid], gold[rid], preds.get(rid, [])
        words = tr[rid]["body"].split()
        m, fn, fp = E.match(g, p, C.MATCH_TOLERANCE_WORDS)
        TP += len(m); FN += fn; FP += fp

        ev = [{"at": min(gw, pw), "kind": "match", "gold": gw, "model": pw, "off": d}
              for gw, pw, d in m]
        hg = {x[0] for x in m}; hp = {x[1] for x in m}
        ev += [{"at": x, "kind": "missed", "gold": x, "model": None} for x in g if x not in hg]
        ev += [{"at": x, "kind": "extra", "gold": None, "model": x} for x in p if x not in hp]
        ev.sort(key=lambda x: x["at"])

        body, prev = [], 0
        for x in ev:
            body += [f"<p>{e(t)}</p>" for t in paragraphs(words[prev:x["at"]])]
            body.append(marker(x)); prev = x["at"]
        body += [f"<p>{e(t)}</p>" for t in paragraphs(words[prev:])]

        on = " on" if i == 0 else ""
        tabs.append(f'<button class="tab{on}" data-i="{i}" role="tab" '
                    f'aria-selected="{"true" if i == 0 else "false"}">'
                    f'<span class="t-out">{e(a["outlet"])}</span>'
                    f'<span class="t-show">{e(a["show"][:24])}</span>'
                    f'<span class="t-tag">{e(pk["tag"])}</span></button>')
        panels.append(
            f'<section class="panel{on}" data-i="{i}" role="tabpanel">'
            f'<header class="phead"><div><h2>{e(a["show"])}</h2>'
            f'<p class="meta">{e(a["outlet"])} &middot; {e(a["date"])} &middot; '
            f'{a["word_count"]:,} words &middot; you marked {len(g)}, '
            f'model marked {len(p)}</p></div>'
            f'<div class="chips"><span class="chip c-ok">{len(m)} agreed</span>'
            f'<span class="chip c-miss">{fn} missed</span>'
            f'<span class="chip">{fp} added</span></div></header>'
            f'<div class="script">{"".join(body)}</div></section>')

    pr, rc, f1 = E.prf(TP, FN, FP)
    doc = (f"<title>{e(args.title)}</title><style>{CSS}</style><div class=\"wrap\">"
           f"<h1>{e(args.title)}</h1>"
           f'<p class="sub">Five held-out broadcasts. The transcript appears once; '
           f'<b>your breaks</b> sit in the left rail, <i>the model\'s</i> in the '
           f'right. {e(args.note)}</p>'
           f'<p class="tot">across these five &mdash; <b>{TP}</b> agreed, '
           f'<b>{FN}</b> missed, <b>{FP}</b> added &nbsp;&middot;&nbsp; '
           f'P <b>{pr:.3f}</b> &nbsp; R <b>{rc:.3f}</b> &nbsp; F1 <b>{f1:.4f}</b></p>'
           f'<div class="tabs" role="tablist">{"".join(tabs)}</div>{"".join(panels)}'
           f'<div class="legend"><span><i class="sw sw-m"></i>both agreed</span>'
           f'<span><i class="sw sw-h"></i>you marked, model missed</span>'
           f'<span><i class="sw sw-e"></i>model marked, you did not</span>'
           f'<span>match tolerance &plusmn;25 words</span></div>'
           f"</div><script>{JS}</script>")
    open(args.out, "w", encoding="utf-8").write(doc)
    print(f"{args.out}  ({len(doc)/1024:.0f} KB)  "
          f"tp {TP} fn {FN} fp {FP}  P {pr:.3f} R {rc:.3f} F1 {f1:.4f}")


if __name__ == "__main__":
    main()
