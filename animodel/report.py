"""
report.py — Pěkná HTML prezentace modelu vkusu a doporučení.

Vše je generováno datově z TasteModel / seznamu Recommendation. Žádné externí
JS frameworky; jen vložené CSS + drobné inline SVG grafy, takže výsledné .html
je samostatný soubor, který otevřeš dvojklikem.

Estetika: tmavý „redakční datový žurnál" — serifový display font (Fraunces),
klidná paleta s jedním teplým akcentem, sloupce/bary kreslené čistě v CSS/SVG.
"""
from __future__ import annotations

import html
import re
import datetime as _dt
from collections import defaultdict

# Práh pro štítek lehké/náročné u nálad -- zobrazovací konvence, ne model.
INTENSITY_BADGE = 0.20

ACCENT = "#e8a33d"
ACCENT2 = "#6db0a6"
NEG = "#d2654f"
BG = "#13110f"
PANEL = "#1c1916"
INK = "#ece4d6"
MUT = "#9a8f7e"

_FONTS = ("https://fonts.googleapis.com/css2?"
          "family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,500"
          "&family=Spline+Sans:wght@400;500;600"
          "&family=Spline+Sans+Mono:wght@400;500&display=swap")


def _esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def _bar(value: float, vmax: float, color: str, width_px: int = 220) -> str:
    """Horizontální bar s nulou uprostřed (pro kladné/záporné efekty)."""
    frac = max(-1.0, min(1.0, value / vmax)) if vmax else 0.0
    half = width_px / 2
    w = abs(frac) * half
    if frac >= 0:
        left, bw = half, w
    else:
        left, bw = half - w, w
    return (f'<span class="barwrap" style="width:{width_px}px">'
            f'<span class="bar0"></span>'
            f'<span class="bar" style="left:{left:.1f}px;width:{bw:.1f}px;background:{color}"></span>'
            f'</span>')


# ── HEAD / shell ─────────────────────────────────────────────────────────────

def _head(title: str) -> str:
    return f"""<!doctype html><html lang="cs"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{_FONTS}" rel="stylesheet">
<style>
:root{{--bg:{BG};--panel:{PANEL};--ink:{INK};--mut:{MUT};
--acc:{ACCENT};--acc2:{ACCENT2};--neg:{NEG};}}
*{{box-sizing:border-box}}
body{{margin:0;background:
  radial-gradient(1200px 600px at 80% -10%,rgba(232,163,61,.08),transparent 60%),
  radial-gradient(900px 500px at -10% 20%,rgba(109,176,166,.06),transparent 55%),
  var(--bg);
  color:var(--ink);font-family:'Spline Sans',system-ui,sans-serif;
  font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1040px;margin:0 auto;padding:64px 28px 120px}}
h1{{font-family:'Fraunces',serif;font-weight:600;font-size:clamp(34px,6vw,58px);
  line-height:1.04;letter-spacing:-.02em;margin:0 0 6px}}
h1 em{{font-style:italic;color:var(--acc)}}
h2{{font-family:'Fraunces',serif;font-weight:600;font-size:27px;letter-spacing:-.01em;
  margin:64px 0 4px;padding-top:22px;border-top:1px solid rgba(236,228,214,.1)}}
h3{{font-family:'Spline Sans',sans-serif;font-weight:600;font-size:15px;
  text-transform:uppercase;letter-spacing:.13em;color:var(--mut);margin:30px 0 12px}}
.lead{{color:var(--mut);font-size:17px;max-width:62ch;margin:0 0 8px}}
.kicker{{font-family:'Spline Sans Mono',monospace;font-size:12px;letter-spacing:.25em;
  text-transform:uppercase;color:var(--acc);margin:0 0 18px}}
.note{{color:var(--mut);font-size:13.5px;max-width:70ch}}
.grid{{display:grid;gap:14px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}}
.stat{{background:var(--panel);border:1px solid rgba(236,228,214,.08);border-radius:14px;
  padding:16px 18px}}
.stat .v{{font-family:'Fraunces',serif;font-size:34px;font-weight:600;line-height:1}}
.stat .l{{color:var(--mut);font-size:12.5px;margin-top:6px;letter-spacing:.04em}}
.panel{{background:var(--panel);border:1px solid rgba(236,228,214,.08);border-radius:16px;
  padding:6px 20px;margin:14px 0}}
table{{width:100%;border-collapse:collapse;font-size:14px}}
th{{text-align:left;color:var(--mut);font-weight:500;font-size:11.5px;letter-spacing:.1em;
  text-transform:uppercase;padding:12px 8px;border-bottom:1px solid rgba(236,228,214,.1)}}
td{{padding:9px 8px;border-bottom:1px solid rgba(236,228,214,.05);vertical-align:middle}}
tr:last-child td{{border-bottom:none}}
.mono{{font-family:'Spline Sans Mono',monospace}}
.tag{{display:inline-block;font-size:11px;padding:2px 9px;border-radius:999px;
  background:rgba(236,228,214,.07);color:var(--mut);margin:2px 4px 2px 0;
  font-family:'Spline Sans Mono',monospace;letter-spacing:.02em}}
.tag.cat-genre{{color:#e8c98a;background:rgba(232,163,61,.12)}}
.tag.cat-studio{{color:var(--acc2);background:rgba(109,176,166,.12)}}
.pos{{color:var(--acc)}} .neg{{color:var(--neg)}}
.barwrap{{position:relative;display:inline-block;height:10px;vertical-align:middle;
  background:rgba(236,228,214,.05);border-radius:6px}}
.barwrap .bar0{{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:rgba(236,228,214,.18)}}
.barwrap .bar{{position:absolute;top:0;height:10px;border-radius:6px}}
.cl{{background:var(--panel);border:1px solid rgba(236,228,214,.08);border-radius:16px;
  padding:20px 22px;margin:12px 0;position:relative;overflow:hidden}}
.cl .name{{font-family:'Fraunces',serif;font-size:22px;font-weight:600}}
.cl .meta{{color:var(--mut);font-size:13px;margin:2px 0 12px;font-family:'Spline Sans Mono',monospace}}
.pill{{display:inline-block;font-size:11px;padding:3px 11px;border-radius:999px;
  font-family:'Spline Sans Mono',monospace;letter-spacing:.03em}}
.heavy{{background:rgba(210,101,79,.16);color:#e89b87}}
.light{{background:rgba(109,176,166,.16);color:var(--acc2)}}
.mix{{background:rgba(236,228,214,.08);color:var(--mut)}}
.rec{{background:var(--panel);border:1px solid rgba(236,228,214,.08);border-radius:16px;
  padding:22px 24px;margin:14px 0;position:relative}}
.rec .rank{{position:absolute;top:18px;right:22px;font-family:'Fraunces',serif;
  font-size:40px;font-weight:600;color:rgba(236,228,214,.13);line-height:1}}
.rec .t{{font-family:'Fraunces',serif;font-size:23px;font-weight:600;line-height:1.15;
  max-width:80%}}
.rec .ten{{color:var(--mut);font-size:14px;font-style:italic}}
.rec .scores{{display:flex;gap:26px;margin:14px 0;flex-wrap:wrap}}
.rec .scores .s .n{{font-family:'Fraunces',serif;font-size:26px;font-weight:600}}
.rec .scores .s .k{{color:var(--mut);font-size:11px;letter-spacing:.08em;text-transform:uppercase}}
.rec .why{{font-size:14px;color:#cfc7b8;margin:8px 0}}
.rec .syn{{font-size:13.5px;color:var(--mut);margin:10px 0 0;max-width:74ch}}
.flag{{display:inline-block;font-size:11px;padding:3px 10px;border-radius:6px;
  background:rgba(232,163,61,.16);color:var(--acc);font-family:'Spline Sans Mono',monospace;
  letter-spacing:.05em;margin-left:8px;vertical-align:middle}}
.flag.seen{{background:rgba(236,228,214,.10);color:var(--mut)}}
.src{{font-family:'Spline Sans Mono',monospace;font-size:11px;color:var(--mut)}}
footer{{margin-top:80px;padding-top:20px;border-top:1px solid rgba(236,228,214,.1);
  color:var(--mut);font-size:12.5px;font-family:'Spline Sans Mono',monospace}}
a{{color:var(--acc)}}
.spoiltoggle{{position:fixed;top:14px;right:18px;z-index:9;cursor:pointer;
  font-family:'Spline Sans Mono',monospace;font-size:12px;color:var(--mut);
  background:var(--panel);border:1px solid rgba(236,228,214,.14);
  border-radius:999px;padding:6px 14px;user-select:none}}
.spoiltoggle input{{vertical-align:middle;margin-right:6px;accent-color:{ACCENT}}}
body.nospoil .spoiler-item{{display:none}}
</style></head><body>
<label class="spoiltoggle" title="AniList spoiler-flagged tagy (Tragedy, Tearjerker, …) — odškrtni pro skrytí">
<input type="checkbox" checked
 onchange="document.body.classList.toggle('nospoil',!this.checked)">spoiler tagy</label>
<div class="wrap">"""


def _foot(extra: str = "") -> str:
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    return (f'<footer>animodel · vygenerováno {ts}{(" · " + extra) if extra else ""}'
            f'</footer></div></body></html>')


# ── Graf rozdělení známek (inline SVG) ───────────────────────────────────────

def _score_hist_svg(dist: dict[int, int]) -> str:
    scores = list(range(1, 11))
    vals = [dist.get(s, 0) for s in scores]
    mx = max(vals) if vals else 1
    W, H, pad = 760, 200, 28
    bw = (W - 2 * pad) / 10
    bars = []
    for i, (s, v) in enumerate(zip(scores, vals)):
        bh = (v / mx) * (H - 2 * pad - 18) if mx else 0
        x = pad + i * bw + bw * 0.16
        y = H - pad - bh
        col = ACCENT if s >= 9 else (ACCENT2 if s >= 7 else MUT)
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw*0.68:.1f}" height="{bh:.1f}" '
            f'rx="4" fill="{col}" opacity="{0.95 if v else 0.25}"/>'
            f'<text x="{x+bw*0.34:.1f}" y="{H-pad+15:.0f}" fill="{MUT}" font-size="12" '
            f'text-anchor="middle" font-family="Spline Sans Mono">{s}</text>'
            + (f'<text x="{x+bw*0.34:.1f}" y="{y-6:.1f}" fill="{INK}" font-size="12" '
               f'text-anchor="middle" font-family="Spline Sans Mono">{v}</text>' if v else ""))
    return (f'<svg viewBox="0 0 {W} {H}" width="100%" '
            f'style="max-width:760px;display:block;margin:6px 0 4px">{"".join(bars)}</svg>')


# ── MODEL report ─────────────────────────────────────────────────────────────

def render_model_html(model, userinfo: dict, stats: dict, out_path: str) -> str:
    u = _esc(userinfo.get("user_name", "uživatel"))
    parts = [_head(f"Model vkusu — {u}")]
    parts.append(f'<p class="kicker">animodel · profil anime vkusu</p>')
    parts.append(f'<h1>Co ve skutečnosti<br><em>sleduješ rád</em></h1>')
    parts.append(f'<p class="lead">Datový rozbor {stats.get("n_rated",0)} ohodnocených '
                 f'titulů uživatele <b>{u}</b>. Model neměří, co hodnotíš vysoko v absolutních '
                 f'číslech — to děláš skoro u všeho — ale o kolik se odchyluješ od komunity '
                 f'a které atributy ten rozdíl táhnou.</p>')

    # stat karty
    parts.append('<div class="cards">')
    cards = [
        (f'{stats.get("n_rated",0)}', "ohodnoceno"),
        (f'{model.u_mean:.2f}', "tvůj průměr"),
        (f'{model.c_mean:.2f}', "průměr komunity"),
        (f'+{model.beta:.2f}', "sklon ke komunitě β"),
        (f'{len(model.clusters)}', "nálady / módy"),
        (f'{stats.get("n_ptw",0)}', "plan-to-watch"),
    ]
    for v, l in cards:
        parts.append(f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div></div>')
    parts.append('</div>')

    # rozdělení známek
    parts.append('<h2>Rozdělení tvých známek</h2>')
    parts.append('<p class="note">Skoro nic pod 7 — to je ten náročný předvýběr. '
                 'Klasická regrese na surových známkách proto skoro nemá co vysvětlovat; '
                 'proto model cílí na <i>odchylku</i>, ne na známku samotnou.</p>')
    parts.append('<div class="panel">' + _score_hist_svg(stats.get("dist", {})) + '</div>')

    # afinitní efekty
    parts.append('<h2>Které atributy táhnou tvůj vkus</h2>')
    parts.append('<p class="note">„Efekt" = o kolik atribut posouvá tvou odchylku od baseline '
                 '(po smrštění malých vzorků k nule). „Δ komunita" = o kolik výš než komunita '
                 'hodnotíš tituly s tímto atributem. Bar je škálovaný na největší efekt.</p>')

    pos = model.top_effects(n=16, sign=1)
    neg = model.top_effects(n=10, sign=-1)
    vmax = max([abs(e.effect) for e in (pos + neg)] + [0.01])

    def _eff_table(items, head):
        rows = [f'<h3>{head}</h3><div class="panel"><table>'
                '<tr><th>atribut</th><th>kat.</th><th>efekt</th>'
                '<th>Δ komunita</th><th>n</th><th></th></tr>']
        for e in items:
            cls = "pos" if e.effect >= 0 else "neg"
            col = ACCENT if e.effect >= 0 else NEG
            spoil = ' class="spoiler-item"' if getattr(e, "spoiler", False) else ""
            rows.append(
                f'<tr{spoil}><td>{_esc(e.label)}</td>'
                f'<td><span class="tag cat-{_esc(e.category)}">{_esc(e.category)}</span></td>'
                f'<td class="mono {cls}">{e.effect:+.2f}</td>'
                f'<td class="mono">{e.distinct:+.2f}</td>'
                f'<td class="mono" style="color:{MUT}">{e.n_eff:.0f}</td>'
                f'<td>{_bar(e.effect, vmax, col)}</td></tr>')
        rows.append('</table></div>')
        return "".join(rows)

    parts.append(_eff_table(pos, "Co tě táhne nahoru"))
    parts.append(_eff_table(neg, "Co tě táhne dolů"))

    # interakce (synergie dvojic/trojic) -- obě znaménka: model je obě počítá
    # i používá v predikci/doporučeních, tak ať jsou obě i vidět
    triples = getattr(model, "triples", [])
    if model.interactions or triples:
        parts.append('<h2>Synergie atributů</h2>')
        parts.append('<p class="note">Kombinace, kde se tvá afinita liší od prostého součtu '
                     'jednotlivých efektů (lift, po smrštění malých vzorků). Kladné = '
                     '„sladké tečky", záporné = kombinace, které ti nesedí. Obojí vstupuje '
                     'do predikce i řazení doporučení.</p>')

        def _int_table(items, head):
            rows = [f'<h3>{head}</h3><div class="panel"><table>'
                    '<tr><th>kombinace</th><th>lift</th><th>n</th></tr>']
            for it in items:
                cls = "pos" if it.lift >= 0 else "neg"
                spoil = ' class="spoiler-item"' if getattr(it, "spoiler", False) else ""
                rows.append(f'<tr{spoil}><td>{_esc(it.label)}</td>'
                            f'<td class="mono {cls}">{it.lift:+.2f}</td>'
                            f'<td class="mono" style="color:{MUT}">{it.n:.0f}</td></tr>')
            rows.append('</table></div>')
            return "".join(rows)

        pos_int = sorted((it for it in model.interactions if it.lift > 0),
                         key=lambda x: -x.lift)[:12]
        neg_int = sorted((it for it in model.interactions if it.lift < 0),
                         key=lambda x: x.lift)[:12]
        if pos_int:
            parts.append(_int_table(pos_int, "Dělají víc než součet"))
        if neg_int:
            parts.append(_int_table(neg_int, "Nesedí si"))
        if triples:
            # hierarchický lift: zbytek NAD singly a páry (viz taste.py)
            parts.append(_int_table(
                sorted(triples, key=lambda x: -abs(x.lift))[:10],
                "Trojice — jádra nálad (experiment)"))

    # klastry / nálady
    parts.append('<h2>Tvé nálady — mezi čím přepínáš</h2>')
    parts.append('<p class="note">Tituly seskupené podle atributového otisku. „Náročnost" '
                 'měří poměr těžkých (drama, psycho, tragédie) vs. lehkých (komedie, slice-of-life) '
                 'prvků — to je ta osa „emocionální únavy". „Afinita" = o kolik tituly téhle '
                 'nálady hodnotíš nad svůj baseline (komunita + tvůj posun) — synergický efekt '
                 'celé nálady, ne jen součtu jejích atributů.</p>')
    for c in model.clusters:
        if c.intensity > INTENSITY_BADGE:
            pill = f'<span class="pill heavy">náročné · {c.intensity:+.2f}</span>'
        elif c.intensity < -INTENSITY_BADGE:
            pill = f'<span class="pill light">lehké · {c.intensity:+.2f}</span>'
        else:
            pill = f'<span class="pill mix">smíšené · {c.intensity:+.2f}</span>'
        sig = "".join(
            f'<span class="tag cat-{_esc(cat)}{" spoiler-item" if spoil else ""}">{_esc(lab)}</span>'
            for _key, lab, cat, _dist, spoil in c.signature[:6])
        mem = " · ".join(_esc(m[1]) for m in c.members[:6])
        parts.append(
            f'<div class="cl"><div class="name">{_esc(c.name)}</div>'
            f'<div class="meta">{c.size} titulů · průměr {c.mean_user_score:.1f} '
            f'· afinita {c.affinity:+.2f} &nbsp; {pill}</div>'
            f'<div>{sig}</div>'
            f'<div class="note" style="margin-top:10px">{mem}</div></div>')

    # metodika
    parts.append('<h2>Jak to model počítá</h2>')
    parts.append(
        '<div class="panel" style="padding:18px 22px"><p class="note" style="max-width:78ch">'
        '<b>1. Baseline.</b> Pro každý titul: <span class="mono">tvůj_průměr + β·(komunita − průměr_komunity)</span>. '
        'Komunita vstupuje jako <i>jeden</i> kalibrovaný sklon β, ne jako atribut — jinak by se '
        '„kvalita" počítala dvakrát.<br>'
        '<b>2. Cíl = afinita.</b> Co zbude po odečtení baseline. To je tvůj osobní podpis nad rámec toho, '
        'co by čekal kdokoli.<br>'
        '<b>3. Efekty atributů.</b> Empiricko-bayesovsky smrštěný vážený průměr afinity na atribut '
        '(malé vzorky táhnuté k nule koeficientem K). Atributy se objevují samy z dat — žádný ruční '
        'konfigurák.<br>'
        '<b>4. Nálady.</b> KMeans na normalizovaných atributových vektorech, počet klastrů dle siluety.<br>'
        '<b>5. Kalibrace.</b> Globální škála a interval predikce z 5-násobné cross-validace.'
        '</p></div>')
    parts.append('<p class="note" style="margin-top:14px">CV RMSE modelu: '
                 f'<span class="mono">{model.cv_rmse:.3f}</span> vs. samotný baseline '
                 f'<span class="mono">{stats.get("baseline_rmse", model.cv_rmse):.3f}</span>. '
                 'Že je rozdíl malý, není chyba — znamená to, že tvá známka ≈ komunita + konstantní '
                 'posun; atributy proto neslouží k hádání čísla, ale k tomu, <i>co</i> vybrat a do '
                 'jaké nálady to patří.</p>')

    parts.append(_foot(f"shrinkage K={model.K:g}"))
    out = "\n".join(parts)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)
    return out_path


# ── RECOMMENDATIONS report ───────────────────────────────────────────────────

def _anchor_id(name: str) -> str:
    """Převede název klastru na validní HTML anchor ID."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "cluster"


def _rec_card(r, rank: int) -> str:
    """HTML karta jednoho doporučení. Sdílená mezi globálním i per-klastr pohledem."""
    ten = (f' · <span class="ten">{_esc(r.title_en)}</span>'
           if r.title_en and r.title_en != r.title else "")
    flag = '<span class="flag">na tvém PTW</span>' if r.ptw else ""
    why_parts = []
    for lab, cat, val, spoil in r.why:
        sign = "pos" if val >= 0 else "neg"
        cls = f"{sign} spoiler-item" if spoil else sign
        why_parts.append(f'<span class="{cls}">{_esc(lab)}</span>')
    why = ", ".join(why_parts) if why_parts else "—"
    seeds = ""
    if r.cf_seeds:
        seeds = ('<div class="note" style="margin-top:6px">protože máš rád: '
                 + ", ".join(f'<i>{_esc(s)}</i>' for s in r.cf_seeds) + '</div>')
    cl = f'<span class="tag">{_esc(r.cluster_name)}</span>' if r.cluster_name else ""
    comm = f'{r.community:.2f}' if r.community is not None else '—'
    syn = _esc(r.synopsis[:340] + ("…" if len(r.synopsis) > 340 else "")) if r.synopsis else ""
    src = " · ".join(_esc(s) for s in r.sources)
    # oddělené CF signály (viz recommend.py: 4složkový kompozit) -- ukázat,
    # jen když jsou nenulové, ať karty bez daného zdroje nemají prázdné nuly
    cf_boxes = ""
    if r.cf_signal:
        cf_boxes += (f'<div class="s"><div class="n">{r.cf_signal:.0f}</div>'
                     f'<div class="k">graf podobnosti</div></div>')
    if getattr(r, "user_cf_signal", 0.0):
        cf_boxes += (f'<div class="s"><div class="n">{r.user_cf_signal:.1f}</div>'
                     f'<div class="k">user-CF</div></div>')
    return (
        f'<div class="rec"><div class="rank">{rank:02d}</div>'
        f'<div class="t">{_esc(r.title)}{flag}</div>'
        f'<div style="margin:2px 0 4px">{ten}</div>'
        f'<div class="scores">'
        f'<div class="s"><div class="n pos">{r.pred:.1f}</div>'
        f'<div class="k">tvůj odhad ({r.pred_lo:.1f}–{r.pred_hi:.1f})</div></div>'
        f'<div class="s"><div class="n">{comm}</div><div class="k">MAL score</div></div>'
        f'{cf_boxes}'
        f'</div>'
        f'<div class="why"><b>Proč:</b> {why} &nbsp;{cl}</div>'
        f'{seeds}'
        + (f'<div class="syn">{syn}</div>' if syn else "")
        + f'<div class="src" style="margin-top:10px">zdroj: {src}</div>'
        f'</div>'
    )


def render_recommendations_html(recs: list, out_path: str, userinfo: dict = None) -> str:
    u = _esc((userinfo or {}).get("user_name", "tebe"))
    parts = [_head("Doporučení — animodel")]
    parts.append('<p class="kicker">animodel · doporučení na míru</p>')
    parts.append('<h1>Co sledovat<br><em>dál</em></h1>')
    parts.append(f'<p class="lead">{len(recs)} dosud neshlédnutých titulů, seřazených podle '
                 'kompozitního skóre: shoda s tvými atributy a náladami + kolik tvých oblíbených '
                 'je „doporučuje" + mírná preference kvality. Tituly z tvého plan-to-watch '
                 'jsou označené.</p>')

    for i, r in enumerate(recs, 1):
        parts.append(_rec_card(r, i))

    parts.append(_foot())
    out = "\n".join(parts)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)
    return out_path


def render_cluster_recommendations_html(
    recs: list, model, out_path: str, userinfo: dict = None,
    top_per_cluster: int = 15,
) -> str:
    """
    Per-klastrový (per-náladový) pohled na doporučení.

    Tituly jsou seskupeny podle cluster_name. Pořadí v každé sekci je
    zachováno z globálního composite skóre (tj. nejlepší titul v dané
    náladě je na prvním místě). Klastry jsou řazeny podle průměrného
    composite skóre svých doporučení — takže vepředu jsou nálady, kde
    máš nejsilnější match.

    Výsledný soubor je samostatné HTML; odkaz na globální přehled
    (recommendations.html) je v perexu.
    """
    # ── Seskupení recs ───────────────────────────────────────────────────
    cluster_groups: dict[str, list] = defaultdict(list)
    no_cluster: list = []
    for r in recs:   # recs jsou seřazeny globálním composite → pořadí v klastru zachováno
        if r.cluster_name:
            cluster_groups[r.cluster_name].append(r)
        else:
            no_cluster.append(r)

    # Metadata klastrů z modelu (intensity, signature, size)
    cluster_meta = {c.name: c for c in (model.clusters if model and hasattr(model, "clusters") else [])}

    # Klastry seřadit: průměrné composite doporučení sestupně
    def _avg_composite(name: str) -> float:
        g = cluster_groups[name]
        return sum(r.composite for r in g) / len(g) if g else 0.0

    ordered = sorted(cluster_groups.keys(), key=lambda n: -_avg_composite(n))

    # ── HTML ─────────────────────────────────────────────────────────────
    parts = [_head("Doporučení podle nálady — animodel")]
    parts.append('<p class="kicker">animodel · doporučení podle nálady</p>')
    parts.append('<h1 id="top">Co sledovat<br><em>podle nálady</em></h1>')

    n_cl = len(ordered)
    parts.append(
        f'<p class="lead">{len(recs)} doporučení v {n_cl} náladách. '
        f'Pořadí v každé sekci odpovídá kompozitnímu skóre pro danou náladu. '
        f'Globální přehled viz '
        f'<a href="recommendations.html">recommendations.html</a>.</p>'
    )

    # Navigační panel — přehled nálad
    parts.append('<div class="panel" style="padding:16px 22px;margin-bottom:32px">')
    parts.append('<h3 style="margin:0 0 10px">Nálady</h3>')
    parts.append('<div style="line-height:2.4">')
    for name in ordered:
        count = len(cluster_groups[name])
        anchor = _anchor_id(name)
        meta = cluster_meta.get(name)
        if meta:
            if meta.intensity > INTENSITY_BADGE:
                badge = f' <span class="pill heavy" style="font-size:10px">náročné</span>'
            elif meta.intensity < -INTENSITY_BADGE:
                badge = f' <span class="pill light" style="font-size:10px">lehké</span>'
            else:
                badge = f' <span class="pill mix" style="font-size:10px">smíšené</span>'
        else:
            badge = ""
        parts.append(
            f'<a href="#{anchor}" style="margin-right:22px;white-space:nowrap">'
            f'<b>{_esc(name)}</b>{badge} '
            f'<span style="color:{MUT};font-size:12px">({count})</span></a>'
        )
    if no_cluster:
        parts.append(
            f'<a href="#ostatni" style="margin-right:22px">'
            f'ostatní <span style="color:{MUT};font-size:12px">({len(no_cluster)})</span></a>'
        )
    parts.append('</div></div>')

    # ── Sekce pro každý klastr ────────────────────────────────────────────
    for name in ordered:
        group = cluster_groups[name]
        anchor = _anchor_id(name)
        meta = cluster_meta.get(name)

        # Hlavička klastru
        if meta:
            if meta.intensity > INTENSITY_BADGE:
                pill = f'<span class="pill heavy">náročné · {meta.intensity:+.2f}</span>'
            elif meta.intensity < -INTENSITY_BADGE:
                pill = f'<span class="pill light">lehké · {meta.intensity:+.2f}</span>'
            else:
                pill = f'<span class="pill mix">smíšené · {meta.intensity:+.2f}</span>'
            sig = "".join(
                f'<span class="tag cat-{_esc(cat)}{" spoiler-item" if spoil else ""}">{_esc(lab)}</span>'
                for _key, lab, cat, _dist, spoil in meta.signature[:6]
            )
            meta_html = (
                f'<div class="meta">'
                f'{meta.size} titulů v modelu · průměr {meta.mean_user_score:.1f}'
                f' · afinita {meta.affinity:+.2f}'
                f' &nbsp; {pill}</div>'
                f'<div style="margin:6px 0 4px">{sig}</div>'
            )
        else:
            meta_html = ""

        parts.append(f'<h2 id="{anchor}" style="margin-top:72px">{_esc(name)}</h2>')
        group_display = group[:top_per_cluster]
        total_in_cluster = len(group)
        parts.append(
            f'<div class="cl" style="margin-bottom:16px">'
            f'{meta_html}'
            f'<div class="note">zobrazeno {len(group_display)} z {total_in_cluster} kandidátů v této náladě</div>'
            f'</div>'
        )

        for i, r in enumerate(group_display, 1):
            parts.append(_rec_card(r, i))

        parts.append(
            f'<p style="text-align:right;margin-top:2px;margin-bottom:0">'
            f'<a href="#top" style="font-size:12px;color:{MUT}">↑ zpět nahoru</a></p>'
        )

    # ── Tituly bez klastru ────────────────────────────────────────────────
    if no_cluster:
        parts.append('<h2 id="ostatni" style="margin-top:72px">Ostatní</h2>')
        parts.append('<p class="note">Tituly bez přiřazené nálady.</p>')
        for i, r in enumerate(no_cluster, 1):
            parts.append(_rec_card(r, i))
        parts.append(
            f'<p style="text-align:right;margin-top:2px">'
            f'<a href="#top" style="font-size:12px;color:{MUT}">↑ zpět nahoru</a></p>'
        )

    parts.append(_foot())
    out_html = "\n".join(parts)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out_html)
    return out_path


def _cf_rec_card(r_cf: dict, enr, rank: int, watched: bool = False) -> str:
    """Karta CF doporučeni. Kombinuje enr (Recommendation) a r_cf (raw CF dict)."""
    mid       = r_cf.get("mal_id")
    cf_score  = r_cf.get("cf_score", 0.0)
    community = r_cf.get("community", 0.0)
    diff      = r_cf.get("diff", 0.0)
    n_users   = r_cf.get("n_users", 0)
    raters    = r_cf.get("top_raters", [])

    # Tituly
    title    = _esc((enr.title    if enr else None) or r_cf.get("title") or f"MAL #{mid}")
    title_en = _esc((enr.title_en if enr else None) or r_cf.get("title_en", ""))
    if title_en == title:
        title_en = ""
    ten  = f' \xb7 <span class="ten">{title_en}</span>' if title_en else ""
    ptw  = enr.ptw if enr else False
    # CF report je surov\u00fd pohled: shl\u00e9dnut\u00e9 tituly se z n\u011bj nefiltruj\u00ed
    # (potvrzuj\u00ed shodu vkusu se sp\u0159\u00edzn\u011bn\u00fdmi du\u0161emi), jen se ozna\u010d\u00ed --
    # do fin\u00e1ln\u00edch \u017eeb\u0159\u00ed\u010dk\u016f se na rozd\u00edl odsud nikdy nedostanou.
    if watched:
        flag = '<span class="flag seen">u\u017e shl\u00e9dnuto</span>'
    elif ptw:
        flag = '<span class="flag">na tve\u0301m PTW</span>'
    else:
        flag = ""
    mal_url = f"https://myanimelist.net/anime/{mid}" if mid else "#"

    # Skore sekce
    cf_color   = ACCENT  if cf_score >= 9.0  else (ACCENT2 if cf_score >= 7.5 else MUT)
    diff_color = ACCENT  if diff >= 0         else NEG
    comm_str   = f"{community:.2f}" if community else "\u2014"

    scores_html = (
        f'<div class="scores">'
        f'<div class="s"><div class="n" style="color:{cf_color}">{cf_score:.2f}</div>'
        f'<div class="k">CF sk\xf3re</div></div>'
        f'<div class="s"><div class="n">{comm_str}</div>'
        f'<div class="k">AL pr\u016fm\u011br</div></div>'
        f'<div class="s"><div class="n" style="color:{diff_color}">{diff:+.2f}</div>'
        f'<div class="k">\u0394 od pr\u016fm\u011bru</div></div>'
        + (f'<div class="s"><div class="n pos">{enr.pred:.1f}</div>'
           f'<div class="k">pred. sk\xf3re ({enr.pred_lo:.1f}\u2013{enr.pred_hi:.1f})</div></div>'
           if enr else "")
        + '</div>'
    )

    # Spriznenene duse
    raters_html = ""
    if raters:
        names = ", ".join(
            f'<span style="color:{ACCENT2}">{_esc(nm)}</span>'
            f'<span style="color:{MUT};font-size:11px"> r={_esc(str(sim))}</span>'
            for nm, sim in raters[:5]
        )
        raters_html = (
            f'<div class="note" style="margin-top:6px">'
            f'sp\u0159\xedzn\u011bn\xe9 du\u0161e ({n_users}): {names}</div>'
        )

    # Proc / atributy z enr
    why_html = ""
    if enr and enr.why:
        why_parts = []
        for lab, cat, val, spoil in enr.why:
            sign = "pos" if val >= 0 else "neg"
            cls = f"{sign} spoiler-item" if spoil else sign
            why_parts.append(f'<span class="{cls}">{_esc(lab)}</span>')
        why = ", ".join(why_parts)
        cl  = f'<span class="tag">{_esc(enr.cluster_name)}</span>' if enr.cluster_name else ""
        why_html = f'<div class="why"><b>Pro\u010d:</b> {why} &nbsp;{cl}</div>'

    # Seeds z item-CF
    seeds_html = ""
    if enr and enr.cf_seeds:
        seeds_html = (
            '<div class="note" style="margin-top:6px">proto\u017ee m\xe1\u0161 r\xe1d: '
            + ", ".join(f'<i>{_esc(s)}</i>' for s in enr.cf_seeds)
            + '</div>'
        )

    # Synopsis
    syn_html = ""
    synopsis = (enr.synopsis if enr else "") or ""
    if synopsis:
        syn = _esc(synopsis[:340] + ("\u2026" if len(synopsis) > 340 else ""))
        syn_html = f'<div class="syn">{syn}</div>'

    # Zdroje
    sources = list(enr.sources) if enr else ["user-CF"]
    if "user-CF" not in sources:
        sources.append("user-CF")
    src = " \xb7 ".join(_esc(s) for s in sources)

    return (
        f'<div class="rec">'
        f'<div class="rank">{rank:02d}</div>'
        f'<a href="{mal_url}" target="_blank" style="text-decoration:none;color:inherit">'
        f'<div class="t">{title}{flag}</div></a>'
        f'<div style="margin:2px 0 4px">{ten}</div>'
        + scores_html
        + why_html
        + raters_html
        + seeds_html
        + syn_html
        + f'<div class="src" style="margin-top:10px">zdroj: {src}</div>'
        f'</div>'
    )


def render_cf_recommendations_html(
    cf_recs: list[dict],
    out_path: str,
    userinfo: dict = None,
    enr_data: dict = None,
    watched_ids: set = None,
) -> str:
    """Standalone HTML report pro vysledky user-based CF, karta-style."""
    parts = [_head("CF doporu\u010den\xed \u2014 animodel")]
    parts.append('<p class="kicker">animodel \xb7 user-based collaborative filtering</p>')
    parts.append('<h1>Doporu\u010den\xed<br><em>od sp\u0159\xedzn\u011bn\xfdch du\u0161\xed</em></h1>')
    parts.append(
        f'<p class="lead">'
        f'{len(cf_recs)} titul\u016f nalezen\xfdch p\u0159es podobn\xe9 u\u017eivatele na AniList. '
        f'CF sk\xf3re\xa0=\xa0pr\u016fm\u011br komunity\xa0+\xa0v\xe1\u017een\xe1 odchylka sp\u0159\xedn\u011bn\xfdch du\u0161\xed. '
        f'Kladn\xe1\xa0\u0394\xa0=\xa0sp\u0159\xedn\u011bn\xe9 du\u0161e hodnotow\xed v\xfd\u0161 ne\u017e komunita. '
        f'Se\u0159azeno sestupn\u011b dle CF sk\xf3re. Na rozd\xedl od fin\xe1ln\xedch '
        f'\u017eeb\u0159\xed\u010dk\u016f se tu shl\xe9dnut\xe9 tituly nefiltruj\xed, jen '
        f'ozna\u010duj\xed \u0161t\xedtkem \u2014 potvrzuj\xed shodu vkusu.</p>'
    )
    watched = watched_ids or set()
    for i, r_cf in enumerate(cf_recs, 1):
        mid = r_cf.get("mal_id")
        enr = (enr_data or {}).get(mid)
        parts.append(_cf_rec_card(r_cf, enr, i, watched=mid in watched))
    parts.append(_foot())
    out = "\n".join(parts)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)
    return out_path

