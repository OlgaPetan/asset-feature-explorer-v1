"""
Asset Intelligence — Asset Feature Explorer
The Coca-Cola Company
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Asset Intelligence",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── palette ───────────────────────────────────────────────────────────────────
RED    = "#E8002D"
DARK   = "#1C1C1C"
MID    = "#6A6660"
LIGHT  = "#FAFAF8"
BLUE   = "#2D5BE3"
GREEN  = "#2A8050"
AMBER  = "#906820"
PURPLE = "#7030A0"
BORDER = "#EAE8E2"

M_COLOR = {
    "Attention_T2B":             BLUE,
    "Persuasion_T2B":            RED,
    "Likeability_Love_Like_T2B": GREEN,
    "SCD_score":                 PURPLE,
    "Experience_Recall_T2B":     "#B45309",
    "Brand_Linkage_T2B":         "#0E7490",
    "Uniqueness_T2B":            "#6D28D9",
    "Shareability_T2B":          "#065F46",
}
M_LABEL = {
    "Attention_T2B":             "Attention",
    "Persuasion_T2B":            "Persuasion",
    "Likeability_Love_Like_T2B": "Likeability",
    "SCD_score":                 "SCD Score",
    "Experience_Recall_T2B":     "Experience Recall",
    "Brand_Linkage_T2B":         "Brand Linkage",
    "Uniqueness_T2B":            "Uniqueness",
    "Shareability_T2B":          "Shareability",
}
NEW_METRICS = ["Experience_Recall_T2B","Brand_Linkage_T2B","Uniqueness_T2B","Shareability_T2B"]
RULE_STYLE = {
    "Conflict":           ("warn-conflict",     "⚡"),
    "Heterogeneity":      ("warn-heterogeneity","⚠"),
    "Boundary Condition": ("warn-boundary",     "◈"),
    "Opportunity":        ("warn-opportunity",  "◎"),
    "Outlier":            ("warn-outlier",      "↗"),
    "Consensus":          ("warn-consensus",    "✓"),
    "Anti-pattern":       ("warn-antipattern",  "✕"),
    "Insight":            ("warn-insight",      "ℹ"),
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700&family=Source+Sans+3:wght@300;400;500;600&display=swap');

html,body,[class*="css"]{font-family:'Source Sans 3',sans-serif;font-size:15px;background:#FAFAF8;color:#1C1C1C;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0 2.8rem 4rem 2.8rem!important;max-width:1440px;}

.topbar{background:#E8002D;margin:0 -2.8rem 0 -2.8rem;padding:.6rem 2.8rem;display:flex;align-items:center;justify-content:space-between;}
.topbar-left{display:flex;align-items:center;gap:.9rem;}
.topbar-logo{font-family:'Merriweather',serif;font-size:1.05rem;font-weight:400;color:#fff;}
.topbar-pipe{width:1px;height:16px;background:rgba(255,255,255,.3);}
.topbar-sub{font-size:.75rem;color:rgba(255,255,255,.65);letter-spacing:.1em;text-transform:uppercase;}

.hero{padding:2rem 0 1.5rem 0;border-bottom:1px solid #EAE8E2;margin-bottom:2rem;}
.hero-eyebrow{font-size:.72rem;font-weight:600;letter-spacing:.2em;text-transform:uppercase;color:#E8002D;margin-bottom:.5rem;}
.hero-title{font-family:'Merriweather',serif;font-size:2.4rem;font-weight:300;line-height:1.2;color:#1C1C1C;margin-bottom:.7rem;}
.hero-sub{font-size:.95rem;color:#7A7670;font-weight:300;max-width:620px;line-height:1.75;}

.sec-label{font-size:.72rem;font-weight:600;letter-spacing:.18em;text-transform:uppercase;color:#6A6660;margin-bottom:.8rem;}

.scope-chip{display:inline-flex;align-items:center;background:#fff0f0;border:1px solid #f0c0c0;border-radius:3px;padding:.2rem .7rem;font-size:.82rem;color:#c00020;margin:.18rem .25rem .18rem 0;font-weight:500;}

.kpi-card{background:#fff;border:1px solid #EAE8E2;border-radius:6px;padding:1.1rem 1.3rem;}
.kpi-val{font-family:'Merriweather',serif;font-size:2rem;font-weight:300;line-height:1;}
.kpi-lbl{font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:#AAA8A0;margin-top:.3rem;}
.kpi-sub{font-size:.78rem;color:#888;margin-top:.22rem;}

.explain-box{background:#F5F3EF;border-left:3px solid #E8002D;border-radius:0 4px 4px 0;padding:.8rem 1rem;font-size:.82rem;color:#555;line-height:1.65;margin-bottom:1rem;}

.badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:.69rem;font-weight:600;letter-spacing:.03em;}
.b-pos{background:#E8F8EF;color:#2A8050;}.b-neg{background:#FDEDEC;color:#C00020;}
.b-ns{background:#F2F2F2;color:#888;}.b-sig{background:#F3EEFF;color:#7030A0;}
.b-metric{background:#EBF3FB;color:#2D5BE3;}.b-scope{background:#F5F5F5;color:#555;}
.b-hi{background:#EAFAF1;color:#27AE60;}.b-med{background:#FEF9E7;color:#E67E22;}.b-lo{background:#F2F2F2;color:#888;}

.feat-row{display:flex;align-items:center;padding:.55rem 0;border-bottom:1px solid #F5F3EF;}
.feat-name{width:140px;font-size:.88rem;font-weight:600;color:#1C1C1C;flex-shrink:0;}
.feat-bar-wrap{flex:1;display:flex;align-items:center;gap:.5rem;}
.feat-val{width:68px;text-align:right;font-size:.85rem;font-weight:600;flex-shrink:0;}
.feat-sig{width:28px;font-size:.72rem;text-align:center;flex-shrink:0;}
.feat-n{width:52px;font-size:.72rem;color:#AAA;text-align:right;flex-shrink:0;}

.combo-card{background:#fff;border:1px solid #EAE8E2;border-left:3px solid #E8002D;border-radius:0 6px 6px 0;padding:1.6rem 1.8rem;margin-bottom:1.3rem;box-shadow:0 1px 4px rgba(0,0,0,.04);}
.combo-uplift{font-family:'Merriweather',serif;font-size:3.2rem;color:#E8002D;line-height:1;font-weight:300;}
.pill-row{display:flex;flex-wrap:wrap;gap:.35rem;margin-top:.9rem;}
.pill{display:inline-flex;align-items:center;gap:.4rem;background:#FAFAF8;border:1px solid #EAE8E2;border-radius:3px;padding:.26rem .75rem;font-size:.82rem;color:#5A5650;}
.pill-n{background:#E8002D;color:#fff;border-radius:2px;width:17px;height:17px;display:inline-flex;align-items:center;justify-content:center;font-size:.65rem;font-weight:700;flex-shrink:0;}
.pill-gain{color:#2A8050;font-size:.79rem;font-weight:600;}

.wf-row{display:flex;align-items:center;gap:.9rem;padding:.52rem 0;border-bottom:1px solid #F5F3EF;}
.wf-idx{width:20px;font-size:.74rem;color:#CCC8C2;text-align:center;flex-shrink:0;}
.wf-label{flex:1;font-size:.88rem;color:#3A3830;}
.wf-bar{width:140px;flex-shrink:0;}
.wf-n{width:50px;font-size:.73rem;color:#BBB8B2;text-align:right;flex-shrink:0;}
.wf-pp{width:60px;font-size:.86rem;font-weight:600;color:#2A8050;text-align:right;flex-shrink:0;}
.wf-col{font-size:.64rem;letter-spacing:.12em;text-transform:uppercase;color:#CCC8C2;}

.explorer-card{background:#fff;border:1px solid #EAE8E2;border-radius:6px;padding:1.6rem 1.8rem;margin-top:1.2rem;box-shadow:0 1px 4px rgba(0,0,0,.04);}
.explorer-title{font-family:'Merriweather',serif;font-size:1.3rem;font-weight:300;color:#1C1C1C;margin-bottom:.22rem;}
.explorer-sub{font-size:.87rem;color:#AAA89E;line-height:1.65;margin-bottom:1.5rem;}
.feat-group{font-size:.65rem;font-weight:600;letter-spacing:.18em;text-transform:uppercase;color:#CCC8C2;margin:1.1rem 0 .65rem 0;padding-bottom:.3rem;border-bottom:1px solid #F0EEE8;}

.scoreboard{display:flex;gap:.9rem;margin-top:1.3rem;flex-wrap:wrap;}
.score-cell{flex:1;background:#FAFAF8;border:1px solid #EAE8E2;border-radius:5px;padding:1rem 1.1rem;text-align:center;min-width:100px;}
.score-cell.hi{border-color:#E8002D;background:#FFF8F8;}
.score-metric{font-size:.67rem;letter-spacing:.14em;text-transform:uppercase;color:#CCC8C2;margin-bottom:.35rem;}
.score-val{font-family:'Merriweather',serif;font-size:2.1rem;font-weight:300;color:#1C1C1C;line-height:1;}
.score-delta{font-size:.82rem;font-weight:500;margin-top:.25rem;}
.d-up{color:#2A8050;}.d-dn{color:#C03030;}.d-flat{color:#CCC8C2;}
.score-n{font-size:.72rem;color:#D8D4CE;margin-top:.2rem;}

.asset-card{background:#fff;border:1px solid #EAE8E2;border-radius:6px;padding:.85rem 1rem;margin-bottom:.5rem;}
.asset-name{font-size:.8rem;font-weight:600;color:#1C1C1C;word-break:break-all;}
.asset-meta{font-size:.72rem;color:#AAA89E;margin-top:.15rem;}
.asc{font-size:.7rem;padding:2px 7px;border-radius:3px;font-weight:600;margin-right:.3rem;}
.asc-a{background:#EBF3FB;color:#2D5BE3;}.asc-p{background:#FFF0F0;color:#C00020;}
.asc-l{background:#EAFAF1;color:#2A8050;}.asc-s{background:#F3EEFF;color:#7030A0;}

.insight-strip{display:flex;flex-direction:column;gap:.38rem;margin:.45rem 0;}
.insight-warn{display:flex;align-items:flex-start;gap:.6rem;border-radius:4px;padding:.52rem .82rem;font-size:.82rem;line-height:1.5;border:1px solid;}
.iw-icon{flex-shrink:0;margin-top:.04rem;}.iw-body{flex:1;}
.iw-type{font-size:.64rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.14rem;}
.warn-conflict{background:#FFF0F0;border-color:#F0C0C0;}.warn-conflict .iw-type{color:#C00020;}
.warn-heterogeneity{background:#FFFBF0;border-color:#EDD080;}.warn-heterogeneity .iw-type{color:#906820;}
.warn-boundary{background:#F8F0FF;border-color:#D4B0F0;}.warn-boundary .iw-type{color:#7030A0;}
.warn-opportunity{background:#F0FFF4;border-color:#90D8A8;}.warn-opportunity .iw-type{color:#1A7040;}
.warn-outlier{background:#EBF3FB;border-color:#90B0E8;}.warn-outlier .iw-type{color:#1A3080;}
.warn-consensus{background:#F0FFF4;border-color:#90D8A8;}.warn-consensus .iw-type{color:#1A7040;}
.warn-antipattern{background:#FFF0F0;border-color:#F0C0C0;}.warn-antipattern .iw-type{color:#C00020;}
.warn-insight{background:#FAFAF8;border-color:#E4E0DA;}.warn-insight .iw-type{color:#6A6660;}

.insight-card{background:#fff;border:1px solid #E8E4DC;border-left:4px solid #E8002D;border-radius:6px;padding:12px 16px;margin-bottom:8px;}
.ic-feat{font-weight:600;font-size:.9rem;color:#1C1C1C;}
.ic-text{font-size:.82rem;color:#444;margin-top:3px;line-height:1.5;}
.ic-meta{display:flex;gap:6px;margin-top:7px;flex-wrap:wrap;}

.low-n{background:#FFFBF0;border:1px solid #EDD080;border-radius:4px;padding:.46rem .85rem;font-size:.81rem;color:#906820;margin-top:.6rem;}
hr.div{border:none;border-top:1px solid #EAE8E2;margin:1.8rem 0;}
.footer{display:flex;justify-content:space-between;font-size:.72rem;color:#CCC8C2;padding:1.1rem 0 .7rem 0;border-top:1px solid #EAE8E2;margin-top:2rem;}

.stButton>button{background:#fff!important;border:1px solid #DAD6D0!important;color:#6A6660!important;border-radius:4px!important;font-size:.88rem!important;width:100%!important;}
.stButton>button:hover{border-color:#E8002D!important;color:#E8002D!important;}
.stTabs [data-baseweb="tab-list"]{background:transparent;border-bottom:1px solid #EAE8E2;gap:0;}
.stTabs [data-baseweb="tab"]{background:transparent;color:#AAA89E;font-size:.9rem;border-radius:0;padding:.52rem 1.3rem;border:none;border-bottom:2px solid transparent;margin-bottom:-1px;}
.stTabs [aria-selected="true"]{background:transparent!important;color:#1C1C1C!important;border-bottom:2px solid #E8002D!important;font-weight:600!important;}
.stSelectbox label{font-size:.8rem!important;color:#AAA89E!important;font-weight:400!important;letter-spacing:.05em!important;text-transform:uppercase!important;}
.stSelectbox>div>div{background:#FAFAF8!important;border:1px solid #E4E0DA!important;border-radius:4px!important;color:#1C1C1C!important;font-size:.9rem!important;}
</style>
""", unsafe_allow_html=True)


# ── data loading ──────────────────────────────────────────────────────────────
@st.cache(allow_output_mutation=True)
def load_all():
    with open("precomputed_data.pkl", "rb") as f:
        payload = pickle.load(f)
    results = payload["results"]
    meta    = payload["meta"]
    df      = meta["df"].copy()

    # SCD score
    see  = ["Experience_Recall_T2B_percentile","Brand_Linkage_T2B_percentile","Comprehension_T2B_percentile"]
    conn = ["Likeability_Love_Like_T2B_percentile","Uniqueness_T2B_percentile","Brand_Interest_T2B_percentile"]
    do   = ["Persuasion_T2B_percentile","Shareability_T2B_percentile"]
    df["SCD_score"] = (
        df[[c for c in see  if c in df.columns]].mean(axis=1, skipna=False) * 0.1 +
        df[[c for c in conn if c in df.columns]].mean(axis=1, skipna=False) * 0.3 +
        df[[c for c in do   if c in df.columns]].mean(axis=1, skipna=False) * 0.6
    ).round(4)

    # insight catalog + rulebook
    try:
        catalog  = pd.read_csv("insight_catalog.csv")
        rulebook = pd.read_csv("rulebook.csv")
        for d in [catalog, rulebook]:
            for c in d.select_dtypes(include="object").columns:
                d[c] = d[c].fillna("")
        catalog["evidence_uplift_pp"] = pd.to_numeric(catalog["evidence_uplift_pp"], errors="coerce")
        has_ins = True
    except FileNotFoundError:
        catalog = rulebook = None
        has_ins = False

    try:
        uplift_df = pd.read_csv("uplift_all_scopes.csv")
        uplift_df["uplift_pp"] = pd.to_numeric(uplift_df["uplift_pp"], errors="coerce")
    except FileNotFoundError:
        uplift_df = None

    camp_map = (df[["campaign_sk_id","campaign_display_name","campaign_code"]]
                .drop_duplicates("campaign_sk_id")
                .set_index("campaign_sk_id"))

    return df, results, meta, catalog, rulebook, uplift_df, has_ins, camp_map

try:
    df_full, results, meta, catalog, rulebook, uplift_df, has_ins, camp_map = load_all()
except FileNotFoundError as e:
    st.error(f"Required file not found: {e}"); st.stop()

METRICS      = meta["metrics"]   # {col: label}
FEAT_LABEL   = meta["feat_label"]
BIN_FEATS    = meta["binary_feats"]
CAT_FEATS    = meta["cat_feats"]
ALL_FEATS    = [
    "animals_and_pets_presence","animatics_cartoons_presence","food_presence",
    "human_presence","outdoors","indoors","product_presence",
    "color_contrast_cat","text_color_contrast_cat","color_spectrum",
    "occasions","passion_point","moments","music_style","seasonal",
]
SCOPE_COL = {"ou":"operating_unit_code","category":"category",
             "brand":"brand_name","market":"country_name"}
ALL_METRICS = list(dict.fromkeys(list(METRICS.keys()) + ["SCD_score"] + NEW_METRICS))


# ── html helpers ──────────────────────────────────────────────────────────────
def badge(txt, cls): return f'<span class="badge {cls}">{txt}</span>'
def bpos(v):
    if pd.isna(v): return badge("—","b-ns")
    cls = "b-pos" if v>=0 else "b-neg"
    return badge(f'{"▲" if v>=0 else "▼"} {abs(v):.1f}pp', cls)
def sig_badge(s):
    if s in ("***","**","*"): return badge(s,"b-sig")
    return badge("ns","b-ns")
def conf_badge(c): return badge(c.upper(), f"b-{c[:2]}")

def render_alerts(items, max_items=6):
    if not items: return
    import re
    def highlight_pp(text):
        # highlight patterns like +5.1pp, -4.4pp, 5.1pp with bold colored spans
        def repl(m):
            val = m.group(0)
            color = "#2A8050" if "+" in val or (val[0].isdigit()) else "#C00020"
            # detect sign more carefully
            try:
                num = float(re.sub(r"[^\d.\-]","", val.replace("pp","")))
                color = "#2A8050" if num >= 0 else "#C00020"
            except: pass
            return f'<strong style="color:{color};font-size:.88rem">{val}</strong>'
        return re.sub(r"[+\-]?\d+\.?\d*pp", repl, text)

    html = '<div class="insight-strip">'
    for it in items[:max_items]:
        css, icon = RULE_STYLE.get(it["type"], ("warn-insight","ℹ"))
        html += (f'<div class="insight-warn {css}">'
                 f'<div class="iw-icon">{icon}</div>'
                 f'<div class="iw-body">'
                 f'<div class="iw-type">{it["type"]}</div>'
                 f'<div style="font-size:.82rem;color:#333">{highlight_pp(it["text"])}</div>'
                 f'</div></div>')
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def get_scope_alerts(scope_filters, max_items=6):
    if not has_ins or rulebook is None: return []
    smap = {"ou":"OU","category":"Category","brand":"Brand","market":"Market"}
    if not scope_filters:
        rb = rulebook[(rulebook["scope"]=="Global") &
                      (rulebook["rule_type"].isin(["Conflict","Anti-pattern","Consensus"]))]
    else:
        mask = pd.Series(False, index=rulebook.index)
        for t,v in scope_filters:
            mask |= ((rulebook["scope"]==smap.get(t,t)) & (rulebook["scope_value"]==v))
        mask |= (rulebook["scope"]=="Global")
        rb = rulebook[mask]
    rb = rb.sort_values("severity", key=lambda s: s.map({"high":0,"medium":1,"low":2}))
    return [{"type":r["rule_type"],"text":r["text"]} for _,r in rb.head(max_items).iterrows()]

def get_feature_alerts(feature, scope_filters, metric_col, max_items=3):
    if not has_ins or rulebook is None or catalog is None: return []
    smap = {"ou":"OU","category":"Category","brand":"Brand","market":"Market"}
    ml_map = {"Attention_T2B":"Attention","Persuasion_T2B":"Persuasion",
              "Likeability_Love_Like_T2B":"Likeability","SCD_score":"SCD Score",
              "Experience_Recall_T2B":"Experience Recall","Brand_Linkage_T2B":"Brand Linkage",
              "Uniqueness_T2B":"Uniqueness","Shareability_T2B":"Shareability"}
    ml = ml_map.get(metric_col, metric_col)
    items = []
    # rulebook for this feature
    rb = rulebook[rulebook["feature"]==feature]
    if scope_filters:
        sv = {smap.get(t,t):v for t,v in scope_filters}
        rb_s = rb[rb.apply(lambda r: r["scope"] in sv and sv.get(r["scope"])==r["scope_value"],axis=1)]
        rb = pd.concat([rb_s, rb[rb["scope"]=="Global"]]).drop_duplicates()
    else:
        rb = rb[rb["scope"]=="Global"]
    rb = rb.sort_values("severity", key=lambda s: s.map({"high":0,"medium":1,"low":2}))
    for _,r in rb.head(max_items).iterrows():
        items.append({"type":r["rule_type"],"text":r["text"]})
    # catalog insight
    if len(items) < max_items:
        cat = catalog[(catalog["feature"]==feature) &
                      (catalog["metric_display"]==ml) &
                      (catalog["confidence"].isin(["high","medium"]))]
        added = False
        if scope_filters:
            for t,v in scope_filters:
                sl = smap.get(t,t)
                for _,r in cat[(cat["filter"]==sl)&(cat["filter_value"]==v)].head(1).iterrows():
                    items.append({"type":"Insight","text":r["text"]}); added=True
        if not added:
            for _,r in cat[cat["filter"]=="Global"].head(1).iterrows():
                items.append({"type":"Insight","text":r["text"]})
    return items[:max_items]


# ── statistical helpers ───────────────────────────────────────────────────────
def compute_uplift(sub, feat, mc):
    if feat not in sub.columns or mc not in sub.columns: return None,None,None
    if feat in BIN_FEATS:
        g1=sub.loc[sub[feat]==1,mc].dropna(); g0=sub.loc[sub[feat]==0,mc].dropna()
    else:
        mask=sub[feat].notna()&(sub[feat].astype(str).str.strip()!="")
        g1=sub.loc[mask,mc].dropna(); g0=sub.loc[~mask,mc].dropna()
    if len(g1)<3 or len(g0)<3: return None,None,None
    u=(g1.mean()-g0.mean())*100
    try: _,p=mannwhitneyu(g1,g0,alternative="two-sided")
    except: p=1.0
    sig="***" if p<.001 else ("**" if p<.01 else ("*" if p<.05 else "ns"))
    return round(u,2),sig,len(g1)

def feature_combinations(sub, target, mc, top_n=5):
    """Top binary feature partners for target feat on metric mc."""
    if target not in sub.columns or mc not in sub.columns: return []
    gm = sub[mc].dropna().mean()*100
    g1t = sub.loc[sub[target]==1,mc].dropna() if target in BIN_FEATS \
          else sub.loc[sub[target].notna()&(sub[target].astype(str).str.strip()!=""),mc].dropna()
    if len(g1t)<3: return []
    solo_t = g1t.mean()*100 - gm
    rows=[]
    for f in BIN_FEATS:
        if f==target or f not in sub.columns: continue
        if target in BIN_FEATS:
            mb=(sub[target]==1)&(sub[f]==1)
        else:
            mt=sub[target].notna()&(sub[target].astype(str).str.strip()!="")
            mb=mt&(sub[f]==1)
        nb=int(mb.sum())
        if nb<3: continue
        cu=sub.loc[mb,mc].dropna().mean()*100-gm
        g1p=sub.loc[sub[f]==1,mc].dropna()
        sp=g1p.mean()*100-gm if len(g1p)>=3 else 0
        rows.append({"partner":f,"label":FEAT_LABEL.get(f,f),
                     "combined":round(cu,2),"solo_t":round(solo_t,2),
                     "solo_p":round(sp,2),"synergy":round(cu-max(solo_t,sp),2),"n":nb})
    rows.sort(key=lambda x:x["combined"],reverse=True)
    return rows[:top_n]


# ── precomputed combo helpers ─────────────────────────────────────────────────
def get_scope_key(scope_filters):
    if not scope_filters: return "global||All"
    if len(scope_filters)==1:
        t,v=scope_filters[0]; k=f"{t}||{v}"
        return k if k in results else None
    return None

def get_combo(sub_df, scope_key, mc, min_n=5):
    if scope_key and scope_key in results and mc in results[scope_key]:
        c=results[scope_key][mc]
        return c["combo"],c["steps"],c["baseline_pp"],c["n_total"]
    return None,[],0,len(sub_df)

def default_sel(combo):
    sel={f:"__any__" for f in ALL_FEATS}
    if combo:
        for feat,val in combo:
            if feat in ALL_FEATS:
                sel[feat]="1" if val==1 else ("0" if val==0 else str(val))
    return sel

def apply_sel(sub_df,sel):
    mask=pd.Series(True,index=sub_df.index)
    for feat,val in sel.items():
        if val=="__any__" or feat not in sub_df.columns: continue
        mask&=(sub_df[feat]==(1 if val=="1" else 0)) if feat in BIN_FEATS else (sub_df[feat]==val)
    return mask

def score_sel(sub_df,sel):
    mask=apply_sel(sub_df,sel); n=int(mask.sum())
    return {mc:(round(sub_df.loc[mask,mc].dropna().mean()*100,2),n) if n>0 else (None,0)
            for mc in ALL_METRICS if mc in sub_df.columns}

def get_opts(feat,scope_key,mc,sub_df):
    fv=None
    if scope_key and scope_key in results and mc in results[scope_key]:
        fv=results[scope_key][mc].get("feature_value_uplift",{}).get(feat) or \
           results[scope_key][mc].get("feature_options",{}).get(feat)
    raws,disp=["__any__"],["Any"]
    if fv:
        for row in fv:
            n_row=row.get("n",0)
            if n_row<3: continue
            raws.append(str(row["val"]))
            up=row.get("uplift_pp")
            if up is not None and not row.get("is_baseline",False):
                disp.append(f"{row['label']}  ({'+'if up>=0 else ''}{up:.1f}pp, n={n_row:,})")
            else:
                disp.append(f"{row['label']}  (n={n_row:,})")
    elif feat in BIN_FEATS and feat in sub_df.columns:
        n1=int((sub_df[feat]==1).sum()); n0=int((sub_df[feat]==0).sum())
        if n1>=3: raws+=["1"]; disp+=[f"Yes  (n={n1:,})"]
        if n0>=3: raws+=["0"]; disp+=[f"No  (n={n0:,})"]
    elif feat in sub_df.columns:
        for v in sorted(sub_df[feat].dropna().unique()):
            n=int((sub_df[feat]==v).sum())
            if n>=3: raws.append(str(v)); disp.append(f"{str(v)[:32]}  (n={n:,})")
    return raws,disp

def bar_html(frac):
    pct=max(2,min(100,frac*100))
    return (f'<div style="background:#F0EDE8;border-radius:2px;height:5px;">'
            f'<div style="width:{pct:.0f}%;height:5px;background:#E8002D;border-radius:2px;"></div></div>')

def make_heatmap(piv,title,fs=(13,5)):
    cmap=LinearSegmentedColormap.from_list("rg",["#D32F2F","white","#2E7D32"],N=256)
    vals=piv.values.astype(float); vmax=max(np.nanpercentile(np.abs(vals),95),1.0)
    fig,ax=plt.subplots(figsize=fs); fig.patch.set_facecolor(LIGHT); ax.set_facecolor(LIGHT)
    im=ax.imshow(vals,cmap=cmap,vmin=-vmax,vmax=vmax,aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns,fontsize=8.5,rotation=30,ha="right")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index,fontsize=8.5)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v=vals[i,j]
            if not np.isnan(v):
                ax.text(j,i,f"{v:+.1f}",ha="center",va="center",fontsize=7.5,
                        color="white" if abs(v)>vmax*.55 else DARK)
            else:
                ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,fc="#E8E4DE",ec="white",lw=.5))
    plt.colorbar(im,ax=ax,shrink=.55,label="Uplift (pp)")
    ax.set_title(title,fontsize=10,fontweight="bold",color=DARK,pad=8)
    ax.spines[:].set_visible(False); plt.tight_layout()
    return fig


# ── top bar ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-left">
    <div class="topbar-logo">The Coca&#8209;Cola Company</div>
    <div class="topbar-pipe"></div>
    <div class="topbar-sub">Asset Intelligence</div>
  </div>
  <div class="topbar-sub">Asset Feature Explorer</div>
</div>
""", unsafe_allow_html=True)

# ── navigation ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("Go to", [
        "01 · Overview & Performance",
        "02 · Feature Impact",
        "03 · Combination Explorer",
        "04 · Feature Combinations & OU Impact",
        "05 · Insight Catalog",
        "06 · Rulebook",
    ])
    st.markdown("---")
    st.markdown(f"<small style='color:#888'>{len(df_full):,} assets · {len(results)} precomputed scopes</small>",
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SHARED SCOPE + CAMPAIGN FILTER BAR  (appears on every page)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div style="padding:1.4rem 0 .3rem 0"></div>', unsafe_allow_html=True)
st.markdown('<div class="sec-label">01 &nbsp;·&nbsp; Scope &amp; Campaign</div>', unsafe_allow_html=True)

SCOPE_TYPES = ["Global","OU","Category","Brand","Market"]
if "scope_types" not in st.session_state:
    st.session_state.scope_types = ["Global"]

sc = st.columns([1,1,1,1,1,4])
for i,s in enumerate(SCOPE_TYPES):
    with sc[i]:
        active = s in st.session_state.scope_types
        if st.button(f"· {s}" if active else s, key=f"sb_{s}"):
            if s=="Global":
                st.session_state.scope_types=["Global"]
                for k in ["sel_ou","sel_cat","sel_brand","sel_market"]:
                    st.session_state.pop(k,None)
            else:
                if "Global" in st.session_state.scope_types:
                    st.session_state.scope_types=[]
                if s in st.session_state.scope_types:
                    st.session_state.scope_types.remove(s)
                else:
                    st.session_state.scope_types.append(s)
            st.experimental_experimental_rerun()

scope_filters=[]
KEY_MAP={
    "OU":       ("ou",       sorted(df_full["operating_unit_code"].dropna().unique()),"sel_ou"),
    "Category": ("category", sorted(df_full["category"].dropna().unique()),            "sel_cat"),
    "Brand":    ("brand",    sorted(df_full["brand_name"].dropna().unique()),           "sel_brand"),
    "Market":   ("market",   sorted(df_full["country_name"].dropna().unique()),         "sel_market"),
}
active_types = st.session_state.scope_types
if active_types and "Global" not in active_types:
    vcols=st.columns(len(active_types)+1)
    for i,stype in enumerate(active_types):
        with vcols[i]:
            tk,opts,sk=KEY_MAP[stype]
            sv=st.selectbox(f"Select {stype}",opts,key=sk)
            scope_filters.append((tk,sv))
    # campaign within scope
    sub_pre=df_full.copy()
    for t,v in scope_filters:
        sub_pre=sub_pre[sub_pre[SCOPE_COL[t]]==v]
    cdf=(sub_pre[["campaign_sk_id","campaign_display_name","campaign_code"]]
         .drop_duplicates("campaign_sk_id").sort_values("campaign_display_name"))
    with vcols[len(active_types)]:
        clabels=["All campaigns"]+[f"{r['campaign_display_name']} ({r['campaign_code']})"
                                    for _,r in cdf.iterrows()]
        cids=[None]+cdf["campaign_sk_id"].tolist()
        ci=st.selectbox("Campaign",range(len(clabels)),
                         format_func=lambda i:clabels[i],key="sel_camp")
        sel_camp=cids[ci]
else:
    cdf_g=(df_full[["campaign_sk_id","campaign_display_name","campaign_code"]]
           .drop_duplicates("campaign_sk_id").sort_values("campaign_display_name"))
    clabels_g=["All campaigns"]+[f"{r['campaign_display_name']} ({r['campaign_code']})"
                                   for _,r in cdf_g.iterrows()]
    cids_g=[None]+cdf_g["campaign_sk_id"].tolist()
    ci_g=st.selectbox("Campaign (optional)",range(len(clabels_g)),
                       format_func=lambda i:clabels_g[i],key="sel_camp_g")
    sel_camp=cids_g[ci_g]

# apply filters
sub_df=df_full.copy()
for t,v in scope_filters:
    sub_df=sub_df[sub_df[SCOPE_COL[t]]==v]
if sel_camp:
    sub_df=sub_df[sub_df["campaign_sk_id"]==sel_camp]

scope_key=get_scope_key(scope_filters) if not sel_camp else None
min_n=5 if scope_filters else 10

# chips
if scope_filters:
    lmap={"ou":"OU","category":"Category","brand":"Brand","market":"Market"}
    chips="".join(f'<span class="scope-chip">{lmap[t]}: {v}</span>' for t,v in scope_filters)
    if sel_camp and sel_camp in camp_map.index:
        chips+=f'<span class="scope-chip">Campaign: {camp_map.loc[sel_camp,"campaign_display_name"]}</span>'
    st.markdown(f'<div style="margin:.5rem 0 0 0">{chips}</div>',unsafe_allow_html=True)
else:
    suf=""
    if sel_camp and sel_camp in camp_map.index:
        suf=f' &nbsp;&middot;&nbsp; Campaign: {camp_map.loc[sel_camp,"campaign_display_name"]}'
    st.markdown(f'<p style="font-size:.74rem;color:#CCC8C2;margin:.4rem 0 0 0">Global scope{suf}</p>',
                unsafe_allow_html=True)

st.markdown('<hr class="div">', unsafe_allow_html=True)

if len(sub_df)<3:
    st.warning(f"Only {len(sub_df)} assets in this scope — too few for analysis."); st.stop()

scope_label = ("Global" if not scope_filters
               else " · ".join(f"{t.upper()}={v}" for t,v in scope_filters))
if sel_camp and sel_camp in camp_map.index:
    scope_label += f" · {camp_map.loc[sel_camp,'campaign_display_name']}"


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 01 — OVERVIEW & PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
if page == "01 · Overview & Performance":
    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">Overview &amp; Performance</div>
      <div class="hero-title">How is this Selection performing?</div>
      <div class="hero-sub">A summary of Attention, Persuasion, Likeability and SCD scores
        for the assets in this selection, with scope alerts and top performing assets.</div>
    </div>""", unsafe_allow_html=True)

    # KPI cards — row 1: assets + core 4
    k1,k2,k3,k4,k5 = st.columns(5)
    with k1:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-val">{len(sub_df):,}</div>
          <div class="kpi-lbl">Assets</div>
          <div class="kpi-sub">{sub_df["campaign_sk_id"].nunique():,} campaigns</div>
        </div>""", unsafe_allow_html=True)
    for ml,(mc,col) in zip(
        ["Attention","Persuasion","Likeability","SCD Score"],
        [("Attention_T2B",k2),("Persuasion_T2B",k3),("Likeability_Love_Like_T2B",k4),("SCD_score",k5)]
    ):
        if mc in sub_df.columns:
            mv=sub_df[mc].dropna().mean()*100
            with col:
                st.markdown(f"""<div class="kpi-card" style="border-top:3px solid {RED}">
                  <div class="kpi-val">{mv:.1f}<span style="font-size:1.1rem">pp</span></div>
                  <div class="kpi-lbl">{ml}</div>
                  <div class="kpi-sub">Mean score in scope</div>
                </div>""", unsafe_allow_html=True)

    # KPI cards — row 2: additional metrics
    n1,n2,n3,n4 = st.columns(4)
    for ml,mc,col in [
        ("Experience Recall","Experience_Recall_T2B",n1),
        ("Brand Linkage",    "Brand_Linkage_T2B",    n2),
        ("Uniqueness",       "Uniqueness_T2B",        n3),
        ("Shareability",     "Shareability_T2B",      n4),
    ]:
        if mc in sub_df.columns:
            mv=sub_df[mc].dropna().mean()*100
            with col:
                st.markdown(f"""<div class="kpi-card" style="border-top:3px solid {RED}">
                  <div class="kpi-val">{mv:.1f}<span style="font-size:1.1rem">pp</span></div>
                  <div class="kpi-lbl">{ml}</div>
                  <div class="kpi-sub">Mean score in scope</div>
                </div>""", unsafe_allow_html=True)

    # Glossary
    with st.expander("What do these numbers mean?"):
        st.markdown("""
**pp (percentage points)** — scores are expressed as the percentage of survey respondents who gave a positive answer.
A score of 62pp means 62% of people who watched the ad responded positively to that question.

**Attention** — did the ad successfully grab people's attention? (Top 2 Box survey response)

**Persuasion** — did the ad make people more likely to buy the product? (Top 2 Box)

**Likeability** — did people like the ad — did they love it or like it? (Love + Like combined)

**SCD Score** — a composite score combining all three dimensions: See (awareness/recall, 10%), Connect (likeability/interest, 30%), Do (persuasion/shareability, 60%). Weighted to reflect that conversion is the ultimate goal. Higher is better.

**Uplift** — the difference in average score between ads that have a feature and ads that don't. An uplift of +3pp means ads with that feature score 3 percentage points higher on average.
        """)

    st.markdown('<hr class="div">', unsafe_allow_html=True)

    # Scope alerts
    alerts = get_scope_alerts(scope_filters)
    if alerts:
        with st.expander(f"⚠ {len(alerts)} alert{'s' if len(alerts)>1 else ''} for this scope", expanded=True):
            render_alerts(alerts)

    # Top assets by metric
    st.markdown('<div class="sec-label">Top performing assets</div>', unsafe_allow_html=True)
    st.markdown("""<div class="explain-box">
    Select a metric below to rank assets by that score. Click <strong>▶ View</strong> to see the asset.
    Features listed are the binary elements present in each asset.</div>""",
    unsafe_allow_html=True)

    if "top_asset_metric" not in st.session_state:
        st.session_state.top_asset_metric = "SCD_score"

    btn_cols = st.columns(4)
    metric_btns = [
        ("SCD Score",        "SCD_score",                PURPLE),
        ("Attention",        "Attention_T2B",             BLUE),
        ("Persuasion",       "Persuasion_T2B",            RED),
        ("Likeability",      "Likeability_Love_Like_T2B", GREEN),
    ]
    for col, (lbl, mc_key, mc_col) in zip(btn_cols, metric_btns):
        with col:
            if st.button(lbl, key=f"top_btn_{mc_key}"):
                st.session_state.top_asset_metric = mc_key
                st.experimental_rerun()
    btn_cols2 = st.columns(4)
    metric_btns2 = [
        ("Exp. Recall",  "Experience_Recall_T2B", "#B45309"),
        ("Brand Linkage","Brand_Linkage_T2B",      "#0E7490"),
        ("Uniqueness",   "Uniqueness_T2B",          "#6D28D9"),
        ("Shareability", "Shareability_T2B",        "#065F46"),
    ]
    for col, (lbl, mc_key, mc_col) in zip(btn_cols2, metric_btns2):
        with col:
            if st.button(lbl, key=f"top_btn_{mc_key}"):
                st.session_state.top_asset_metric = mc_key
                st.experimental_rerun()
    all_metric_btns = metric_btns + metric_btns2

    sort_mc = st.session_state.top_asset_metric
    sort_label = {m[1]: m[0] for m in all_metric_btns}[sort_mc]
    top_assets = sub_df[sub_df[sort_mc].notna()].sort_values(sort_mc, ascending=False).head(8)

    for _,r in top_assets.iterrows():
        feats=[FEAT_LABEL.get(f,f) for f in BIN_FEATS if r.get(f)==1]
        url=r.get("asset_url",""); name=r.get("asset_name",f"Asset {r.get('asset_sk_id','')}")
        link=f'<a href="{url}" target="_blank" style="color:{RED};font-size:.72rem;font-weight:600;text-decoration:none">▶ View</a>' if url else ""
        att=f"{r.get('Attention_T2B',0)*100:.0f}" if pd.notna(r.get("Attention_T2B")) else "—"
        pers=f"{r.get('Persuasion_T2B',0)*100:.0f}" if pd.notna(r.get("Persuasion_T2B")) else "—"
        like=f"{r.get('Likeability_Love_Like_T2B',0)*100:.0f}" if pd.notna(r.get("Likeability_Love_Like_T2B")) else "—"
        scd=f"{r.get('SCD_score',0):.2f}" if pd.notna(r.get("SCD_score")) else "—"
        rec=f"{r.get('Experience_Recall_T2B',0)*100:.0f}" if pd.notna(r.get("Experience_Recall_T2B")) else "—"
        lnk=f"{r.get('Brand_Linkage_T2B',0)*100:.0f}" if pd.notna(r.get("Brand_Linkage_T2B")) else "—"
        uniq=f"{r.get('Uniqueness_T2B',0)*100:.0f}" if pd.notna(r.get("Uniqueness_T2B")) else "—"
        shr=f"{r.get('Shareability_T2B',0)*100:.0f}" if pd.notna(r.get("Shareability_T2B")) else "—"
        # highlight the active sort metric
        att_s  = f"font-weight:700;color:{BLUE}"  if sort_mc=="Attention_T2B"              else ""
        pers_s = f"font-weight:700;color:{RED}"   if sort_mc=="Persuasion_T2B"             else ""
        like_s = f"font-weight:700;color:{GREEN}" if sort_mc=="Likeability_Love_Like_T2B"  else ""
        scd_s  = f"font-weight:700;color:{PURPLE}"if sort_mc=="SCD_score"                  else ""
        st.markdown(f"""<div class="asset-card">
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div class="asset-name">{name}</div>{link}
          </div>
          <div class="asset-meta">{r.get("brand_name","")} &nbsp;·&nbsp; {r.get("country_name","")} &nbsp;·&nbsp; {r.get("asset_category","")}</div>
          <div class="asset-meta" style="margin-top:.14rem">Features: {", ".join(feats) if feats else "—"}</div>
          <div style="margin-top:.4rem">
            <span class="asc asc-a" style="{att_s}">Att {att}pp</span>
            <span class="asc asc-p" style="{pers_s}">Pers {pers}pp</span>
            <span class="asc asc-l" style="{like_s}">Like {like}pp</span>
            <span class="asc asc-s" style="{scd_s}">SCD {scd}</span>
            <span class="asc asc-a">Rec {rec}pp</span>
            <span class="asc asc-a">Link {lnk}pp</span>
            <span class="asc asc-a">Uniq {uniq}pp</span>
            <span class="asc asc-a">Share {shr}pp</span>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="div">', unsafe_allow_html=True)

    # ── Asset viewer widget ───────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Asset viewer</div>', unsafe_allow_html=True)
    st.markdown("""<div class="explain-box">
    Select any asset from the current selection to see its full feature profile and scores.</div>""",
    unsafe_allow_html=True)

    av_df = sub_df[sub_df["asset_name"].notna()].copy()
    av_options = av_df["asset_name"].tolist()
    if av_options:
        av_sel = st.selectbox("Select asset", av_options, key="asset_viewer_sel")
        av_row = av_df[av_df["asset_name"] == av_sel].iloc[0]

        av_col1, av_col2 = st.columns([1.2, 1])
        with av_col1:
            # asset info card
            av_url  = av_row.get("asset_url", "")
            av_link = f'<a href="{av_url}" target="_blank" style="display:inline-block;margin-top:.6rem;background:{RED};color:#fff;padding:.35rem .9rem;border-radius:4px;font-size:.82rem;font-weight:600;text-decoration:none">▶ View asset</a>' if av_url else ""
            av_feats_bin = [FEAT_LABEL.get(f,f) for f in BIN_FEATS if av_row.get(f)==1]
            av_feats_cat = {FEAT_LABEL.get(f,f): av_row.get(f) for f in CAT_FEATS
                            if pd.notna(av_row.get(f)) and str(av_row.get(f)).strip() not in ("","nan")}

            st.markdown(f"""<div style="background:#fff;border:1px solid #EAE8E2;border-radius:6px;padding:1.2rem 1.4rem;">
              <div style="font-size:.95rem;font-weight:600;color:#1C1C1C;margin-bottom:.3rem">{av_sel}</div>
              <div style="font-size:.78rem;color:#AAA89E;margin-bottom:.6rem">
                {av_row.get("brand_name","")} &nbsp;·&nbsp; {av_row.get("country_name","")} &nbsp;·&nbsp;
                {av_row.get("campaign_display_name","")}
              </div>
              {av_link}
              <div style="margin-top:1rem">
                <div style="font-size:.68rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:#6A6660;margin-bottom:.4rem">Binary features present</div>
                <div style="font-size:.83rem;color:#3A3830">
                  {", ".join(av_feats_bin) if av_feats_bin else "None detected"}
                </div>
              </div>
              <div style="margin-top:.9rem">
                <div style="font-size:.68rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:#6A6660;margin-bottom:.4rem">Categorical features</div>
                {"".join(f'<div style="display:flex;gap:.5rem;padding:.22rem 0;border-bottom:1px solid #F5F3EF"><span style="font-size:.78rem;color:#888;width:130px;flex-shrink:0">{k}</span><span style="font-size:.82rem;color:#1C1C1C;font-weight:500">{v}</span></div>' for k,v in av_feats_cat.items()) if av_feats_cat else '<div style="font-size:.83rem;color:#AAA">No categorical features</div>'}
              </div>
            </div>""", unsafe_allow_html=True)

        with av_col2:
            # score cards for this asset
            st.markdown('<div style="font-size:.68rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:#6A6660;margin-bottom:.7rem">Scores for this asset</div>', unsafe_allow_html=True)
            for ml, mc_key, mc_col in [
                ("Attention",        "Attention_T2B",             BLUE),
                ("Persuasion",       "Persuasion_T2B",            RED),
                ("Likeability",      "Likeability_Love_Like_T2B", GREEN),
                ("SCD Score",        "SCD_score",                 PURPLE),
                ("Experience Recall","Experience_Recall_T2B",     "#B45309"),
                ("Brand Linkage",    "Brand_Linkage_T2B",         "#0E7490"),
                ("Uniqueness",       "Uniqueness_T2B",             "#6D28D9"),
                ("Shareability",     "Shareability_T2B",           "#065F46"),
            ]:
                val = av_row.get(mc_key)
                if pd.notna(val):
                    disp = f"{val*100:.1f}pp" if mc_key != "SCD_score" else f"{val:.3f}"
                    # compare to scope mean
                    scope_mean = sub_df[mc_key].dropna().mean()
                    diff = val - scope_mean
                    diff_disp = f"{diff*100:+.1f}pp vs scope" if mc_key != "SCD_score" else f"{diff:+.3f} vs scope"
                    diff_col = GREEN if diff > 0 else (RED if diff < 0 else "#AAA")
                    st.markdown(f"""<div style="display:flex;align-items:center;gap:.8rem;padding:.5rem 0;border-bottom:1px solid #F5F3EF">
                      <div style="width:8px;height:8px;border-radius:50%;background:{mc_col};flex-shrink:0"></div>
                      <div style="flex:1;font-size:.82rem;color:#6A6660">{ml}</div>
                      <div style="font-size:1rem;font-weight:700;color:{mc_col}">{disp}</div>
                      <div style="font-size:.75rem;color:{diff_col};min-width:80px;text-align:right">{diff_disp}</div>
                    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page == "02 · Feature Impact":
    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">Feature Impact</div>
      <div class="hero-title">How does each feature affect performance?</div>
      <div class="hero-sub">For each Asset feature, see whether including it in an ad
        is associated with higher or lower scores — globally and within each OU.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="explain-box">
    <strong>How to read this page:</strong> Each bar shows the <em>uplift</em> for that feature —
    the difference in average score between ads that have the feature and ads that don't.
    A bar to the right means the feature is associated with better performance; left means worse.
    Stars show how statistically confident we are: *** = very confident, ** = confident,
    * = directional, ns = not significant (do not brief from ns results).
    The dashed line is always zero — the point where the feature makes no difference.
    </div>""", unsafe_allow_html=True)

    mc_sel=st.selectbox("Metric to display",
                         ["SCD Score","Attention","Persuasion","Likeability",
                          "Experience Recall","Brand Linkage","Uniqueness","Shareability"],
                         key="fi_mc")
    mc_map={"SCD Score":"SCD_score","Attention":"Attention_T2B",
            "Persuasion":"Persuasion_T2B","Likeability":"Likeability_Love_Like_T2B",
            "Experience Recall":"Experience_Recall_T2B","Brand Linkage":"Brand_Linkage_T2B",
            "Uniqueness":"Uniqueness_T2B","Shareability":"Shareability_T2B"}
    mc=mc_map[mc_sel]

    col_chart, col_alerts = st.columns([1.6,1])

    with col_chart:
        rows=[]
        for feat in ALL_FEATS:
            if feat not in sub_df.columns: continue
            u,sig,n=compute_uplift(sub_df,feat,mc)
            if u is not None:
                rows.append({"feat":feat,"label":FEAT_LABEL.get(feat,feat),"uplift":u,"sig":sig,"n":n})
        if rows:
            fd=pd.DataFrame(rows).sort_values("uplift",ascending=False)
            tbl = """<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:.85rem">
            <thead><tr style="border-bottom:2px solid #EAE8E2">
              <th style="text-align:left;padding:.45rem .6rem;font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#6A6660">Feature</th>
              <th style="text-align:center;padding:.45rem .6rem;font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#6A6660">Uplift</th>
              <th style="text-align:center;padding:.45rem .6rem;font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#6A6660">Direction</th>
              <th style="text-align:center;padding:.45rem .6rem;font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#7030A0">Significance</th>
              <th style="text-align:center;padding:.45rem .6rem;font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:#6A6660">n (with feature)</th>
            </tr></thead><tbody>"""
            mc_col_active = M_COLOR.get(mc, DARK)
            max_abs = fd["uplift"].abs().max() or 1
            for i, (_, r) in enumerate(fd.iterrows()):
                bg = "#FAFAF8" if i%2==0 else "#FFFFFF"
                u2 = r["uplift"]
                uc = GREEN if u2>=0 else RED
                arrow = "▲" if u2>=0 else "▼"
                sig2 = r["sig"]
                sig_color = "#7030A0" if sig2 in ("***","**","*") else "#AAA"
                # mini bar
                bar_w = int(abs(u2)/max_abs*80)
                bar_dir = "left" if u2>=0 else "right"
                bar_html_str = (f'<div style="display:flex;align-items:center;gap:.3rem;justify-content:{"flex-start" if u2>=0 else "flex-end"}">'
                                f'<div style="width:{bar_w}px;height:6px;background:{uc};border-radius:2px;opacity:.7"></div>'
                                f'</div>')
                tbl += (f'<tr style="background:{bg};border-bottom:1px solid #F0EEE8">'
                        f'<td style="padding:.42rem .6rem;font-weight:600;color:#1C1C1C">{r["label"]}</td>'
                        f'<td style="text-align:center;font-weight:700;color:{uc};font-size:.9rem">{arrow} {abs(u2):.1f}pp</td>'
                        f'<td style="padding:.42rem .2rem">{bar_html_str}</td>'
                        f'<td style="text-align:center;font-weight:600;color:{sig_color}">{sig2}</td>'
                        f'<td style="text-align:center;color:#888">{r["n"]:,}</td>'
                        f'</tr>')
            tbl += "</tbody></table></div>"
            tbl += f'<div style="font-size:.73rem;color:#AAA;margin-top:.4rem">Sorted by {mc_sel} uplift. pp = percentage points. *** p&lt;0.001, ** p&lt;0.01, * p&lt;0.05, ns = not significant.</div>'
            st.markdown(tbl, unsafe_allow_html=True)
        else:
            st.info("Not enough data to compute feature impact for this scope.")

    with col_alerts:
        st.markdown('<div class="sec-label">Alerts for this scope</div>', unsafe_allow_html=True)
        alerts=get_scope_alerts(scope_filters)
        if alerts:
            render_alerts(alerts, max_items=8)
        else:
            st.markdown('<p style="color:#AAA;font-size:.82rem">No alerts for this scope.</p>',
                        unsafe_allow_html=True)

    # Insight catalog entries for this metric
    if has_ins and catalog is not None:
        st.markdown('<hr class="div">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Insight catalog — top findings for this metric</div>',
                    unsafe_allow_html=True)
        st.markdown("""<div class="explain-box">
        These are statistically validated findings from the full insight pipeline.
        <strong>Confidence</strong> reflects how certain we are: High = safe to brief,
        Medium = directional, Low = do not brief. <strong>Baseline</strong> is the average
        score for assets that do <em>not</em> have this feature — the starting point.</div>""",
        unsafe_allow_html=True)

        smap={"ou":"OU","category":"Category","brand":"Brand","market":"Market"}
        cat_view=catalog[catalog["metric_display"]==mc_sel].copy()
        if scope_filters:
            sl=smap.get(scope_filters[0][0],scope_filters[0][0]); sv=scope_filters[0][1]
            cat_view=cat_view[(cat_view["filter"]==sl)&(cat_view["filter_value"]==sv)]
        else:
            cat_view=cat_view[cat_view["filter"]=="Global"]
        cat_view=cat_view[cat_view["confidence"].isin(["high","medium"])].sort_values(
            "evidence_uplift_pp",ascending=False)

        for _,r in cat_view.head(12).iterrows():
            v=r["evidence_uplift_pp"]; bc=GREEN if v>=0 else RED
            st.markdown(f"""<div class="insight-card" style="border-left-color:{bc}">
              <div style="display:flex;justify-content:space-between">
                <div class="ic-feat">{r["feature_display"]}</div>
                <div>{bpos(v)}</div>
              </div>
              <div class="ic-text">{r["text"]}</div>
              <div class="ic-meta">
                {conf_badge(r["confidence"])}
                {badge(r["metric_display"],"b-metric")}
                {sig_badge(r["evidence_sig"])}
                <span class="badge b-scope">n={int(r["evidence_n_has"]):,}</span>
                <span class="badge b-ns">base {r["evidence_baseline_pp"]:.1f}pp</span>
              </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 03 — COMBINATION EXPLORER  (dashboard_app logic)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "03 · Combination Explorer":
    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">Combination Explorer</div>
      <div class="hero-title">What feature combination drives the highest scores?</div>
      <div class="hero-sub">The best combination is found automatically for each metric.
        Then build and score your own combination in real time below.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="explain-box">
    <strong>How the best combination works:</strong> The algorithm starts with the baseline
    (mean score of all assets in scope) and greedily adds one feature at a time — always
    choosing the next feature that adds the most uplift given what's already selected.
    The <strong>Gain</strong> column shows each feature's individual contribution.
    <strong>Total</strong> is the cumulative uplift from baseline.
    <strong>n</strong> is how many assets match all selected features at that step —
    smaller n means the estimate is less certain.
    <strong>pp</strong> = percentage points (e.g. +8pp on Attention means 8 more percentage
    points of respondents said they noticed the ad).
    </div>""", unsafe_allow_html=True)

    # scope alerts
    alerts=get_scope_alerts(scope_filters)
    if alerts:
        with st.expander(f"⚠ {len(alerts)} scope alert{'s' if len(alerts)>1 else ''}",expanded=False):
            render_alerts(alerts)

    tab_a,tab_p,tab_l = st.tabs(["Attention","Persuasion","Likeability"])
    for mc,tab in zip(["Attention_T2B","Persuasion_T2B","Likeability_Love_Like_T2B"],[tab_a,tab_p,tab_l]):
        with tab:
            ml=METRICS.get(mc,mc)
            combo,steps,baseline_pp,n_total=get_combo(sub_df,scope_key,mc,min_n)

            if not steps:
                st.markdown('<p style="color:#CCC8C2;padding:1.5rem 0;font-size:.84rem">No combination found for this scope.</p>',
                            unsafe_allow_html=True)
                continue

            final_mean=steps[-1]["metric_mean"]
            total_uplift=steps[-1]["cumulative_uplift_pp"]
            final_n=steps[-1]["n"]

            col_wf,col_right=st.columns([1.5,1])
            with col_wf:
                # combo summary card
                st.markdown(f"""
                <div class="combo-card">
                  <div style="display:flex;align-items:flex-end;gap:.9rem;margin-bottom:.25rem">
                    <span class="combo-uplift">+{total_uplift:.1f}</span>
                    <span style="font-family:'Merriweather',serif;font-size:1.3rem;color:#D0C8C0;font-weight:300">pp</span>
                    <div style="padding-bottom:.22rem">
                      <div style="font-size:.74rem;color:#AAA89E">cumulative uplift on {ml}</div>
                      <div style="font-size:.68rem;color:#CCC8C2;margin-top:.08rem">
                        Baseline {baseline_pp:.1f}pp &rarr; {final_mean:.1f}pp &nbsp;&middot;&nbsp;
                        {len(steps)} features &nbsp;&middot;&nbsp; n={final_n:,}
                      </div>
                    </div>
                  </div>""", unsafe_allow_html=True)
                pills='<div class="pill-row">'
                for s in steps:
                    pills+=(f'<div class="pill"><span class="pill-n">{s["step"]}</span>'
                            f'{s["label"]}<span class="pill-gain">+{s["step_gain_pp"]:.1f}pp</span></div>')
                st.markdown(pills+"</div></div>",unsafe_allow_html=True)
                if final_n<10:
                    st.markdown(f'<div class="low-n">Only {final_n} assets match — treat as directional.</div>',
                                unsafe_allow_html=True)

                # waterfall
                st.markdown('<div style="margin-top:1.4rem"></div>',unsafe_allow_html=True)
                st.markdown(f"""
                <div class="wf-row" style="border-bottom:1px solid #EAE8E2;padding-bottom:.2rem">
                  <div class="wf-idx"></div><div class="wf-label wf-col">Feature</div>
                  <div class="wf-bar wf-col">Gain vs previous step</div>
                  <div class="wf-n wf-col">n</div><div class="wf-pp wf-col">Total</div>
                </div>
                <div class="wf-row">
                  <div class="wf-idx">—</div>
                  <div class="wf-label" style="color:#CCC8C2;font-style:italic">Baseline — {n_total:,} assets in scope ({baseline_pp:.1f}pp)</div>
                  <div class="wf-bar">{bar_html(0)}</div>
                  <div class="wf-n">{n_total:,}</div>
                  <div class="wf-pp" style="color:#CCC8C2">{baseline_pp:.1f}</div>
                </div>""",unsafe_allow_html=True)
                mx=max(s["step_gain_pp"] for s in steps) if steps else 1
                for s in steps:
                    frac=s["step_gain_pp"]/mx if mx>0 else 0
                    st.markdown(f"""
                    <div class="wf-row">
                      <div class="wf-idx" style="color:#E8002D;font-weight:600">{s["step"]}</div>
                      <div class="wf-label">+ {s["label"]}</div>
                      <div class="wf-bar">{bar_html(frac)}</div>
                      <div class="wf-n">{s["n"]:,}</div>
                      <div class="wf-pp">+{s["cumulative_uplift_pp"]:.1f}</div>
                    </div>""",unsafe_allow_html=True)

            with col_right:
                # alerts for top combo feature
                if steps:
                    top_f=steps[0]["feat"]
                    st.markdown(f'<div class="sec-label">Alerts — {FEAT_LABEL.get(top_f,top_f)}</div>',
                                unsafe_allow_html=True)
                    fit=get_feature_alerts(top_f,scope_filters,mc)
                    if fit: render_alerts(fit,max_items=5)
                    else: st.markdown('<p style="color:#AAA;font-size:.81rem">No alerts for this feature.</p>',unsafe_allow_html=True)

                # SCD asset sample
                st.markdown('<div style="margin-top:1rem"></div>',unsafe_allow_html=True)
                st.markdown('<div class="sec-label">Top assets matching this combination</div>',unsafe_allow_html=True)
                if combo:
                    sel_tmp=default_sel(combo)
                    mask_tmp=apply_sel(sub_df,sel_tmp)
                    top_tmp=sub_df.loc[mask_tmp].sort_values("SCD_score",ascending=False).head(4)
                    for _,r in top_tmp.iterrows():
                        url=r.get("asset_url",""); nm=r.get("asset_name",f"Asset {r.get('asset_sk_id','')}")
                        lnk=f'<a href="{url}" target="_blank" style="color:{RED};font-size:.71rem;font-weight:600;text-decoration:none">▶ View</a>' if url else ""
                        scd=f"{r['SCD_score']:.2f}" if pd.notna(r.get("SCD_score")) else "—"
                        st.markdown(f"""<div class="asset-card">
                          <div style="display:flex;justify-content:space-between">
                            <div class="asset-name" style="font-size:.76rem">{nm[:50]}</div>{lnk}
                          </div>
                          <div class="asset-meta">SCD {scd} &nbsp;·&nbsp; {r.get("country_name","")}</div>
                        </div>""",unsafe_allow_html=True)

            # ── Feature Explorer ─────────────────────────────────────────────
            st.markdown("""<div class="explorer-card">
              <div class="explorer-title">Explore any combination</div>
              <div class="explorer-sub">
                Pre-filled with the best combination above. Change any feature to see all
                four metrics update in real time. The scoreboard shows your selection's
                average score vs the scope mean. Feature values include their pp uplift
                so you can see which specific values drive performance.
              </div>""",unsafe_allow_html=True)

            ss=f"sel_{mc}_{hash(str(scope_filters))}_{sel_camp}"
            if ss not in st.session_state:
                st.session_state[ss]=default_sel(combo) if combo else {f:"__any__" for f in ALL_FEATS}
            cur=st.session_state[ss]; new=dict(cur); changed=False

            st.markdown('<div class="feat-group">Asset features — select a value to filter</div>',unsafe_allow_html=True)
            rows_f=[ALL_FEATS[i:i+5] for i in range(0,len(ALL_FEATS),5)]
            for row_feats in rows_f:
                cols=st.columns(5)
                for ci,feat in enumerate(row_feats):
                    with cols[ci]:
                        if feat not in sub_df.columns: new[feat]="__any__"; continue
                        raws,disps=get_opts(feat,scope_key,mc,sub_df)
                        cr=cur.get(feat,"__any__")
                        try: ci2=raws.index(cr)
                        except ValueError: ci2=0
                        ch=st.selectbox(FEAT_LABEL.get(feat,feat),range(len(raws)),
                                         format_func=lambda i,d=disps:d[i],index=ci2,
                                         key=f"d_{mc}_{feat}_{hash(str(scope_filters))}_{sel_camp}")
                        chv=raws[ch]; new[feat]=chv
                        if chv!=cr: changed=True
                for ci in range(len(row_feats),5): cols[ci].empty()

            if changed: st.session_state[ss]=new; st.experimental_experimental_rerun()

            r1,r2,_=st.columns([1,1,7])
            with r1:
                if st.button("↺ Reset",key=f"rst_{mc}_{hash(str(scope_filters))}_{sel_camp}"):
                    st.session_state[ss]=default_sel(combo) if combo else {f:"__any__" for f in ALL_FEATS}
                    st.experimental_experimental_rerun()
            with r2:
                if st.button("✕ Clear",key=f"clr_{mc}_{hash(str(scope_filters))}_{sel_camp}"):
                    st.session_state[ss]={f:"__any__" for f in ALL_FEATS}; st.experimental_experimental_rerun()

            # Scoreboard
            scores=score_sel(sub_df,cur)
            bases={mc2:round(sub_df[mc2].dropna().mean()*100,2) for mc2 in ALL_METRICS if mc2 in sub_df.columns}
            sb='<div class="scoreboard">'
            for mc2 in ALL_METRICS:
                if mc2 not in sub_df.columns: continue
                ml2 = M_LABEL.get(mc2, METRICS.get(mc2, mc2))
                val,nv=scores.get(mc2,(None,0)); base=bases.get(mc2,0)
                hi=" hi" if mc2==mc else ""; gc=M_COLOR.get(mc2,DARK)
                if val is not None:
                    d=val-base; dc="d-up" if d>.1 else("d-dn" if d<-.1 else"d-flat")
                    sb+=(f'<div class="score-cell{hi}" style="border-top:2px solid {gc}">'
                         f'<div class="score-metric">{ml2}</div>'
                         f'<div class="score-val">{val:.1f}</div>'
                         f'<div class="score-delta {dc}">{"+" if d>0 else ""}{d:.1f}pp vs mean</div>'
                         f'<div class="score-n">n={nv:,}</div></div>')
                else:
                    sb+=(f'<div class="score-cell{hi}" style="border-top:2px solid {gc}">'
                         f'<div class="score-metric">{ml2}</div>'
                         f'<div class="score-val" style="font-size:.9rem;color:#D8D4CE">—</div>'
                         f'<div class="score-n">n=0</div></div>')
            st.markdown(sb+"</div>",unsafe_allow_html=True)

            _,mn=scores.get(mc,(None,0))
            if mn is not None and 0<mn<10:
                st.markdown(f'<div class="low-n">Only {mn} assets match — directional only.</div>',unsafe_allow_html=True)

            # Insights for active selection
            active_sel=[f for f,v in cur.items() if v not in ("__any__",None,"")]
            if active_sel:
                items=[]
                for f in active_sel[:4]: items+=get_feature_alerts(f,scope_filters,mc,max_items=2)
                if items:
                    st.markdown('<div style="margin-top:.9rem;border-top:1px solid #EAE8E2;padding-top:.9rem">'
                                '<div class="sec-label">Insights for your selection</div>',unsafe_allow_html=True)
                    render_alerts(items[:5]); st.markdown("</div>",unsafe_allow_html=True)

            st.markdown("</div>",unsafe_allow_html=True)  # close explorer-card


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 04 — FEATURE COMBINATIONS & OU IMPACT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "04 · Feature Combinations & OU Impact":
    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">Combinations &amp; OU Impact</div>
      <div class="hero-title">Which features work together, and where?</div>
      <div class="hero-sub">Find the best partner features for any feature you select,
        see the expected combined impact, and compare performance across Operating Units.</div>
    </div>""", unsafe_allow_html=True)

    feat_opts=[(f,FEAT_LABEL.get(f,f)) for f in BIN_FEATS if f in sub_df.columns]
    feat_labels=[fl for _,fl in feat_opts]
    feat_cols=[fc for fc,_ in feat_opts]
    fi=st.selectbox("Select a feature to analyse",range(len(feat_labels)),
                     format_func=lambda i:feat_labels[i],key="combo_feat")
    sel_f=feat_cols[fi]; sel_fl=feat_labels[fi]

    mc_c=st.selectbox("Metric",["SCD Score","Attention","Persuasion","Likeability",
                                 "Experience Recall","Brand Linkage","Uniqueness","Shareability"],key="combo_mc")
    mc_c_col={"SCD Score":"SCD_score","Attention":"Attention_T2B",
               "Persuasion":"Persuasion_T2B","Likeability":"Likeability_Love_Like_T2B",
               "Experience Recall":"Experience_Recall_T2B","Brand Linkage":"Brand_Linkage_T2B",
               "Uniqueness":"Uniqueness_T2B","Shareability":"Shareability_T2B"}[mc_c]

    # Solo uplift
    u_s,sig_s,n_s=compute_uplift(sub_df,sel_f,mc_c_col)

    col_combo,col_ou=st.columns([1.2,1])

    with col_combo:
        # ── Section A: best combinations ─────────────────────────────────────
        st.markdown(f'<div class="sec-label">Part A — Best combinations with {sel_fl}</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""<div class="explain-box">
        <strong>Solo uplift of {sel_fl}:</strong>
        {f"{u_s:+.1f}pp on {mc_c} ({sig_s}, n={n_s:,})" if u_s is not None else "Insufficient data"}<br><br>
        The table below shows which other binary features, when paired with <strong>{sel_fl}</strong>,
        produce the highest combined score. <strong>Combined uplift</strong> is the expected gain
        when an asset has <em>both</em> features vs the scope average.
        <strong>Synergy</strong> is the extra gain from pairing beyond the best individual feature alone —
        a positive synergy means the two features amplify each other.</div>""",
        unsafe_allow_html=True)

        combos=feature_combinations(sub_df,sel_f,mc_c_col,top_n=7)
        if combos:
            tbl="""<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:.83rem">
            <thead><tr style="border-bottom:2px solid #EAE8E2">
              <th style="text-align:left;padding:.48rem .6rem;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:#6A6660">Partner feature</th>
              <th style="text-align:center;padding:.48rem .6rem;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:#6A6660">Combined uplift</th>
              <th style="text-align:center;padding:.48rem .6rem;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:#6A6660">Partner alone</th>
              <th style="text-align:center;padding:.48rem .6rem;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:#7030A0">Synergy</th>
              <th style="text-align:center;padding:.48rem .6rem;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:#6A6660">n assets</th>
              <th style="text-align:left;padding:.48rem .6rem;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:#6A6660">Reading</th>
            </tr></thead><tbody>"""
            for ci,c in enumerate(combos):
                bg="#FAFAF8" if ci%2==0 else "#FFF"
                cu=c["combined"]; sy=c["synergy"]
                cuc=GREEN if cu>=0 else RED; syc=GREEN if sy>=0 else RED
                interp=("Strong synergy — better together" if sy>=1.5
                        else ("Additive — modest extra gain" if sy>=0
                              else "Diminishing — one may suffice"))
                ic="#1A7040" if sy>=1.5 else ("#555" if sy>=0 else "#906820")
                tbl+=(f'<tr style="background:{bg};border-bottom:1px solid #F0EEE8">'
                      f'<td style="padding:.42rem .6rem;font-weight:600">{c["label"]}</td>'
                      f'<td style="text-align:center;font-weight:600;color:{cuc}">{"+" if cu>=0 else ""}{cu:.1f}pp</td>'
                      f'<td style="text-align:center;color:#555">{"+" if c["solo_p"]>=0 else ""}{c["solo_p"]:.1f}pp</td>'
                      f'<td style="text-align:center;font-weight:600;color:{syc}">{"+" if sy>=0 else ""}{sy:.1f}pp</td>'
                      f'<td style="text-align:center;color:#AAA">{c["n"]}</td>'
                      f'<td style="font-size:.77rem;font-style:italic;color:{ic}">{interp}</td></tr>')
            tbl+="</tbody></table></div>"
            tbl+='<div style="font-size:.73rem;color:#AAA;margin-top:.4rem">Combined uplift = mean score when both features present, vs scope mean. Synergy = combined uplift minus the better of the two individual uplifts.</div>'
            st.markdown(tbl,unsafe_allow_html=True)
        else:
            st.info("Not enough data to compute combinations in this scope.")

        # Rulebook alerts for this feature
        st.markdown('<div style="margin-top:1.1rem"></div>',unsafe_allow_html=True)
        st.markdown(f'<div class="sec-label">Rulebook alerts — {sel_fl}</div>',unsafe_allow_html=True)
        fit=get_feature_alerts(sel_f,scope_filters,mc_c_col,max_items=5)
        if fit: render_alerts(fit,max_items=5)
        else: st.markdown('<p style="color:#AAA;font-size:.81rem">No alerts for this feature.</p>',unsafe_allow_html=True)

    with col_ou:
        # ── Section B: OU impact ──────────────────────────────────────────────
        st.markdown(f'<div class="sec-label">Part B — OU impact for {sel_fl}</div>',unsafe_allow_html=True)
        st.markdown(f"""<div class="explain-box">
        The same feature can perform very differently across Operating Units.
        Each bar shows the {mc_c} uplift for <strong>{sel_fl}</strong> within that OU.
        The blue dotted line is the global average. A red bar means the feature is
        associated with <em>lower</em> scores in that OU — critical to check before
        briefing a globally positive feature.</div>""",unsafe_allow_html=True)

        ou_rows=[]
        for ou in sorted(df_full["operating_unit_code"].dropna().unique()):
            ou_sub=df_full[df_full["operating_unit_code"]==ou]
            if scope_filters:
                for t,v in scope_filters:
                    if t!="ou": ou_sub=ou_sub[ou_sub[SCOPE_COL[t]]==v]
            u2,sig2,n2=compute_uplift(ou_sub,sel_f,mc_c_col)
            if u2 is not None: ou_rows.append({"OU":ou,"uplift":u2,"sig":sig2,"n":n2})
        if ou_rows:
            ou_df=pd.DataFrame(ou_rows).sort_values("uplift")
            g_u,_,_=compute_uplift(df_full,sel_f,mc_c_col)
            fig,ax=plt.subplots(figsize=(6,max(3.5,len(ou_df)*.5)))
            fig.patch.set_facecolor(LIGHT); ax.set_facecolor(LIGHT)
            ax.barh(ou_df["OU"],ou_df["uplift"],
                     color=[GREEN if v>=0 else RED for v in ou_df["uplift"]],
                     alpha=.85,height=.62)
            ax.axvline(0,color=DARK,lw=1.2,ls="--")
            if g_u is not None:
                ax.axvline(g_u,color=BLUE,lw=1.5,ls=":",alpha=.75,label=f"Global avg: {g_u:+.1f}pp")
                ax.legend(fontsize=8)
            for i,(_,r) in enumerate(ou_df.iterrows()):
                v=r["uplift"]
                ax.text(v+(.12 if v>=0 else -.12),i,f"{v:+.1f} {r['sig']}",
                        va="center",ha="left" if v>=0 else "right",fontsize=8,color=DARK)
            ax.set_xlabel(f"{sel_fl} uplift on {mc_c} (pp)",fontsize=8.5)
            ax.set_title(f"{sel_fl} — {mc_c} by OU",fontsize=9,fontweight="bold",color=DARK)
            ax.spines[["top","right"]].set_visible(False)
            ax.spines[["left","bottom"]].set_color("#DDD")
            ax.grid(axis="x",alpha=.3); plt.tight_layout()
            st.pyplot(fig); plt.close()

            # best combos per OU
            st.markdown('<div style="margin-top:1rem"></div>',unsafe_allow_html=True)
            st.markdown(f'<div class="sec-label">Best combination per OU — {sel_fl} + top partner</div>',
                        unsafe_allow_html=True)
            st.markdown("""<div class="explain-box" style="font-size:.78rem">
            For each OU, the best partner feature to combine with the selected feature on SCD Score.</div>""",
            unsafe_allow_html=True)
            ou_combo_rows=[]
            for ou in sorted(df_full["operating_unit_code"].dropna().unique()):
                ou_sub2=df_full[df_full["operating_unit_code"]==ou]
                if scope_filters:
                    for t,v in scope_filters:
                        if t!="ou": ou_sub2=ou_sub2[ou_sub2[SCOPE_COL[t]]==v]
                cc=feature_combinations(ou_sub2,sel_f,"SCD_score",top_n=1)
                if cc:
                    ou_combo_rows.append({"OU":ou,"Partner":cc[0]["label"],
                                          "Combined SCD":f'+{cc[0]["combined"]:.1f}pp',
                                          "Synergy":f'+{cc[0]["synergy"]:.1f}pp' if cc[0]["synergy"]>=0
                                                    else f'{cc[0]["synergy"]:.1f}pp',
                                          "n":cc[0]["n"]})
            if ou_combo_rows:
                st.dataframe(pd.DataFrame(ou_combo_rows).set_index("OU"),)
        else:
            st.info("Not enough data for OU breakdown.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 05 — INSIGHT CATALOG
# ══════════════════════════════════════════════════════════════════════════════
elif page == "05 · Insight Catalog":
    if not has_ins or catalog is None:
        st.warning("insight_catalog.csv not found. Run the insight pipeline first."); st.stop()

    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">Insight Catalog</div>
      <div class="hero-title">All validated findings</div>
      <div class="hero-sub">Every finding from the statistical pipeline, tagged with
        evidence and confidence. Filter by scope, feature, metric, or OU.
        High-confidence insights are safe to use in briefs.</div>
    </div>""", unsafe_allow_html=True)

    with st.expander("How to use this catalog"):
        st.markdown("""
**Confidence** tells you how much to trust a finding:
- **High** — significant at ** or ***, at least 50 assets on both sides, no sample size warning. Safe to brief.
- **Medium** — significant at any level, at least 20 assets. Use as directional guidance.
- **Low** — not significant, or very small sample. Do not brief from this.

**Uplift (pp)** — how much higher (▲) or lower (▼) the average score is for assets with this feature vs assets without it, in percentage points.

**Baseline** — the average score of assets that do *not* have this feature. This is the starting point.

**Significance** — *** means less than 1-in-1,000 chance the result is noise. ** = 1-in-100. * = 1-in-20. ns = not significant.

**Quality flags** — small_sample (fewer than 30 assets on one side), multiple_testing_risk (many features tested at once), low_value_presence (very few assets have this feature).
        """)

    # Filters
    cf1,cf2,cf3,cf4,cf5 = st.columns(5)
    with cf1:
        smap={"ou":"OU","category":"Category","brand":"Brand","market":"Market"}
        default_scope="All"
        if scope_filters:
            sl=smap.get(scope_filters[0][0],""); sv=scope_filters[0][1]
            default_scope=f"{sl}: {sv}"
        scope_opts=["All"]+sorted(
            [f"{r['filter']}: {r['filter_value']}" for _,r in
             catalog[["filter","filter_value"]].drop_duplicates().iterrows()
             if r["filter"]!="Global"])
        cat_scope=st.selectbox("Scope",scope_opts,
                                index=scope_opts.index(default_scope) if default_scope in scope_opts else 0)
    with cf2:
        cat_metric=st.multiselect("Metric",sorted(catalog["metric_display"].unique()),
                                   default=sorted(catalog["metric_display"].unique()))
    with cf3:
        cat_conf=st.multiselect("Confidence",["high","medium","low"],default=["high","medium"])
    with cf4:
        cat_dir=st.radio("Direction",["Both","Positive","Negative"])
    with cf5:
        cat_sort=st.selectbox("Sort",["Highest uplift","Lowest uplift","Confidence","Feature"])

    fc=catalog.copy()
    if cat_scope!="All":
        parts=cat_scope.split(": ",1)
        if len(parts)==2: fc=fc[(fc["filter"]==parts[0])&(fc["filter_value"]==parts[1])]
    if cat_metric: fc=fc[fc["metric_display"].isin(cat_metric)]
    if cat_conf:   fc=fc[fc["confidence"].isin(cat_conf)]
    if cat_dir=="Positive": fc=fc[fc["direction"]=="positive"]
    elif cat_dir=="Negative": fc=fc[fc["direction"]=="negative"]
    if cat_sort=="Highest uplift": fc=fc.sort_values("evidence_uplift_pp",ascending=False)
    elif cat_sort=="Lowest uplift": fc=fc.sort_values("evidence_uplift_pp",ascending=True)
    elif cat_sort=="Confidence":
        fc["_co"]=fc["confidence"].map({"high":0,"medium":1,"low":2})
        fc=fc.sort_values(["_co","evidence_uplift_pp"],ascending=[True,False])
    elif cat_sort=="Feature": fc=fc.sort_values("feature_display")

    vmode=st.radio("View",["Cards","Table"])
    st.markdown(f"<span style='color:{MID};font-size:.83rem'>{len(fc):,} insights</span>",unsafe_allow_html=True)

    if vmode=="Cards":
        pg=max(1,int(np.ceil(len(fc)/20)))
        pnum=st.slider("Page",1,pg,1) if pg>1 else 1
        for _,r in fc.iloc[(pnum-1)*20:pnum*20].iterrows():
            v=r["evidence_uplift_pp"]; bc=GREEN if v>=0 else RED
            flags=""
            if r.get("quality_flags",""):
                flags="".join(badge(x.strip(),"b-ns") for x in str(r["quality_flags"]).split(",") if x.strip())

            # OU impact snippet for this feature
            ou_snippet=""
            if has_ins and rulebook is not None:
                ou_het=rulebook[(rulebook["feature"]==r["feature"]) &
                                (rulebook["rule_type"]=="Heterogeneity") &
                                (rulebook["severity"]=="high")]
                if not ou_het.empty:
                    ou_snippet=f'<div style="margin-top:5px;font-size:.76rem;color:#906820">⚠ Reversal in some OUs — check Rulebook before briefing</div>'

            # best combo snippet
            cb=feature_combinations(sub_df,r["feature"],
                                     {"Attention":"Attention_T2B","Persuasion":"Persuasion_T2B",
                                      "Likeability":"Likeability_Love_Like_T2B","SCD Score":"SCD_score",
                                      "Experience Recall":"Experience_Recall_T2B","Brand Linkage":"Brand_Linkage_T2B",
                                      "Uniqueness":"Uniqueness_T2B","Shareability":"Shareability_T2B"}.get(r["metric_display"],"SCD_score"),
                                     top_n=1)
            combo_snippet=""
            if cb:
                combo_snippet=(f'<div style="margin-top:5px;font-size:.76rem;color:#555">'
                               f'Best combination: <strong>{cb[0]["label"]}</strong> → '
                               f'{cb[0]["combined"]:+.1f}pp combined (synergy {cb[0]["synergy"]:+.1f}pp)</div>')

            st.markdown(f"""<div class="insight-card" style="border-left-color:{bc}">
              <div style="display:flex;justify-content:space-between">
                <div class="ic-feat">{r["feature_display"]}</div>
                <div>{bpos(v)}</div>
              </div>
              <div class="ic-text">{r["text"]}</div>
              {ou_snippet}{combo_snippet}
              <div class="ic-meta">
                {conf_badge(r["confidence"])}
                {badge(r["metric_display"],"b-metric")}
                {sig_badge(r["evidence_sig"])}
                <span class="badge b-scope">n={int(r["evidence_n_has"]):,}</span>
                <span class="badge b-ns">base {r["evidence_baseline_pp"]:.1f}pp</span>
                {badge(r.get("filter","")+" "+r.get("filter_value",""),"b-scope")}
                {flags}
              </div>
            </div>""",unsafe_allow_html=True)
    else:
        sc2=["feature_display","metric_display","filter","filter_value",
             "evidence_uplift_pp","evidence_sig","confidence","effect_size",
             "evidence_n_has","evidence_baseline_pp","quality_flags"]
        st.dataframe(fc[[c for c in sc2 if c in fc.columns]].rename(columns={
            "feature_display":"Feature","metric_display":"Metric","filter":"Scope",
            "filter_value":"Scope value","evidence_uplift_pp":"Uplift (pp)",
            "evidence_sig":"Sig.","confidence":"Confidence","effect_size":"Effect size",
            "evidence_n_has":"n (has)","evidence_baseline_pp":"Baseline (pp)",
            "quality_flags":"Flags"}),height=600,)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 06 — RULEBOOK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "06 · Rulebook":
    if not has_ins or rulebook is None:
        st.warning("rulebook.csv not found. Run the insight pipeline first."); st.stop()

    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">Automated Alerts</div>
      <div class="hero-title">Rulebook</div>
      <div class="hero-sub">Seven types of pattern detected automatically across all scopes.
        Review high-severity entries before writing any brief.</div>
    </div>""", unsafe_allow_html=True)

    with st.expander("What does each rule type mean?"):
        st.markdown("""
| Rule | What it means | What to do |
|---|---|---|
| **Conflict** | Feature improves one metric but hurts another | Check the trade-off — choose which metric matters more for this campaign |
| **Heterogeneity** | Feature is positive globally but negative in a specific scope | Never brief from the global number alone — always check the specific OU/market first |
| **Boundary Condition** | Feature works in Video but not Print, or vice versa | Brief this feature only for the format where it works |
| **Opportunity** | High positive effect but the feature appears in fewer than 20% of assets | This feature works — test more assets with it |
| **Outlier** | One scope is unusually different from its peers | Investigate before briefing — the difference may be cultural or a tagging anomaly |
| **Consensus** | Feature is positive across all three metrics | Safest feature to brief — no trade-offs |
| **Anti-pattern** | Feature is negative across all three metrics in a scope | Brief against this feature in this specific scope |
        """)

    rule_counts=rulebook["rule_type"].value_counts()
    RULE_BG={"Conflict":"#FFF0F0","Heterogeneity":"#FFFBF0","Boundary Condition":"#F8F0FF",
              "Opportunity":"#F0FFF4","Outlier":"#EBF3FB","Consensus":"#F0FFF4","Anti-pattern":"#FFF0F0"}
    RULE_AC={"Conflict":RED,"Heterogeneity":"#906820","Boundary Condition":"#7030A0",
              "Opportunity":"#1A7040","Outlier":"#1A3080","Consensus":"#1A7040","Anti-pattern":RED}
    rcols=st.columns(min(len(rule_counts),7))
    for i,(rt,cnt) in enumerate(rule_counts.items()):
        bg=RULE_BG.get(rt,"#F9F9F9"); ac=RULE_AC.get(rt,MID)
        with rcols[i%7]:
            st.markdown(f"""<div style="background:{bg};border:1px solid rgba(0,0,0,.07);
              border-left:3px solid {ac};border-radius:6px;padding:.85rem .95rem;
              margin-bottom:.7rem;text-align:center">
              <div style="font-size:1.4rem;font-weight:700;color:{ac}">{cnt}</div>
              <div style="font-size:.66rem;font-weight:700;color:{ac};
                text-transform:uppercase;letter-spacing:.06em">{rt}</div>
            </div>""",unsafe_allow_html=True)

    rf1,rf2,rf3=st.columns(3)
    with rf1: rt_f=st.multiselect("Rule type",sorted(rulebook["rule_type"].unique()),
                                   default=sorted(rulebook["rule_type"].unique()))
    with rf2: sv_f=st.multiselect("Severity",["high","medium","low"],default=["high","medium"])
    with rf3: sc_f=st.multiselect("Scope",sorted(rulebook["scope"].unique()),
                                   default=sorted(rulebook["scope"].unique()))

    rb_f=rulebook.copy()
    if rt_f: rb_f=rb_f[rb_f["rule_type"].isin(rt_f)]
    if sv_f: rb_f=rb_f[rb_f["severity"].isin(sv_f)]
    if sc_f: rb_f=rb_f[rb_f["scope"].isin(sc_f)]
    st.markdown(f"<span style='color:{MID};font-size:.83rem'>{len(rb_f)} entries</span>",unsafe_allow_html=True)

    SEV_COL={"high":RED,"medium":"#906820","low":"#2A8050"}
    for _,r in rb_f.iterrows():
        bg=RULE_BG.get(r["rule_type"],"#F9F9F9"); ac=RULE_AC.get(r["rule_type"],MID)
        sc2=SEV_COL.get(r.get("severity","medium"),MID)
        st.markdown(f"""<div style="background:{bg};border:1px solid rgba(0,0,0,.07);
          border-left:4px solid {ac};border-radius:6px;padding:11px 15px;margin-bottom:8px">
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div style="font-size:.67rem;font-weight:700;text-transform:uppercase;
              letter-spacing:.08em;color:{ac};margin-bottom:4px">{r["rule_type"]}</div>
            <span style="font-size:.67rem;font-weight:700;color:{sc2};
              text-transform:uppercase">{r.get("severity","").upper()}</span>
          </div>
          <div style="font-size:.83rem;color:#333;line-height:1.5">{r["text"]}</div>
          <div style="display:flex;gap:5px;margin-top:.4rem;flex-wrap:wrap">
            {badge(r.get("feature",""),"b-metric")}
            {badge(r.get("scope",""),"b-scope")}
            {badge(r.get("scope_value","") or "Global","b-ns")}
          </div>
        </div>""",unsafe_allow_html=True)


# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <span>Uplift = mean(feature present) &minus; mean(feature absent) &nbsp;&middot;&nbsp;
    Significance: Mann-Whitney U &nbsp;&middot;&nbsp; pp = percentage points &nbsp;&middot;&nbsp;
    SCD = 10% See + 30% Connect + 60% Do</span>
  <span>The Coca&#8209;Cola Company &nbsp;&middot;&nbsp; Asset Intelligence</span>
</div>
""", unsafe_allow_html=True)
