"""
Analisis aturan asosiasi (Apriori) profil mahasiswa baru terhadap IP semester 1.
Data   : data/Data PMB Siap Apriori.xlsx (sheet "Data Turunan"), baris masuk_analisis == "Ya"
         (dicari di folder data/ di samping skrip atau satu tingkat di atasnya)
Output : Hasil Analisis Apriori.xlsx (di folder yang sama dengan skrip)
Pakai  : python apriori_analisis.py [path_data] [path_output]
Butuh  : pandas, openpyxl, mlxtend, scipy
"""
import sys
from pathlib import Path
import pandas as pd
from itertools import combinations
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.stats import chi2_contingency, fisher_exact, false_discovery_control

HERE = Path(__file__).resolve().parent
_cands = [HERE / "data" / "Data PMB Siap Apriori.xlsx", HERE.parent / "data" / "Data PMB Siap Apriori.xlsx"]
SRC = sys.argv[1] if len(sys.argv) > 1 else next((str(c) for c in _cands if c.exists()), str(_cands[0]))
OUT = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "Hasil Analisis Apriori.xlsx")

MIN_SUP = 0.02      # minimal 2% record (sekitar 27 mahasiswa)
MIN_LIFT = 1.10     # aturan harus menaikkan peluang konsekuen minimal 10%
MAX_LEN = 4         # maksimal 3 item antecedent + 1 item IP
MIN_GAIN = 1.10     # confidence aturan minimal 10% lebih tinggi dari subset antecedent terbaiknya
ALPHA = 0.05

# ---------- 1. Data dan item ----------
d = pd.read_excel(SRC, sheet_name="Data Turunan", dtype={"npm": str})
d = d[d.masuk_analisis == "Ya"].copy()

def items(r):
    t = [f"Jurusan={r.jurusan}", f"Jalur={r.JALUR}", f"Prodi={r.prodi}",
         f"Usia={r.kategori_usia}", f"IP={r.kategori_ip}"]
    if r.status_sekolah in ("Negeri", "Swasta"):
        t.append(f"Sekolah={r.status_sekolah}")
    if r.kelompok_tempat_lahir != "Tidak teridentifikasi":
        t.append("Lahir=" + ("Palembang" if r.kelompok_tempat_lahir == "Kota Palembang" else "Luar Palembang"))
    t.append("Domisili=" + ("Palembang" if r.kelompok_domisili == "Kota Palembang" else "Luar Palembang"))
    return t

d["items"] = d.apply(items, axis=1)

def onehot(tx):
    te = TransactionEncoder()
    return pd.DataFrame(te.fit(tx).transform(tx), columns=te.columns_)

def fmt(s): return ", ".join(sorted(s))

def mine(df_items, min_sup=MIN_SUP, min_lift=MIN_LIFT):
    """Aturan dengan konsekuen tunggal item IP, antecedent tanpa item IP."""
    oh = onehot(list(df_items))
    fi = apriori(oh, min_support=min_sup, use_colnames=True, max_len=MAX_LEN)
    if fi.empty: return pd.DataFrame(), oh
    ar = association_rules(fi, metric="lift", min_threshold=min_lift)
    ar = ar[ar.consequents.apply(lambda c: len(c) == 1 and next(iter(c)).startswith("IP="))]
    ar = ar[ar.antecedents.apply(lambda a: not any(x.startswith("IP=") for x in a))]
    return ar, oh

def enrich(ar, oh):
    """Tambah uji chi-square/Fisher dan buang aturan redundan (tidak lebih baik dari subset-nya)."""
    if ar.empty: return ar
    n = len(oh)
    conf_lookup = {(frozenset(r.antecedents), frozenset(r.consequents)): r.confidence for r in ar.itertuples()}
    rows = []
    for r in ar.itertuples():
        A, C = list(r.antecedents), next(iter(r.consequents))
        a = oh[A].all(axis=1); c = oh[C]
        tab = [[(a & c).sum(), (a & ~c).sum()], [(~a & c).sum(), (~a & ~c).sum()]]
        if min(min(x) for x in tab) < 5:
            p = fisher_exact(tab)[1]; uji = "Fisher"
        else:
            p = chi2_contingency(tab)[1]; uji = "Chi-square"
        # confidence terbaik dari subset antecedent (termasuk himpunan kosong = base rate)
        best = c.mean()
        for k in range(1, len(A)):
            for sub in combinations(A, k):
                v = conf_lookup.get((frozenset(sub), frozenset([C])))
                if v is None:
                    s = oh[list(sub)].all(axis=1)
                    v = (s & c).sum() / s.sum() if s.sum() else 0
                best = max(best, v)
        rows.append({"antecedent": fmt(A), "konsekuen": C, "n_aturan": int(tab[0][0]),
                     "support": r.support, "confidence": r.confidence, "base_rate": c.mean(),
                     "lift": r.lift, "leverage": r.leverage, "conviction": r.conviction,
                     "improvement": r.confidence - best, "conf_subset_terbaik": best, "uji": uji, "p_value": p,
                     "panjang": len(A)})
    out = pd.DataFrame(rows)
    out["q_value_BH"] = false_discovery_control(out.p_value, method="bh")
    out["signifikan"] = out.p_value < ALPHA
    out["signifikan_BH"] = out.q_value_BH < ALPHA
    out["non_redundan"] = out.confidence >= MIN_GAIN * out.conf_subset_terbaik
    return out.sort_values(["konsekuen", "lift"], ascending=[True, False])

def rule_stats(oh, A, C):
    a = oh[A.split(", ")].all(axis=1) if all(x in oh for x in A.split(", ")) else pd.Series(False, index=oh.index)
    c = oh[C] if C in oh else pd.Series(False, index=oh.index)
    n = len(oh); na = a.sum(); nac = (a & c).sum()
    conf = nac / na if na else float("nan"); base = c.mean()
    return pd.Series({"n": int(nac), "support": nac / n, "confidence": conf, "lift": conf / base if base else float("nan")})

# ---------- 2. Analisis utama ----------
ar, oh = mine(d["items"])
rules = enrich(ar, oh)
utama = rules[rules.signifikan & rules.non_redundan].copy()

# ---------- 3. Analisis sensitivitas ----------
sens = []
for s in (0.02, 0.03, 0.05, 0.10):
    a2, o2 = mine(d["items"], min_sup=s)
    r2 = enrich(a2, o2)
    for lf in (1.1, 1.2, 1.3, 1.5):
        k = r2[(r2.lift >= lf) & r2.signifikan & r2.non_redundan] if not r2.empty else r2
        sens.append({"min_support": s, "min_lift": lf, "jumlah_aturan": len(k)})
sens = pd.DataFrame(sens).pivot(index="min_support", columns="min_lift", values="jumlah_aturan")

# ---------- 4. Stabilitas lintas angkatan ----------
stab = []
for r in utama.itertuples():
    row = {"antecedent": r.antecedent, "konsekuen": r.konsekuen, "lift_semua": r.lift}
    for ang in sorted(d.angkatan.unique()):
        oh_a = onehot(list(d[d.angkatan == ang]["items"]))
        st = rule_stats(oh_a, r.antecedent, r.konsekuen)
        row[f"lift_{ang}"] = st.lift; row[f"n_{ang}"] = st.n
    lifts = [row[f"lift_{a}"] for a in sorted(d.angkatan.unique())]
    row["stabil (lift>1 di semua angkatan)"] = all(pd.notna(x) and x > 1 for x in lifts)
    stab.append(row)
stab = pd.DataFrame(stab)

# ---------- 5. Pembanding per fakultas ----------
fak = []
per_fak = {}
for f in ("FIKR", "FEB"):
    sub = d[d.fakultas == f]
    a3, o3 = mine(sub["items"])
    r3 = enrich(a3, o3)
    per_fak[f] = r3[r3.signifikan & r3.non_redundan] if not r3.empty else r3
for r in utama.itertuples():
    row = {"antecedent": r.antecedent, "konsekuen": r.konsekuen, "lift_semua": r.lift}
    for f in ("FIKR", "FEB"):
        st = rule_stats(onehot(list(d[d.fakultas == f]["items"])), r.antecedent, r.konsekuen)
        row[f"lift_{f}"] = st.lift; row[f"n_{f}"] = st.n
    fak.append(row)
fak = pd.DataFrame(fak)

# ---------- 6. Itemset sering (RQ1) ----------
fi_all = apriori(oh, min_support=0.10, use_colnames=True, max_len=4)
fi_all = fi_all[fi_all.itemsets.apply(len) >= 2]
fi_all = fi_all.assign(itemset=fi_all.itemsets.apply(fmt), panjang=fi_all.itemsets.apply(len))[["itemset", "panjang", "support"]]
fi_all = fi_all.sort_values("support", ascending=False)

# ---------- 7. Profil item ----------
prof = oh.mean().rename("support").reset_index().rename(columns={"index": "item"}).sort_values("item")

with pd.ExcelWriter(OUT, engine="openpyxl") as w:
    pd.DataFrame({"Parameter": ["Jumlah record", "min_support", "min_lift", "Panjang maks itemset", "Alpha uji",
                                "Filter aturan utama"],
                  "Nilai": [len(d), MIN_SUP, MIN_LIFT, MAX_LEN, ALPHA,
                            "konsekuen = item IP, p < alpha, confidence >= 1,10 x confidence subset terbaik (tidak redundan)"]}
                 ).to_excel(w, sheet_name="Parameter", index=False)
    prof.to_excel(w, sheet_name="Profil Item", index=False)
    utama.to_excel(w, sheet_name="Aturan Utama", index=False)
    stab.to_excel(w, sheet_name="Stabilitas Angkatan", index=False)
    fak.to_excel(w, sheet_name="Pembanding Fakultas", index=False)
    for f, t in per_fak.items():
        t.to_excel(w, sheet_name=f"Aturan {f}", index=False)
    sens.to_excel(w, sheet_name="Sensitivitas")
    fi_all.to_excel(w, sheet_name="Itemset Sering", index=False)
    rules.to_excel(w, sheet_name="Semua Aturan (lift>=1.1)", index=False)

print(f"Record: {len(d)} | aturan kandidat: {len(rules)} | aturan utama: {len(utama)}")
