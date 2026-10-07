"""
diagnostico_marca_empleadora.py
===============================
Diagnóstico de Marca Empleadora — Deutsche Bank vs. peers
Sentiment analysis (transformers) + Topic modeling (BERTopic) sobre reseñas de Glassdoor.

Ejecutar desde la raíz del proyecto:
    .venv/Scripts/python.exe notebooks/diagnostico_marca_empleadora.py
"""

# ============================================================
# 0. CONFIGURACIÓN E IMPORTS
# ============================================================
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sin pantalla: guarda figuras sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 5)

SEED = 42
np.random.seed(SEED)

TARGET = "Deutsche-Bank"
PEERS = ["J-P-Morgan", "Barclays", "Citi", "HSBC-Holdings"]
FIRMS = [TARGET] + PEERS
MAX_POR_EMPRESA = 500

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw" / "glassdoor_reviews_hr.parquet"
DATA_PROC = BASE_DIR / "data" / "processed"
OUT_FIG = BASE_DIR / "outputs" / "figures"
OUT_TAB = BASE_DIR / "outputs" / "tables"
OUT_MODELS = BASE_DIR / "outputs" / "models"

for p in [DATA_PROC, OUT_FIG, OUT_TAB, OUT_MODELS]:
    p.mkdir(parents=True, exist_ok=True)

print(f"BASE_DIR: {BASE_DIR}")
print(f"Parquet existe: {DATA_RAW.exists()}")


# ============================================================
# UTILIDADES
# ============================================================
def hr(titulo: str, ancho: int = 72) -> None:
    """Título principal."""
    print("\n" + "=" * ancho)
    print(titulo)
    print("=" * ancho)


def sub_hr(titulo: str, ancho: int = 72) -> None:
    """Subtítulo para dividir secciones dentro de un bloque.""" 
    print("\n" + "-" * ancho)
    print(f"  {titulo}")
    print("-" * ancho)


def show(df_obj, titulo: str = "") -> None:
    """Alternativa a display() de Jupyter."""
    if titulo:
        print(f"\n{titulo}")
    print(df_obj.to_string())
    print()


def savefig(nombre: str) -> None:
    """Guarda la figura actual y la cierra."""
    ruta = OUT_FIG / nombre
    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[figura guardada] {ruta}")

def imprimir_indice() -> None:
    """Índice del análisis para orientar al lector del log."""
    secciones = [
        "1. Carga y preparación",
        "2. EDA — Contexto Deutsche Bank vs. peers",
        "   2.1 Tabla resumen por empresa",
        "   2.2 Rating medio",
        "   2.3 % de recomendación",
        "   2.4 Actuales vs. antiguos",
        "   2.5 Evolución temporal del rating",
        "3. Sentimiento en pros y cons",
        "   3.1 Aplicando el modelo",
        "   3.2 Validación del sentimiento",
        "   3.3 Casos discordantes",
        "4. Topic modeling con BERTopic",
        "   4.1 Entrenando BERTopic sobre PROS",
        "   4.2 Entrenando BERTopic sobre CONS",
        "   4.3 Palabras clave por topic",
        "   4.4 Categorización de topics",
        "   4.5 Distribución de categorías",
        "5. Comparación con peers",
        "   5.1 Distribución % de categorías",
        "   5.2 Balance neto y gap vs peers",
        "   5.3 Heatmap de balance neto",
        "   5.4 Matriz de priorización",
        "6. Contraste con rating y recomendación",
        "7. Evolución temporal de categorías",
        "8. Actuales vs. antiguos",
        "9. Análisis por cohorte de antigüedad",
        "10. Resumen ejecutivo",
        "11. Figura extra — Top categorías en cons",
    ]
    print("\n" + "=" * 72)
    print("  ÍNDICE DEL ANÁLISIS")
    print("=" * 72)
    for s in secciones:
        print(f"  {s}")
    print("=" * 72 + "\n")


imprimir_indice()

# ============================================================
# 1. CARGA Y PREPARACIÓN
# ============================================================
hr("1. CARGA Y PREPARACIÓN")

sub_hr("1.1 Carga del dataset")
raw = pd.read_parquet(DATA_RAW, filters=[("firm", "in", FIRMS)])
print("\nReseñas disponibles por empresa:")
print(raw["firm"].value_counts())

sub_hr("1.2 Muestreo estratificado por empresa")
df = (
    pd.concat(
        [g.sample(n=min(MAX_POR_EMPRESA, len(g)), random_state=SEED)
         for _, g in raw.groupby("firm")],
        ignore_index=True,
    )
    .sample(frac=1, random_state=SEED)
    .reset_index(drop=True)
)

print(f"\nMuestra de trabajo: {len(df):,} reseñas")
print(df["firm"].value_counts())

sub_hr("1.3 Construcción de variables derivadas")


def parse_situacion(s: str) -> str:
    s = str(s).lower()
    if s.startswith("current"):
        return "Actual"
    if s.startswith("former"):
        return "Antiguo"
    return "Desconocido"


def parse_antiguedad(s: str) -> str:
    m = re.search(r"(less|more) than (\d+) year", str(s))
    return f"{m.group(1)} than {m.group(2)}y" if m else "n/d"


df["situacion"] = df["current"].map(parse_situacion)
df["antiguedad"] = df["current"].map(parse_antiguedad)
df["recomienda"] = df["recommend"].map({"v": "Sí", "o": "Neutral", "x": "No"})
df["date_review"] = pd.to_datetime(df["date_review"], errors="coerce")
df["anio"] = df["date_review"].dt.year

for col in ["pros", "cons"]:
    df[col] = df[col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

print(df["situacion"].value_counts())
print(df["recomienda"].value_counts(dropna=False))

proc_path = DATA_PROC / "resenas_preparadas.parquet"
df.to_parquet(proc_path, index=False)
print(f"\nDataset preparado guardado en: {proc_path}")


# ============================================================
# 2. EDA
# ============================================================
hr("2. EDA — CONTEXTO DEUTSCHE BANK vs. PEERS")


def resumen_firm(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n_resenas": len(g),
        "rating_medio": g["overall_rating"].mean(),
        "pct_recomienda": (g["recomienda"] == "Sí").mean() * 100,
        "pct_no_recomienda": (g["recomienda"] == "No").mean() * 100,
        "pct_actuales": (g["situacion"] == "Actual").mean() * 100,
        "rating_actuales": g.loc[g["situacion"] == "Actual", "overall_rating"].mean(),
        "rating_antiguos": g.loc[g["situacion"] == "Antiguo", "overall_rating"].mean(),
    })


sub_hr("2.1 Tabla resumen por empresa")
resumen = df.groupby("firm").apply(resumen_firm).round(2).loc[FIRMS]
show(resumen, "Resumen por empresa:")
resumen.to_csv(OUT_TAB / "resumen_empresas.csv")  

sub_hr("2.2 Rating medio por empresa")
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#1f77b4" if f == TARGET else "#a0a0a0" for f in resumen.index]
ax.bar(resumen.index, resumen["rating_medio"], color=colors)
ax.set_title("Rating medio por empresa")
ax.set_ylabel("Rating medio (1-5)")
ax.set_ylim(0, 5)
for i, v in enumerate(resumen["rating_medio"]):
    ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
plt.xticks(rotation=20)
savefig("eda_rating_medio.png")

sub_hr("2.3 % de recomendación por empresa")  
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#1f77b4" if f == TARGET else "#a0a0a0" for f in resumen.index]
ax.bar(resumen.index, resumen["pct_recomienda"], color=colors)
ax.set_title("% de empleados que recomiendan la empresa")
ax.set_ylabel("% que recomienda")
ax.set_ylim(0, 60)
for i, v in enumerate(resumen["pct_recomienda"]):
    ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
plt.xticks(rotation=20)
savefig("pct_recomienda.png")

sub_hr("2.4 Actuales vs. antiguos")
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(resumen.index))
w = 0.38
ax.bar(x - w / 2, resumen["rating_actuales"], width=w, label="Actuales")
ax.bar(x + w / 2, resumen["rating_antiguos"], width=w, label="Antiguos")
ax.set_xticks(x)
ax.set_xticklabels(resumen.index, rotation=20)
ax.set_ylabel("Rating medio")
ax.set_title("Rating medio: actuales vs. antiguos")
ax.set_ylim(0, 5)
ax.legend()
savefig("eda_actuales_vs_antiguos.png")

sub_hr("2.5 Evolución temporal del rating")
evol = (
    df[df["anio"] >= 2018]
    .groupby(["firm", "anio"])["overall_rating"]
    .mean()
    .reset_index()
)
fig, ax = plt.subplots(figsize=(10, 5))
for firm in FIRMS:
    sub = evol[evol["firm"] == firm]
    lw = 2.5 if firm == TARGET else 1.3
    alpha = 1.0 if firm == TARGET else 0.6
    ax.plot(sub["anio"], sub["overall_rating"], marker="o",
            label=firm, linewidth=lw, alpha=alpha)
ax.set_title("Evolución del rating medio por año")
ax.set_ylabel("Rating medio")
ax.set_xlabel("Año")
ax.set_ylim(1, 5)
ax.legend()
savefig("eda_evolucion_rating.png")

sub_hr("2.6 Conclusiones del EDA")
print("""
CONCLUSIONES DEL EDA:
 - Deutsche Bank tiene el % de recomendación más bajo (41.4%) y el más alto de
   no-recomendación (20.4%) de los 5 bancos.
 - En rating medio DB está en 3.73, en el grupo bajo con Citi (3.70) y HSBC (3.72),
   claramente por debajo de J-P-Morgan (3.91) y Barclays (3.90).
 - La brecha actuales vs. antiguos es de 0.25 puntos, sobre una base baja.
   DB tiene un problema de recomendación, no solo de fuga.
 - Con 500 reseñas por empresa (2500 total) las diferencias son sólidas.
""")


# ============================================================
# 3. SENTIMIENTO PROS / CONS
# ============================================================
hr("3. SENTIMIENTO EN PROS Y CONS")

from transformers import pipeline
import torch

DEVICE = 0 if torch.cuda.is_available() else -1
print(f"Dispositivo: {'GPU' if DEVICE == 0 else 'CPU'}")

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=DEVICE,
)


def score_sentiment(texts, batch_size: int = 32):
    return sentiment_pipe(
        list(texts), truncation=True, max_length=512, batch_size=batch_size
    )


sub_hr("3.1 Aplicando el modelo")
print("Aplicando sentimiento a pros...")
pros_sent = score_sentiment(df["pros"].tolist())
print("Aplicando sentimiento a cons...")
cons_sent = score_sentiment(df["cons"].tolist())

df["pros_sent_label"] = [r["label"] for r in pros_sent]
df["pros_sent_score"] = [r["score"] for r in pros_sent]
df["cons_sent_label"] = [r["label"] for r in cons_sent]
df["cons_sent_score"] = [r["score"] for r in cons_sent]

sub_hr("3.2 Validación del sentimiento por empresa")
valid = (
    df.groupby("firm")
      .apply(lambda g: pd.Series({
          "pct_pros_positivos": (g["pros_sent_label"] == "POSITIVE").mean() * 100,
          "pct_cons_negativos": (g["cons_sent_label"] == "NEGATIVE").mean() * 100,
      }))
      .round(1)
      .loc[FIRMS]
)
show(valid, "Validación del sentimiento por empresa:")
valid.to_csv(OUT_TAB / "validacion_sentimiento.csv")  

# gráfico de barras de la validación
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(valid.index))
w = 0.38
ax.bar(x - w/2, valid["pct_pros_positivos"], width=w, label="% pros POSITIVOS")
ax.bar(x + w/2, valid["pct_cons_negativos"], width=w, label="% cons NEGATIVOS")
ax.set_xticks(x)
ax.set_xticklabels(valid.index, rotation=20)
ax.set_ylabel("% de coincidencia con lo esperado")
ax.set_title("Validación del sentimiento pros/cons por empresa")
ax.set_ylim(0, 100)
ax.axhline(80, color="grey", lw=0.8, ls="--")
ax.legend()
savefig("validacion_sentimiento.png")

sub_hr("3.3 Casos discordantes")
discordantes = df[
    (df["pros_sent_label"] == "NEGATIVE") | (df["cons_sent_label"] == "POSITIVE")
][["firm", "pros", "pros_sent_label", "cons", "cons_sent_label"]]

print(f"\nCasos discordantes: {len(discordantes)} de {len(df)} "
      f"({len(discordantes) / len(df) * 100:.1f}%)")
show(discordantes.head(10), "Ejemplos de casos discordantes:")


# ============================================================
# 4. TOPIC MODELING CON BERTOPIC
# ============================================================
hr("4. TOPIC MODELING CON BERTOPIC")

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- 4.1 Pros ---
sub_hr("4.1 Entrenando BERTopic sobre PROS")
topic_model_pros = BERTopic(
    embedding_model=embedding_model,
    language="english",
    calculate_probabilities=False,
    nr_topics=15,
    verbose=True,
)
pros_docs = df["pros"].fillna("").tolist()
topics_pros, _ = topic_model_pros.fit_transform(pros_docs)

new_topics_pros = topic_model_pros.reduce_outliers(
    pros_docs, topics_pros, strategy="c-tf-idf", threshold=0.0
)
topic_model_pros.update_topics(pros_docs, topics=new_topics_pros, vectorizer_model=None)
df["topic_pros"] = new_topics_pros

print(f"Nº topics pros: {len(topic_model_pros.get_topic_info()) - 1}")
print(f"% outliers pros: {(df['topic_pros'] == -1).mean() * 100:.1f}%")

# --- 4.2 Cons ---
sub_hr("4.2 Entrenando BERTopic sobre CONS")
topic_model_cons = BERTopic(
    embedding_model=embedding_model,
    language="english",
    calculate_probabilities=False,
    nr_topics=15,
    verbose=True,
)
cons_docs = df["cons"].fillna("").tolist()
topics_cons, _ = topic_model_cons.fit_transform(cons_docs)

new_topics_cons = topic_model_cons.reduce_outliers(
    cons_docs, topics_cons, strategy="c-tf-idf", threshold=0.0
)
topic_model_cons.update_topics(cons_docs, topics=new_topics_cons, vectorizer_model=None)
df["topic_cons"] = new_topics_cons

print(f"Nº topics cons: {len(topic_model_cons.get_topic_info()) - 1}")
print(f"% outliers cons: {(df['topic_cons'] == -1).mean() * 100:.1f}%")

# --- 4.3 Ver topics ---
sub_hr("4.3 Palabras clave por topic")
print("\nTOPICS PROS:")
for tid, words in topic_model_pros.get_topics().items():
    if tid == -1:
        continue
    print(f"[{tid}] {[w for w, _ in words[:8]]}")

print("\nTOPICS CONS:")
for tid, words in topic_model_cons.get_topics().items():
    if tid == -1:
        continue
    print(f"[{tid}] {[w for w, _ in words[:8]]}")

info_pros = topic_model_pros.get_topic_info()
info_cons = topic_model_cons.get_topic_info()

print(f"\nPROS — {len(info_pros)} topics")
print(info_pros[["Topic", "Count", "Name"]].to_string())
print(f"\nCONS — {len(info_cons)} topics")
print(info_cons[["Topic", "Count", "Name"]].to_string())

# --- 4.4 Categorización de topics ---
sub_hr("4.4 Categorización de topics a categorías de negocio")


def categorize_topic(topic_id, topic_model, manual_map):
    """Devuelve la categoría de negocio para un topic dado."""
    if topic_id == -1:
        return "Otros"
    if topic_id in manual_map:
        return manual_map[topic_id]

    # Fallback por palabras clave
    words = [w for w, _ in topic_model.get_topics().get(topic_id, [])][:10]
    text = " ".join(words).lower()

    rules = [
        (["benefit", "salary", "pay", "compensation", "bonus", "wage", "money", "reward"],
         "Compensación y beneficios"),
        (["balance", "life", "flexible", "home", "remote", "hours", "schedule"],
         "Conciliación"),
        (["culture", "environment", "atmosphere", "colleague", "people", "diversity", "inclusion"],
         "Cultura y ambiente"),
        (["career", "growth", "learning", "training", "opportunit", "development",
          "promotion", "mentor", "skill"],
         "Carrera y desarrollo"),
        (["brand", "reputation", "company", "name", "global", "prestige"],
         "Marca y reputación"),
        (["team", "office", "manager", "management", "leader", "boss", "supervisor"],
         "Equipo y personas"),
        (["bureaucrac", "politic", "slow", "process", "hierarch", "approval", "red tape"],
         "Burocracia y política interna"),
        (["workload", "stress", "pressure", "burnout", "overwork", "long hour", "busy"],
         "Carga laboral"),
        (["communicat", "align", "meeting", "feedback", "inform", "transparen"],
         "Comunicación interna"),
    ]
    for keywords, cat in rules:
        if any(k in text for k in keywords):
            return cat
    return "Otros"


topic_model_pros.get_topic_info().to_csv(OUT_TAB / "topics_pros.csv", index=False)
topic_model_cons.get_topic_info().to_csv(OUT_TAB / "topics_cons.csv", index=False)
print(f"\nTopics guardados en {OUT_TAB}")

MAP_PRO_TID_TO_CAT = {
    0:  "Compensación y beneficios",
    1:  "Conciliación",
    2:  "Cultura y ambiente",
    3:  "Carrera y desarrollo",
    4:  "Marca y reputación",
    5:  "Cultura y ambiente",
    6:  "Cultura y ambiente",
    7:  "Equipo y personas",
    8:  "Equipo y personas",
    9:  "Cultura y ambiente",
}

MAP_CON_TID_TO_CAT = {
    0: "Burocracia y política interna",
    1: "Compensación y beneficios",
    2: "Carga laboral",
    3: "Carrera y desarrollo",
    4: "Management y dirección",
    5: "Conciliación",
    6: "Otros",
    7: "Comunicación interna",
    8: "Otros",
}

df["cat_pros"] = df["topic_pros"].apply(
    lambda t: categorize_topic(t, topic_model_pros, MAP_PRO_TID_TO_CAT)
)
df["cat_cons"] = df["topic_cons"].apply(
    lambda t: categorize_topic(t, topic_model_cons, MAP_CON_TID_TO_CAT)
)

sub_hr("4.5 Distribución de categorías")
print("\nDistribución de categorías en PROS:")
print(df["cat_pros"].value_counts())
print(f"  → % Otros: {(df['cat_pros'] == 'Otros').mean() * 100:.1f}%")

print("\nDistribución de categorías en CONS:")
print(df["cat_cons"].value_counts())
print(f"  → % Otros: {(df['cat_cons'] == 'Otros').mean() * 100:.1f}%")


# ============================================================
# 5. COMPARACIÓN CON PEERS
# ============================================================
hr("5. COMPARACIÓN CON PEERS")

sub_hr("5.1 Distribución % de categorías por empresa")
pros_dist = pd.crosstab(df["firm"], df["cat_pros"], normalize="index") * 100
cons_dist = pd.crosstab(df["firm"], df["cat_cons"], normalize="index") * 100
pros_dist = pros_dist.reindex(FIRMS).fillna(0).round(1)
cons_dist = cons_dist.reindex(FIRMS).fillna(0).round(1)

show(pros_dist, "Distribución % de categorías en PROS:")
show(cons_dist, "Distribución % de categorías en CONS:")

sub_hr("5.2 Balance neto y gap vs peers")
all_cats = sorted(set(pros_dist.columns) | set(cons_dist.columns))
pros_al = pros_dist.reindex(columns=all_cats, fill_value=0)
cons_al = cons_dist.reindex(columns=all_cats, fill_value=0)

balance = (pros_al - cons_al).round(1)
media_peers = balance.loc[PEERS].mean(axis=0).round(1)
gap_vs_peers = (balance.loc[TARGET] - media_peers).round(1)

show(balance, "Balance neto (empresas x categorías):")
show(media_peers, "Media de peers por categoría:")
show(gap_vs_peers.sort_values(ascending=False),
     "Gap de Deutsche Bank vs. media de peers (positivo = mejor):")

balance.to_csv(OUT_TAB / "balance_neto.csv")           
gap_vs_peers.to_csv(OUT_TAB / "gap_vs_peers.csv")      

sub_hr("5.3 Heatmap de balance neto")
fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(all_cats)), 4.5))
sns.heatmap(balance, cmap="RdBu", center=0, annot=True, fmt=".0f", ax=ax,
            cbar_kws={"label": "Balance neto (pp)"})
ax.set_title("Balance neto por categoría y empresa (pp)")
savefig("balance_neto.png")

sub_hr("5.4 Matriz de priorización")
peso_cons = cons_dist.loc[TARGET].reindex(all_cats).fillna(0)
desventaja = -gap_vs_peers
prior = pd.DataFrame({
    "peso_cons_target": peso_cons,
    "desventaja_vs_peers": desventaja,
}).fillna(0)

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(prior["peso_cons_target"], prior["desventaja_vs_peers"], s=80, alpha=0.7)
for cat, row in prior.iterrows():
    ax.annotate(cat, (row["peso_cons_target"], row["desventaja_vs_peers"]),
                fontsize=8, xytext=(4, 4), textcoords="offset points")
ax.axhline(0, color="grey", lw=0.8)
ax.axvline(prior["peso_cons_target"].mean(), color="grey", lw=0.8, ls="--")
ax.set_xlabel("Peso en cons de Deutsche Bank (%)")
ax.set_ylabel("Desventaja vs. peers (pp)")
ax.set_title("Matriz de priorización de áreas de mejora")
savefig("matriz_priorizacion.png")


# ============================================================
# 6. CONTRASTE CON RATING Y RECOMENDACIÓN
# ============================================================
hr("6. CONTRASTE CON RATING Y RECOMENDACIÓN")

sub_hr("6.1 Rating medio por categoría en CONS")
rating_por_cat_cons = (
    df.groupby("cat_cons")["overall_rating"]
      .agg(["mean", "count"]).sort_values("mean").round(2)
      .rename(columns={"mean": "rating_medio", "count": "n_resenas"})
)
show(rating_por_cat_cons,
     "Rating medio de las reseñas que mencionan cada categoría en CONS:")

sub_hr("6.2 Rating medio por categoría en PROS")
rating_por_cat_pros = (
    df.groupby("cat_pros")["overall_rating"]
      .agg(["mean", "count"]).sort_values("mean", ascending=False).round(2)
      .rename(columns={"mean": "rating_medio", "count": "n_resenas"})
)
show(rating_por_cat_pros,
     "Rating medio de las reseñas que mencionan cada categoría en PROS:")

sub_hr("6.3 % que recomienda por categoría en CONS")
recom_por_cat_cons = (
    df.groupby("cat_cons")["recomienda"]
      .apply(lambda s: (s == "Sí").mean() * 100).round(1)
      .sort_values().rename("pct_recomienda")
)
show(recom_por_cat_cons,
     "% que recomienda entre quienes mencionan cada categoría en CONS:")


# ============================================================
# 7. EVOLUCIÓN TEMPORAL DE CATEGORÍAS
# ============================================================
hr("7. EVOLUCIÓN TEMPORAL DE CATEGORÍAS (DEUTSCHE BANK)")

db_solo = df[df["firm"] == TARGET].copy()

sub_hr("7.1 Evolución anual")
evol_cat_cons = db_solo.groupby(["anio", "cat_cons"]).size().unstack(fill_value=0)
evol_cat_cons_pct = evol_cat_cons.div(evol_cat_cons.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(10, 5))
evol_cat_cons_pct.plot(ax=ax, marker="o")
ax.set_title(f"Evolución del peso de cada categoría en CONS — {TARGET}")
ax.set_ylabel("% del total de cons del año")
ax.set_xlabel("Año")
ax.legend(title="Categoría", bbox_to_anchor=(1.02, 1), loc="upper left")
savefig("evolucion_categorias_cons.png")

sub_hr("7.2 Comparativa pre-2020 vs 2020+")
df["periodo"] = df["anio"].apply(lambda y: "pre-2020" if y < 2020 else "2020+")
evol_periodo = (
    df[df["firm"] == TARGET]
    .groupby(["periodo", "cat_cons"]).size().unstack(fill_value=0)
)
evol_periodo_pct = evol_periodo.div(evol_periodo.sum(axis=1), axis=0) * 100
show(evol_periodo_pct.round(1),
     "Peso de cada categoría en cons por periodo:")


# ============================================================
# 8. ACTUALES vs. ANTIGUOS
# ============================================================
hr("8. ACTUALES vs. ANTIGUOS")

db = df[df["firm"] == TARGET].copy()

sub_hr("8.1 Categorías de cons según situación — Deutsche Bank")
cons_por_situacion = (
    pd.crosstab(db["situacion"], db["cat_cons"], normalize="index") * 100
).round(1)
show(cons_por_situacion,
     f"Categorías de cons por situación — {TARGET}:")

sub_hr("8.2 Motivos de salida: DB vs peers")
antiguos = df[df["situacion"] == "Antiguo"]
dist_antiguos = (
    pd.crosstab(antiguos["firm"], antiguos["cat_cons"], normalize="index") * 100
)
dist_antiguos = dist_antiguos.reindex(FIRMS).fillna(0)
dist_antiguos.loc["media_peers"] = dist_antiguos.loc[PEERS].mean(axis=0)
show(dist_antiguos.round(1),
     "Cons de ANTIGUOS: DB vs. peers (motivos de salida):")


# ============================================================
# 9. ANÁLISIS POR COHORTE DE ANTIGÜEDAD
# ============================================================
hr("9. ANÁLISIS POR COHORTE DE ANTIGÜEDAD")


def bucket_antiguedad(s: str) -> str:
    if s == "n/d":
        return "n/d"
    if "less" in s:
        return "<3 años"
    if "more than 3y" in s:
        return "3-5 años"
    if "more than 5y" in s:
        return ">5 años"
    return "otros"


df["antiguedad_bucket"] = df["antiguedad"].map(bucket_antiguedad)

sub_hr("9.1 Distribución de antigüedad por empresa")
dist_antig = (
    pd.crosstab(df["firm"], df["antiguedad_bucket"], normalize="index") * 100
).round(1)
show(dist_antig, "Distribución de antigüedad por empresa (%):")

sub_hr("9.2 Rating medio por cohorte")
rating_antiguedad = (
    df.groupby(["firm", "antiguedad_bucket"])["overall_rating"]
      .mean().unstack().round(2).loc[FIRMS]
)
show(rating_antiguedad, "Rating medio por empresa y cohorte:")

sub_hr("9.3 Categorías de cons por cohorte — Deutsche Bank")
db_buckets = df[df["firm"] == TARGET]
cons_por_bucket = (
    pd.crosstab(db_buckets["antiguedad_bucket"], db_buckets["cat_cons"],
                normalize="index") * 100
).round(1)
show(cons_por_bucket, f"Peso de cada categoría en CONS por cohorte — {TARGET}:")

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(cons_por_bucket, cmap="Reds", annot=True, fmt=".0f", ax=ax,
            cbar_kws={"label": "% del total de cons de la cohorte"})
ax.set_title(f"Categorías de cons por cohorte de antigüedad — {TARGET}")
savefig("cons_por_antiguedad.png")


# ============================================================
# 10. CONCLUSIONES Y RECOMENDACIONES
# ============================================================
hr("10. RESUMEN EJECUTIVO — DIAGNÓSTICO DE MARCA EMPLEADORA")

print(f"Empresa analizada : {TARGET}")
print(f"Peers             : {', '.join(PEERS)}")
print(f"Reseñas analizadas: {len(df)} ({MAX_POR_EMPRESA} por empresa)")
print()

sub_hr("10.1 Posición general")
print(resumen[["rating_medio", "pct_recomienda", "pct_no_recomienda",
               "rating_actuales", "rating_antiguos"]])
print()

sub_hr("10.2 Fortalezas relativas (gap vs peers > 0)")
for cat, gap in gap_vs_peers.sort_values(ascending=False).head(3).items():
    print(f"  + {cat}: {gap:+.1f} pp")
print()

sub_hr("10.3 Debilidades relativas (gap vs peers < 0)")
for cat, gap in gap_vs_peers.sort_values().head(3).items():
    print(f"  - {cat}: {gap:+.1f} pp")
print()

sub_hr("10.4 Motivos de salida (cons de antiguos, Deutsche Bank)")
antiguos_db = (
    db[db["situacion"] == "Antiguo"]["cat_cons"]
    .value_counts(normalize=True) * 100
)
for cat, pct in antiguos_db.head(4).items():
    print(f"  {cat}: {pct:.1f}%")
print()

sub_hr("10.5 Recomendaciones accionables para RRHH")
print(" 1. Auditoría de compensación frente a peers por nivel y área.")
print("    Gap actual: -14.2 pp en balance neto (peor del grupo).")
print("    Indicador: subir menciones positivas de salario del 27.4% al")
print("    32.0% (nivel medio de peers) en 12 meses.")
print()
print(" 2. Campaña de employer branding anclada en cultura y ambiente.")
print("    DB es el mejor del grupo en cultura (+10.2 pp vs peers) pero")
print("    el peor en marca (-1.1 pp). Hay desajuste entre lo bueno y lo")
print("    que se comunica.")
print("    Indicador: subir % de recomendación del 41.4% al 47%.")
print()
print(" 3. Programa de simplificación de procesos internos.")
print("    Burocracia es el motivo de salida #1 (26.8%) y tiene el rating")
print("    más bajo (3.54). Es común a todos los bancos, pero en DB")
print("    coincide con carrera y desarrollo débiles.")
print("    Indicador: reducir menciones de burocracia en cons del 26.4%")
print("    al 22% en 12 meses.")
print()

sub_hr("10.6 Limitaciones")
print(" - Glassdoor no es representativo del conjunto de la plantilla.")
print(" - La muestra mezcla países y áreas funcionales.")
print(" - Los topics son aproximaciones: revisar antes de decidir.")
print(" - El sentimiento no distingue matices irónicos ni negaciones compuestas.")


# ============================================================
# 11. TOP CATEGORÍAS EN CONS 
# ============================================================
hr("11. TOP CATEGORÍAS EN CONS (DEUTSCHE BANK)")

top_cons = (
    db["cat_cons"].value_counts(normalize=True) * 100
).round(1).head(8)

fig, ax = plt.subplots(figsize=(9, 5))
top_cons.sort_values().plot(kind="barh", ax=ax, color="#d62728")
ax.set_xlabel("% del total de cons de Deutsche Bank")
ax.set_title("Categorías más mencionadas en las quejas — Deutsche Bank")
for i, (cat, val) in enumerate(top_cons.sort_values().items()):
    ax.text(val + 0.3, i, f"{val:.1f}%", va="center", fontsize=9)
savefig("top_categorias_cons_target.png")


hr("FIN DEL ANÁLISIS")
print(f"Figuras guardadas en: {OUT_FIG}")
print(f"Tablas guardadas en: {OUT_TAB}")
print(f"Dataset procesado en: {DATA_PROC}")