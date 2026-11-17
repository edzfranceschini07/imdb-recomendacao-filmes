"""
Etapa 4 - Conferência: recalcula as tabelas do artigo (2 a 6 e Apêndice A) a partir dos
arquivos gerados e compara com os valores publicados.

Entrada : dados/base_integrada.csv, resultados/metricas.csv, resultados/recomendacoes.csv
Saída   : relatório no terminal (OK / DIFERE) e resultados/conferencia.txt
"""
from pathlib import Path
import ast
import io
import re
import unicodedata
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
placar = {"ok": 0, "difere": 0}


def checar(nome, obtido, esperado, tol=0.0):
    try:
        bate = abs(float(obtido) - float(esperado)) <= tol
    except (TypeError, ValueError):
        bate = str(obtido) == str(esperado)
    placar["ok" if bate else "difere"] += 1
    print(f"[{'OK    ' if bate else 'DIFERE'}] {nome}: obtido={obtido}  esperado={esperado}")


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def conferir():
    df = pd.read_csv(RAIZ / "dados" / "base_integrada.csv")
    for c in ["diretores_ids", "elenco_ids"]:
        if c in df.columns:
            df[c] = df[c].apply(ast.literal_eval)
    df["generos"] = df["genres"].str.split(",")

    print("=== Seção 4.3 / 5.1 - Base final ===")
    checar("Filmes", len(df), 14375)
    checar("Colunas", df.shape[1], 15)
    checar("Gêneros distintos", df.generos.explode().nunique(), 24)
    checar("tconst únicos", df.tconst.nunique(), 14375)
    checar("Todos titleType = movie", bool((df.titleType == "movie").all()), True)
    checar("Todos isAdult = 0", bool((df.isAdult == 0).all()), True)

    print("\n=== Tabela 2 - Estatísticas descritivas ===")
    for nome, col, esp in [
        ("Ano", "startYear", dict(min=1915, median=2006, mean=2000.12, max=2022)),
        ("Duração", "runtimeMinutes", dict(min=50, median=104, mean=108.56, max=288)),
        ("Nota", "averageRating", dict(min=1.0, median=6.6, mean=6.55, max=9.7)),
        ("Votos", "numVotes", dict(min=5288, median=17382, mean=61609.51, max=2578898)),
    ]:
        for est, valor in esp.items():
            tol = 0.005 if est == "mean" else 0.0
            checar(f"{nome} - {est}", round(getattr(df[col], est)(), 2), valor, tol)
    checar("Duração - 1º quartil", df.runtimeMinutes.quantile(.25), 94)
    checar("Duração - 3º quartil", df.runtimeMinutes.quantile(.75), 118)

    print("\n=== Seção 5.1 - Direção e elenco ===")
    n_elenco = df.elenco_ids.map(len)
    checar("Elenco com 3", int((n_elenco == 3).sum()), 13958)
    checar("Elenco com 1", int((n_elenco == 1).sum()), 42)
    checar("Elenco com 2", int((n_elenco == 2).sum()), 39)
    checar("Elenco vazio", int((n_elenco == 0).sum()), 336)
    checar("Sem diretores", int((df.diretores_ids.map(len) == 0).sum()), 3)

    print("\n=== Tabela 3 - Gêneros ===")
    longa = df.explode("generos")
    esperado3 = {"Drama": (8094, 6.81), "Comedy": (5132, 6.38), "Action": (3242, 6.31),
                 "Crime": (2862, 6.61), "Romance": (2495, 6.61), "Thriller": (2345, 6.29),
                 "Adventure": (2135, 6.41), "Horror": (1782, 5.86), "Mystery": (1373, 6.38),
                 "Fantasy": (991, 6.24), "Biography": (960, 7.07), "Sci-Fi": (939, 6.10)}
    freq = longa.generos.value_counts()
    media = longa.groupby("generos").averageRating.mean()
    for g, (n, m) in esperado3.items():
        checar(f"{g} - filmes", int(freq.get(g, 0)), n)
        checar(f"{g} - nota média", round(float(media.get(g, np.nan)), 2), m, 0.005)
    checar("Top 12 na ordem do artigo", list(freq.head(12).index), list(esperado3))

    print("\n=== Tabela 4 - Décadas ===")
    df["decada"] = (df.startYear // 10 * 10).astype(int)
    por_dec = df.groupby("decada").averageRating.agg(["size", "mean"])
    esperado4 = {1910: (3, 7.07), 1920: (50, 7.69), 1930: (153, 7.42), 1940: (242, 7.43),
                 1950: (356, 7.36), 1960: (496, 7.22), 1970: (674, 7.05), 1980: (1187, 6.59),
                 1990: (1944, 6.52), 2000: (3487, 6.43), 2010: (4953, 6.40), 2020: (830, 6.29)}
    for d, (n, m) in esperado4.items():
        n_obt = int(por_dec["size"].get(d, 0))
        m_obt = round(float(por_dec["mean"].get(d, np.nan)), 2)
        checar(f"{d} - filmes", n_obt, n)
        checar(f"{d} - nota média", m_obt, m, 0.005)

    print("\n=== Seções 5.4 e 5.5 - Correlações ===")
    lv = np.log1p(df.numVotes)
    checar("Pearson duração x nota", round(df.runtimeMinutes.corr(df.averageRating), 3), 0.309, 0.0005)
    checar("Spearman duração x nota",
           round(df.runtimeMinutes.corr(df.averageRating, method="spearman"), 3), 0.331, 0.0005)
    checar("Pearson ln(1+votos) x nota", round(lv.corr(df.averageRating), 3), 0.214, 0.0005)
    checar("Spearman votos x nota",
           round(df.numVotes.corr(df.averageRating, method="spearman"), 3), 0.171, 0.0005)

    print("\n=== Tabela 5 - Métricas dos modelos ===")
    met = pd.read_csv(RAIZ / "resultados" / "metricas.csv").set_index("Modelo")
    for m, (cg, se, dg) in {"A": (1.0, 0.040, 2.61), "B": (1.0, 0.246, 2.60)}.items():
        checar(f"Modelo {m} - CG@5", round(met.loc[m, "Precisao_genero_5"], 3), cg, 0.0005)
        checar(f"Modelo {m} - SE@5", round(met.loc[m, "Sobreposicao_equipe_5"], 3), se, 0.0005)
        checar(f"Modelo {m} - DG@5", round(met.loc[m, "Diversidade_media"], 2), dg, 0.005)
    checar("Pares com equipe em comum - A (de 500)", round(met.loc["A", "Sobreposicao_equipe_5"] * 500), 20)
    checar("Pares com equipe em comum - B (de 500)", round(met.loc["B", "Sobreposicao_equipe_5"] * 500), 123)

    print("\n=== Tabela 6 e Apêndice A - Recomendações ===")
    esperado_a = {
        "connie": {
            "A": [("Kill Your Friends", .9979), ("Sound of Noise", .9945), ("Diva", .9868),
                  ("The Adventure of Sherlock Holmes' Smarter Brother", .9855), ("Airheads", .9848)],
            "B": [("Kill Your Friends", .8242), ("The Adventure of Sherlock Holmes' Smarter Brother", .8154),
                  ("Diva", .8152), ("Airheads", .8132), ("Be Cool", .8114)]},
        "mcqueen": {
            "A": [("Val", .9993), ("De Palma", .9985), ("Listen to Me Marlon", .9978),
                  ("I Am Greta", .9968), ("Life Itself", .9959)],
            "B": [("I Am Greta", .8948), ("Life Itself", .8935),
                  ("Steve Jobs: The Man in the Machine", .8889), ("Won't You Be My Neighbor?", .8866),
                  ("The Wolfpack", .8865)]},
        "altitude": {
            "A": [("Await Further Instructions", .9990), ("Banshee Chapter", .9982), ("Eden Log", .9979),
                  ("Area 51", .9978), ("What Lies Below", .9975)],
            "B": [("What Lies Below", .8303), ("Area 51", .8299), ("Await Further Instructions", .8297),
                  ("Phoenix Forgotten", .8281), ("Banshee Chapter", .8280)]},
        "cach": {
            "A": [("We Need to Talk About Kevin", .9977), ("First Reformed", .9973), ("The Gift", .9960),
                  ("Copycat", .9957), ("Martha Marcy May Marlene", .9954)],
            "B": [("The White Ribbon", .8644), ("We Need to Talk About Kevin", .8260),
                  ("Tinker Tailor Soldier Spy", .8246), ("The Gift", .8243), ("First Reformed", .8243)]},
        "okkadu": {
            "A": [("Heropanti", .7875), ("Lakshya", .7006), ("Arjun Reddy", .6985),
                  ("Sarpatta Parambarai", .6970), ("I", .6970)],
            "B": [("Heropanti", .6342), ("Ghilli", .6221), ("In Your Name", .5975),
                  ("Lakshya", .5894), ("Arjun Reddy", .5875)]},
    }
    rec = pd.read_csv(RAIZ / "resultados" / "recomendacoes.csv")

    def coluna(padroes, excluir=()):
        for c in rec.columns:
            if c not in excluir and any(norm(c).startswith(p) or p in norm(c) for p in padroes):
                return c
        return None

    c_mod = coluna(["model"])
    c_con = coluna(["consulta", "entrada", "query", "filme"], excluir=[c_mod])
    c_rec = coluna(["recomend", "vizinho", "sugest"], excluir=[c_mod, c_con])
    c_sim = coluna(["sim", "score", "escore"], excluir=[c_mod, c_con, c_rec])
    c_pos = coluna(["posicao", "pos", "rank", "ordem"], excluir=[c_mod, c_con, c_rec, c_sim])
    if None in (c_mod, c_con, c_rec, c_sim):
        print("[DIFERE] Não consegui identificar as colunas de recomendacoes.csv:", list(rec.columns))
        placar["difere"] += 1
        return
    rec["_modelo"] = rec[c_mod].astype(str).str.strip().str[-1].str.upper()
    rec["_consulta"] = rec[c_con].map(norm)
    rec["_pos"] = rec[c_pos] if c_pos else rec.groupby(["_modelo", "_consulta"]).cumcount() + 1

    for chave, por_modelo in esperado_a.items():
        for modelo, lista in por_modelo.items():
            sub = rec[(rec._modelo == modelo) & rec._consulta.str.contains(chave)].sort_values("_pos")
            if sub.empty:
                print(f"[DIFERE] Consulta '{chave}' (modelo {modelo}) não está em recomendacoes.csv")
                placar["difere"] += 1
                continue
            for i, (titulo, sim) in enumerate(lista):
                if i >= len(sub):
                    print(f"[DIFERE] {chave}/{modelo}/#{i+1}: faltando")
                    placar["difere"] += 1
                    continue
                linha = sub.iloc[i]
                checar(f"{chave}/{modelo}/#{i+1} título", norm(linha[c_rec]), norm(titulo))
                checar(f"{chave}/{modelo}/#{i+1} similaridade", round(float(linha[c_sim]), 4), sim, 0.00006)


buf = io.StringIO()
with redirect_stdout(buf):
    conferir()
    print(f"\nRESUMO: {placar['ok']} valores conferem, {placar['difere']} diferem.")
texto = buf.getvalue()
print(texto)
(RAIZ / "resultados").mkdir(exist_ok=True)
(RAIZ / "resultados" / "conferencia.txt").write_text(texto, encoding="utf-8")
