"""
Etapa 2 - Modelagem (Modelos A e B) e avaliação Top 5 (seções 4.5 e 4.6 do artigo).

Entrada : dados/base_integrada.csv
Saída   : resultados/metricas.csv e resultados/recomendacoes.csv
"""
from pathlib import Path
import ast

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "resultados"
SAIDA.mkdir(exist_ok=True)

PESO_GENEROS, PESO_NUMERICOS, PESO_EQUIPE = 2.0, 0.35, 0.8
SEMENTE, N_CONSULTAS, TOP = 42, 100, 5

df = pd.read_csv(RAIZ / "dados" / "base_integrada.csv")
for c in ["diretores_ids", "elenco_ids"]:
    df[c] = df[c].apply(ast.literal_eval)
df["generos"] = df["genres"].str.split(",")

# --- Representação vetorial ---
G = MultiLabelBinarizer(sparse_output=True).fit_transform(df["generos"])
numericos = np.column_stack([df["startYear"], df["runtimeMinutes"],
                             df["averageRating"], np.log1p(df["numVotes"])])
Z = csr_matrix(StandardScaler().fit_transform(numericos))
registros = [
    {**{f"diretor_{d}": 1 for d in dirs}, **{f"ator_{a}": 1 for a in elenco}}
    for dirs, elenco in zip(df["diretores_ids"], df["elenco_ids"])
]
E = DictVectorizer().fit_transform(registros)

X_A = hstack([PESO_GENEROS * G, PESO_NUMERICOS * Z], format="csr")
X_B = hstack([PESO_GENEROS * G, PESO_NUMERICOS * Z, PESO_EQUIPE * E], format="csr")
print(f"Modelo A: {X_A.shape[1]} dimensões, {X_A.nnz} não nulos (artigo: 28 e 93.692)")
print(f"Modelo B: {X_B.shape[1]} dimensões, {X_B.nnz} não nulos (artigo: 22.328 e 151.606)")

# --- Consultas ---
rng = np.random.default_rng(SEMENTE)
consultas = rng.choice(len(df), size=N_CONSULTAS, replace=False)


def recomendar(X):
    nn = NearestNeighbors(n_neighbors=TOP + 1, metric="cosine", algorithm="brute").fit(X)
    dist, ind = nn.kneighbors(X[consultas])
    saida = []
    for q, linha_d, linha_i in zip(consultas, dist, ind):
        pares = [(int(j), 1 - float(d)) for j, d in zip(linha_i, linha_d) if j != q]
        saida.append(pares[:TOP])
    return saida


generos = [set(g) for g in df["generos"]]
equipe = [set(d) | set(e) for d, e in zip(df["diretores_ids"], df["elenco_ids"])]


def metricas(resultado):
    cg, se, dg = [], [], []
    for q, pares in zip(consultas, resultado):
        viz = [j for j, _ in pares]
        cg.append(sum(bool(generos[q] & generos[j]) for j in viz) / TOP)
        se.append(sum(bool(equipe[q] & equipe[j]) for j in viz) / TOP)
        dg.append(len(set().union(*(generos[j] for j in viz))))
    return float(np.mean(cg)), float(np.mean(se)), float(np.mean(dg))


res = {"A": recomendar(X_A), "B": recomendar(X_B)}

linhas_m, linhas_r = [], []
for modelo, resultado in res.items():
    cg, se, dg = metricas(resultado)
    linhas_m.append({"Modelo": modelo, "Precisao_genero_5": cg,
                     "Sobreposicao_equipe_5": se, "Diversidade_media": dg})
    for q, pares in list(zip(consultas, resultado))[:5]:
        for pos, (j, sim) in enumerate(pares, start=1):
            linhas_r.append({"Modelo": modelo, "Consulta": df.loc[q, "primaryTitle"],
                             "Posicao": pos, "Recomendacao": df.loc[j, "primaryTitle"],
                             "Similaridade": round(sim, 4)})

pd.DataFrame(linhas_m).to_csv(SAIDA / "metricas.csv", index=False)
pd.DataFrame(linhas_r).to_csv(SAIDA / "recomendacoes.csv", index=False)

print("\n=== CONFERÊNCIA COM O ARTIGO (Tabela 5) ===")
for m in linhas_m:
    print(f"Modelo {m['Modelo']}: CG@5={m['Precisao_genero_5']:.3f}  "
          f"SE@5={m['Sobreposicao_equipe_5']*100:.1f}%  DG@5={m['Diversidade_media']:.2f}")
print("Artigo : A -> CG 1.000, SE 4,0%, DG 2,61 | B -> CG 1.000, SE 24,6%, DG 2,60")
print(f"\nSalvo em: {SAIDA}")
