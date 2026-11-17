"""
Etapa 3 - Gráficos (Figuras 1 a 5 do artigo).

Entrada : dados/base_integrada.csv e resultados/metricas.csv
Saída   : resultados/graficos/*.png
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # só salva os arquivos, sem abrir janelas
import matplotlib.pyplot as plt
import seaborn as sns

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "resultados" / "graficos"
SAIDA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAIZ / "dados" / "base_integrada.csv")
df["generos"] = df["genres"].str.split(",")
sns.set_theme(style="whitegrid")

# Figura 1 - gêneros mais frequentes
freq = df.explode("generos").generos.value_counts().head(12)
plt.figure(figsize=(10, 5))
sns.barplot(x=freq.values, y=freq.index)
plt.title("Generos mais frequentes")
plt.xlabel("Quantidade de filmes")
plt.tight_layout()
plt.savefig(SAIDA / "grafico_generos.png", dpi=160)
plt.close()

# Figura 2 - nota média por década
df["decada"] = (df.startYear // 10 * 10).astype(int)
decadas = df.groupby("decada").averageRating.mean()
plt.figure(figsize=(10, 5))
sns.lineplot(x=decadas.index, y=decadas.values, marker="o")
plt.title("Nota media por decada")
plt.ylabel("Nota media")
plt.tight_layout()
plt.savefig(SAIDA / "grafico_decadas.png", dpi=160)
plt.close()

# Figura 3 - duração x avaliação
plt.figure(figsize=(9, 5))
sns.scatterplot(data=df, x="runtimeMinutes", y="averageRating", alpha=0.25, s=15)
plt.title("Duracao x avaliacao")
plt.tight_layout()
plt.savefig(SAIDA / "grafico_duracao.png", dpi=160)
plt.close()

# Figura 4 - popularidade x avaliação
plt.figure(figsize=(9, 5))
sns.scatterplot(x=np.log1p(df.numVotes), y=df.averageRating, alpha=0.25, s=15)
plt.title("Popularidade x avaliacao")
plt.xlabel("log(1 + numero de votos)")
plt.tight_layout()
plt.savefig(SAIDA / "grafico_votos.png", dpi=160)
plt.close()

# Figura 5 - comparativo dos modelos
metricas = pd.read_csv(RAIZ / "resultados" / "metricas.csv")
percentuais = metricas["Sobreposicao_equipe_5"] * 100
rotulos = "Modelo " + metricas["Modelo"].astype(str)
fig, ax = plt.subplots(figsize=(8, 5))
barras = ax.bar(rotulos, percentuais)
ax.set_title("Sobreposição de profissionais nas recomendações")
ax.set_xlabel("Modelo")
ax.set_ylabel("Recomendações com profissional em comum (%)")
ax.set_ylim(0, max(30, float(percentuais.max()) * 1.2))
ax.bar_label(barras, labels=[f"{v:.1f}%".replace(".", ",") for v in percentuais], padding=4)
fig.tight_layout()
fig.savefig(SAIDA / "grafico_comparacao_modelos.png", dpi=160)
plt.close(fig)

print("Gráficos salvos em:", SAIDA)
for p in sorted(SAIDA.glob("*.png")):
    print(" -", p.name)
