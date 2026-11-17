"""
Etapa 1 - Seleção, limpeza e integração das cinco tabelas do IMDb.

Entrada : pasta dados_brutos/ com os 5 arquivos TSV (title_basics, title_ratings,
          title_crew, title_principals, name_basics), descompactados.
Saída   : dados/base_integrada.csv (um filme por linha, 15 colunas)

Regras (seções 4.3 e 4.4 do artigo):
  1. ratings: nota e votos numéricos, sem ausências, numVotes >= 100;
     fica com os 20.000 títulos mais votados.
  2. basics: junção interna; titleType = movie e isAdult = 0.
  3. \\N = ausente; ano e duração com to_numeric(errors="coerce");
     exclui quem não tem ano, duração, gêneros, nota ou votos;
     duração entre 50 e 300 minutos (inclusive); sem duplicidade de tconst.
  4. direção e elenco por junção à esquerda; elenco = até 3 ocorrências de
     actor/actress, por ordering; ausência = lista vazia.
"""
from pathlib import Path
import csv

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
BRUTOS = RAIZ / "dados_brutos"
SAIDA = RAIZ / "dados"
SAIDA.mkdir(exist_ok=True)

TAM_BLOCO = 1_000_000
LEITURA = dict(sep="\t", na_values="\\N", dtype=str, quoting=csv.QUOTE_NONE)


def achar(nome):
    """Localiza o arquivo pelo nome (aceita title.basics.tsv, title_basics.tsv, .gz...)."""
    for p in sorted(BRUTOS.rglob("*")):
        n = p.name.lower().replace(".", "_")
        if p.is_file() and p.suffix.lower() != ".zip" and n.startswith(nome):
            return p
    raise FileNotFoundError(
        f"Não encontrei '{nome}' em {BRUTOS}. Coloque os 5 arquivos TSV nessa pasta "
        "(descompacte o ZIP do Kaggle)."
    )


def ler_filtrando(nome, coluna, valores, colunas=None):
    """Lê o arquivo em blocos e guarda só as linhas cujo 'coluna' está em 'valores'."""
    partes = []
    for bloco in pd.read_csv(achar(nome), chunksize=TAM_BLOCO, usecols=colunas, **LEITURA):
        partes.append(bloco[bloco[coluna].isin(valores)])
    return pd.concat(partes, ignore_index=True)


# 1. Avaliações
print("1/5 Lendo title_ratings...")
rat = pd.read_csv(achar("title_ratings"), **LEITURA)
rat["averageRating"] = pd.to_numeric(rat["averageRating"], errors="coerce")
rat["numVotes"] = pd.to_numeric(rat["numVotes"], errors="coerce")
rat = rat.dropna(subset=["averageRating", "numVotes"])
rat = rat[rat["numVotes"] >= 100]
rat = rat.nlargest(20000, "numVotes")
ids = set(rat["tconst"])
print(f"    títulos pré-selecionados: {len(rat)}")

# 2. Informações básicas + junção interna
print("2/5 Lendo title_basics...")
bas = ler_filtrando("title_basics", "tconst", ids)
df = bas.merge(rat, on="tconst", how="inner")
df["isAdult"] = pd.to_numeric(df["isAdult"], errors="coerce")
df = df[(df["titleType"] == "movie") & (df["isAdult"] == 0)]

# 3. Limpeza
df["startYear"] = pd.to_numeric(df["startYear"], errors="coerce")
df["runtimeMinutes"] = pd.to_numeric(df["runtimeMinutes"], errors="coerce")
df = df.dropna(subset=["startYear", "runtimeMinutes", "genres", "averageRating", "numVotes"])
df = df[df["runtimeMinutes"].between(50, 300)]
df = df.drop_duplicates("tconst").reset_index(drop=True)
df["startYear"] = df["startYear"].astype(int)
df["runtimeMinutes"] = df["runtimeMinutes"].astype(int)
df["isAdult"] = df["isAdult"].astype(int)
ids = set(df["tconst"])
print(f"    filmes após filtros: {len(df)}")

# 4. Direção
print("3/5 Lendo title_crew (direção)...")
crew = ler_filtrando("title_crew", "tconst", ids, colunas=["tconst", "directors"])
dir_map = {
    t: [x for x in (d.split(",") if isinstance(d, str) else []) if x]
    for t, d in zip(crew["tconst"], crew["directors"])
}
df["diretores_ids"] = df["tconst"].map(lambda t: dir_map.get(t, []))

# 5. Elenco (actor/actress, até 3 por filme, ordem do arquivo)
print("4/5 Lendo title_principals (elenco) - pode demorar alguns minutos...")
pr = ler_filtrando("title_principals", "tconst", ids,
                   colunas=["tconst", "ordering", "nconst", "category"])
pr = pr[pr["category"].isin(["actor", "actress"])].copy()
pr["ordering"] = pd.to_numeric(pr["ordering"], errors="coerce")
pr = pr.sort_values(["tconst", "ordering"]).groupby("tconst").head(3)
elenco_map = pr.groupby("tconst")["nconst"].apply(list).to_dict()
df["elenco_ids"] = df["tconst"].map(lambda t: elenco_map.get(t, []))

# 6. Nomes (só para leitura/explicação; o vetor usa os identificadores)
print("5/5 Lendo name_basics (nomes)...")
precisa = {n for l in df["diretores_ids"] for n in l} | {n for l in df["elenco_ids"] for n in l}
nomes = ler_filtrando("name_basics", "nconst", precisa, colunas=["nconst", "primaryName"])
nome_de = dict(zip(nomes["nconst"], nomes["primaryName"]))
df["diretores_nomes"] = df["diretores_ids"].map(lambda l: [nome_de.get(n, n) for n in l])
df["elenco_nomes"] = df["elenco_ids"].map(lambda l: [nome_de.get(n, n) for n in l])

colunas = ["tconst", "titleType", "primaryTitle", "originalTitle", "isAdult", "startYear",
           "endYear", "runtimeMinutes", "genres", "averageRating", "numVotes",
           "diretores_ids", "elenco_ids", "diretores_nomes", "elenco_nomes"]
df = df[colunas]
df.to_csv(SAIDA / "base_integrada.csv", index=False)

# Conferência com o artigo (valores esperados entre parênteses)
generos = df["genres"].str.split(",").explode().nunique()
print("\n=== CONFERÊNCIA COM O ARTIGO ===")
print(f"Filmes ................ {len(df)}   (14375)")
print(f"Colunas ............... {df.shape[1]}   (15)")
print(f"Gêneros distintos ..... {generos}   (24)")
print(f"Anos .................. {df.startYear.min()} a {df.startYear.max()}   (1915 a 2022)")
print(f"Menor nº de votos ..... {df.numVotes.min():.0f}   (5288)")
print(f"Sem diretores ......... {(df.diretores_ids.map(len) == 0).sum()}   (3)")
print(f"Sem elenco ............ {(df.elenco_ids.map(len) == 0).sum()}   (336)")
print(f"\nSalvo em: {SAIDA / 'base_integrada.csv'}")
