# Integração de bases IMDb e recomendação de filmes por similaridade de conteúdo

Projeto de Mineração de Dados | Turma T04N | 2026
Professor: Pedro Henrique De Avila Tonin

**Integrantes:** Enzo Zanella, Lucas Mendonça, Leonardo Gabriel, Felipe Jagas

## Descrição

Integração de cinco tabelas do IMDb (Non-Commercial Datasets) em uma base analítica
de 14.375 filmes e construção de um recomendador por similaridade de conteúdo,
organizado pela metodologia CRISP-DM. Foram comparados dois modelos, ambos com
`NearestNeighbors` (distância de cosseno):

- **Modelo A:** gêneros + 4 variáveis numéricas (ano, duração, nota, log(1 + votos)).
- **Modelo B:** Modelo A + direção e até três integrantes do elenco.

Avaliação: 100 consultas sorteadas (`np.random.default_rng(42)`), Top 5 por consulta.

| Indicador | Modelo A | Modelo B |
|---|---|---|
| CG@5 (gênero em comum) | 100,0% | 100,0% |
| SE@5 (profissional em comum) | 4,0% | 24,6% |
| DG@5 (gêneros distintos por lista) | 2,61 | 2,60 |

O artigo completo está em `Artigo_IMDb_ABNT_Fontes_Verificadas.pdf`.

## Estrutura do repositório

```
.
├── README.md
├── requirements.txt
├── Artigo_IMDb_ABNT_Fontes_Verificadas.pdf
├── notebooks/
│   └── pipeline_colab.ipynb        # executa as 3 etapas inteiras no Google Colab
├── src/
│   ├── 01_integracao.py            # seleção, limpeza e junção das 5 tabelas -> dados/base_integrada.csv
│   ├── 02_modelagem_avaliacao.py   # Modelos A e B, Top 5, métricas -> resultados/metricas.csv e recomendacoes.csv
│   ├── 03_graficos.py              # Figuras 1 a 5 -> resultados/graficos/
│   └── 04_conferencia.py           # recalcula as Tabelas 2 a 6 e o Apêndice A e compara com o artigo
├── dados/
│   └── base_integrada.csv          # 14.375 filmes, 15 colunas
└── resultados/
    ├── metricas.csv
    ├── recomendacoes.csv
    ├── conferencia.txt             # relatório OK/DIFERE da etapa 4
    └── graficos/                   # PNG das Figuras 1 a 5
```

## Dados de origem

Os arquivos brutos **não** estão neste repositório por causa do tamanho. Fonte:

- Kaggle: <https://www.kaggle.com/datasets/kunwarakash/imdbdatasets>
- Documentação oficial: <https://data.imdb.com/non-commercial-datasets/>

Tabelas usadas: `title.basics`, `title.ratings`, `title.crew`, `title.principals`, `name.basics`.
Os dados do IMDb são para uso pessoal e não comercial.

## Como reproduzir

1. Instale o Python 3.11 ou superior e as dependências:
   ```
   python -m venv .venv
   .venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
   pip install -r requirements.txt
   ```
2. Baixe o dataset do Kaggle, descompacte e coloque os 5 arquivos TSV na pasta `dados_brutos/`
   (crie a pasta na raiz do projeto; ela é ignorada pelo Git).
3. Execute, na raiz do projeto, na ordem:
   ```
   python src/01_integracao.py
   python src/02_modelagem_avaliacao.py
   python src/03_graficos.py
   python src/04_conferencia.py
   ```
   Os scripts 1 e 2 imprimem uma conferência rápida; o script 4 compara todas as tabelas do artigo.

### Alternativa: Google Colab

Abra `notebooks/pipeline_colab.ipynb` no Colab e execute as células em ordem. Ele baixa os dados
do Kaggle (com o `kaggle.json` do usuário) e roda as mesmas três etapas de `src/`.

## Reprodutibilidade

- Versões da auditoria: Python 3.13.5, NumPy 2.3.5, Pandas 2.2.3, Scikit-Learn 1.8.0, SciPy 1.17.0 (versões da auditoria).
- Semente do sorteio: 42, sobre a ordem original de `base_integrada.csv`.
- SHA-256 de `base_integrada.csv`:
  `ebfd1ac4a3e16c855fea5612aa17032fb73789694ac7e7ed763b3ef38cb7592f`
- Parâmetros: multiplicadores gêneros 2,0; numéricos 0,35; equipe 0,8;
  `metric="cosine"`, `algorithm="brute"`, 6 vizinhos solicitados (a consulta é excluída).

## Limitações

Sem avaliação humana de relevância; métricas circulares (gênero e equipe fazem parte do
vetor e da avaliação); seleção orientada à popularidade (títulos mais votados); API não
implementada (implantação restrita ao notebook).
