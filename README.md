# titanic-kaggle

Competição [Titanic](https://www.kaggle.com/competitions/titanic) do Kaggle: prever quais passageiros sobreviveram (`Survived`, 0 ou 1). A métrica é a **acurácia** no teste de 418 passageiros, e o envio é um CSV `PassengerId,Survived`.

O projeto compara um baseline de uma linha com três modelos, avalia **sem vazamento** por validação cruzada estratificada repetida e gera o CSV de envio já validado. Ele **não envia nada ao Kaggle**: o envio é sempre um passo manual.

> **Estado em 04/11/2026:** código, testes (dados sintéticos) e avaliação nos dados reais prontos ([resultados](#resultados)). Primeiro envio ao Kaggle (logística): **placar público 0,76794** (321 acertos em 418), bem abaixo dos 0,8278 da validação cruzada.

## Uso

```bash
uv sync --extra dev
uv run pytest

# 1. Entre na competição no site (Join Competition, aceitando as regras) e baixe os dados.
uvx --with six kaggle competitions download -c titanic -p data
unzip -o data/titanic.zip -d data

# 2. Primeira olhada: faltantes e taxa de sobrevivência por grupo.
uv run titanic resumo

# 3. Compara os modelos (5 dobras x 5 repetições, as mesmas dobras para todos).
uv run titanic avaliar

# 4. Treina no treino todo e gera saidas/submission.csv (modelo = o melhor da validação cruzada).
uv run titanic gerar
uv run titanic gerar --modelo logistica   # ou escolha o modelo
```

Para enviar (decisão sua, uma vez por pedido):

```bash
uvx --with six kaggle competitions submit titanic -f saidas/submission.csv -m "descrição"
```

## Como funciona

- **Baseline** (`regra_genero`): mulher sobrevive, homem não. Qualquer modelo precisa bater isso.
- **Features** (`features.py`): `Titulo` extraído do nome, `TamanhoFamilia`, `Sozinho`, `Deck` da cabine, `IdadeFaltante`, `LogFare`, classe e porto como categorias.
- **Sem vazamento:** as features são calculadas linha a linha. A mediana de idade por (título, classe), a categoria mais frequente e a escala são ajustadas **dentro do Pipeline**, só com o treino de cada dobra.
- **Modelos:** regressão logística, Random Forest e gradient boosting (`HistGradientBoostingClassifier`).
- **Avaliação:** `RepeatedStratifiedKFold` com semente fixa; `desvio` é entre dobras e `ganho` é a diferença pareada contra o baseline.

## Resultados

Acurácia nos 891 passageiros do treino, validação cruzada estratificada (5 dobras x 5 repetições, `uv run titanic avaliar`, 04/11/2026):

| modelo | acurácia média | desvio entre dobras | ganho sobre a regra de gênero |
|---|---|---|---|
| `logistica` | 0,8278 | 0,0244 | +0,0411 |
| `boosting` | 0,8263 | 0,0177 | +0,0395 |
| `floresta` | 0,8256 | 0,0214 | +0,0388 |
| `regra_genero` (baseline) | 0,7868 | 0,0263 | 0,0000 |

- Os três modelos ficam a 0,2 ponto um do outro, com desvio de 2 pontos entre dobras: **não dá para dizer qual é melhor**. Todos ganham cerca de 4 pontos do baseline.
- `titanic gerar` escolheu a logística (a maior média) e previu 169 sobreviventes entre os 418 do teste; concorda com a regra de gênero em 91% das linhas.
- Essas são notas de **validação**. O placar público do Kaggle usa só as 418 linhas do teste.
- **Placar público do envio da logística (04/11/2026): 0,76794**, ou seja, 321 acertos em 418. Ficou 6 pontos abaixo da validação cruzada. O erro padrão de uma acurácia medida em 418 linhas é de cerca de 2 pontos, e o teste não é igual ao treino; por isso esse placar **não separa** o modelo da regra de gênero, apesar dos 4 pontos de ganho na validação.
- Medir só pelo placar público é pouco confiável: com 10 envios por dia dá para "ajustar" o modelo ao teste sem aprender nada. A validação cruzada continua sendo a régua para decidir o que mudar.

## Dados e regras

`data/` e `saidas/` ficam fora do git. Os arquivos da competição pertencem ao Kaggle e têm regras de uso: não os versione nem os redistribua. Os testes usam dados **sintéticos** gerados em `tests/conftest.py`.
