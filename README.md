# titanic-kaggle

Competição [Titanic](https://www.kaggle.com/competitions/titanic) do Kaggle: prever quais passageiros sobreviveram (`Survived`, 0 ou 1). A métrica é a **acurácia** no teste de 418 passageiros, e o envio é um CSV `PassengerId,Survived`.

O projeto compara um baseline de uma linha com três modelos, cada um com e sem features de grupo (bilhete e família), avalia **sem vazamento** por validação cruzada estratificada repetida e gera o CSV de envio já validado. Ele **não envia nada ao Kaggle**: o envio é sempre um passo manual.

> **Estado em 04/11/2026:** código, testes (dados sintéticos) e avaliação nos dados reais prontos ([resultados](#resultados)). Primeiro envio ao Kaggle (logística, sem grupos): **placar público 0,76794** (321 acertos em 418), bem abaixo dos 0,8278 da validação cruzada. As features de grupo deram +0,8 a +1,3 ponto na validação cruzada; enviada (`floresta+grupo`), ela fez **0,80143 de placar público** (335 acertos em 418), 14 acertos a mais que a logística.

## Uso

```bash
uv sync --extra dev
uv run pytest

# 1. Entre na competição no site (Join Competition, aceitando as regras) e baixe os dados.
uvx --with six kaggle competitions download -c titanic -p data
unzip -o data/titanic.zip -d data

# 2. Primeira olhada: faltantes e taxa de sobrevivência por grupo.
uv run titanic resumo

# 3. Compara os 9 modelos e o baseline (5 dobras x 5 repetições, as mesmas dobras para todos)
#    e mostra o efeito dos grupos dobra a dobra.
uv run titanic avaliar

# 4. Treina no treino todo e gera saidas/submission.csv (modelo = o de maior média na validação).
uv run titanic gerar
uv run titanic gerar --modelo logistica   # ou escolha o modelo (ex.: floresta+grupo)
```

Para enviar (decisão sua, uma vez por pedido):

```bash
uvx --with six kaggle competitions submit titanic -f saidas/submission.csv -m "descrição"
```

## Como funciona

- **Baseline** (`regra_genero`): mulher sobrevive, homem não. Qualquer modelo precisa bater isso.
- **Features** (`features.py`): `Titulo` extraído do nome, `TamanhoFamilia`, `Sozinho`, `Deck` da cabine, `IdadeFaltante`, `LogFare`, classe e porto como categorias.
- **Sem vazamento:** as features são calculadas linha a linha. A mediana de idade por (título, classe), a categoria mais frequente e a escala são ajustadas **dentro do Pipeline**, só com o treino de cada dobra.
- **Modelos:** regressão logística, Random Forest e gradient boosting (`HistGradientBoostingClassifier`), cada um em 3 variantes: sem grupos, `+tam` (quantos passageiros dividem o bilhete e a família, contados em treino + teste, sem usar rótulo) e `+grupo` (`+tam` mais o que aconteceu com os **outros** do bilhete e da família).
- **Sobrevivência do grupo sem vazar** (`grupos.py`): no treino, cada linha é codificada só com os rótulos de outras dobras internas (como o teste do Kaggle vê o treino); em linhas novas usa todos os rótulos do treino. Codificar o treino com o próprio rótulo (ou por leave-one-out) faria o modelo decorar a resposta; os testes cobrem isso. A família é sobrenome + classe, e a taxa é suavizada em direção à taxa geral.
- **Avaliação:** `RepeatedStratifiedKFold` com semente fixa; `desvio` é entre dobras e `ganho` é a diferença pareada contra o baseline.

## Resultados

Acurácia nos 891 passageiros do treino, validação cruzada estratificada (5 dobras x 5 repetições, `uv run titanic avaliar`, 04/11/2026):

| modelo | acurácia média | desvio entre dobras | ganho sobre a regra de gênero |
|---|---|---|---|
| `floresta+grupo` | 0,8384 | 0,0250 | +0,0516 |
| `boosting+grupo` | 0,8357 | 0,0219 | +0,0489 |
| `logistica+grupo` | 0,8355 | 0,0235 | +0,0487 |
| `logistica+tam` | 0,8283 | 0,0245 | +0,0415 |
| `logistica` | 0,8278 | 0,0244 | +0,0411 |
| `boosting` | 0,8263 | 0,0177 | +0,0395 |
| `floresta` | 0,8256 | 0,0214 | +0,0388 |
| `boosting+tam` | 0,8247 | 0,0208 | +0,0379 |
| `floresta+tam` | 0,8236 | 0,0214 | +0,0368 |
| `regra_genero` (baseline) | 0,7868 | 0,0263 | 0,0000 |

Efeito dos grupos, dobra a dobra (25 dobras, mesmas dobras), contra o mesmo modelo sem grupos:

| variante | diferença média | dobras melhores / iguais / piores |
|---|---|---|
| `logistica+tam` | +0,0004 | 6 / 14 / 5 |
| `floresta+tam` | -0,0020 | 7 / 6 / 12 |
| `boosting+tam` | -0,0016 | 11 / 1 / 13 |
| `logistica+grupo` | +0,0077 | 18 / 2 / 5 |
| `floresta+grupo` | +0,0128 | 21 / 3 / 1 |
| `boosting+grupo` | +0,0094 | 16 / 1 / 8 |

- O **tamanho** do grupo sozinho não ajuda (de -0,2 a +0,04 ponto). A **sobrevivência** do grupo ajuda nos três modelos, de +0,8 a +1,3 ponto, e vence em mais dobras do que perde. É pouco perto do desvio entre dobras (2 pontos), mas consistente.
- Entre os três modelos com grupos a diferença é de 0,3 ponto: não dá para dizer qual é melhor. Escolher o maior entre vários é um pouco otimista.
- Cobertura: só 152 dos 418 passageiros do teste têm algum outro passageiro do mesmo bilhete no treino (177 pela família). Na validação cruzada cada linha vê só cerca de 80% dos rótulos do treino, então lá os grupos ajudam um pouco menos do que no teste real; mesmo assim, a maioria dos passageiros não tem grupo, então eu esperava um ganho pequeno no placar (o envio real ficou acima disso, veja abaixo).
- `titanic gerar` escolheu `floresta+grupo` (a maior média) e previu 143 sobreviventes entre os 418 do teste; 32 previsões diferem do envio da logística.
- Essas são notas de **validação**. O placar público do Kaggle usa só as 418 linhas do teste.
- **Placar público do envio da logística (04/11/2026): 0,76794**, ou seja, 321 acertos em 418. Ficou 6 pontos abaixo da validação cruzada. O erro padrão de uma acurácia medida em 418 linhas é de cerca de 2 pontos, e o teste não é igual ao treino; por isso esse placar **não separa** o modelo da regra de gênero, apesar dos 4 pontos de ganho na validação.
- **Placar público do envio de `floresta+grupo` (04/11/2026): 0,80143**, ou seja, 335 acertos em 418. Das 32 previsões que diferem do envio da logística, 23 passaram a acertar e 9 passaram a errar (+14). O ganho foi maior que o +1 ponto da validação cruzada; como o erro padrão de uma acurácia em 418 linhas é de cerca de 2 pontos, leia isso como um resultado favorável, não como uma medida precisa do efeito dos grupos.
- Medir só pelo placar público é pouco confiável: com 10 envios por dia dá para "ajustar" o modelo ao teste sem aprender nada. A validação cruzada continua sendo a régua para decidir o que mudar.

## Dados e regras

`data/` e `saidas/` ficam fora do git. Os arquivos da competição pertencem ao Kaggle e têm regras de uso: não os versione nem os redistribua. Os testes usam dados **sintéticos** gerados em `tests/conftest.py`.
