"""Features do Titanic.

`criar_features` olha uma linha por vez, então não vaza informação entre treino e teste.
O que depende de estatística dos dados (mediana de idade, categoria mais frequente, escala)
fica dentro do Pipeline e é ajustado só com o treino de cada dobra.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TITULOS_COMUNS = ("Mr", "Mrs", "Miss", "Master")
TITULOS_EQUIVALENTES = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}

COLUNAS_NUMERICAS = [
    "Age",
    "IdadeFaltante",
    "LogFare",
    "SibSp",
    "Parch",
    "TamanhoFamilia",
    "Sozinho",
]
COLUNAS_CATEGORICAS = ["Sex", "Classe", "Embarked", "Titulo", "Deck"]
# Tamanho do grupo (quantos passageiros dividem o bilhete ou a família), contado em treino + teste.
COLUNAS_TAMANHO_GRUPO = ["TamBilhete", "TamFamilia"]


def extrair_titulo(nome: str) -> str:
    """'Braund, Mr. Owen Harris' -> 'Mr'. Títulos raros viram 'Raro'."""
    achado = re.search(r",\s*([^.]+)\.", str(nome))
    titulo = achado.group(1).strip() if achado else "Raro"
    titulo = TITULOS_EQUIVALENTES.get(titulo, titulo)
    return titulo if titulo in TITULOS_COMUNS else "Raro"


def _chaves(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Bilhete e família (sobrenome + classe). Só usa colunas de entrada, nunca o rótulo."""
    sobrenome = df["Name"].astype(str).str.split(",").str[0].str.strip()
    return df["Ticket"].astype(str), sobrenome + "|" + df["Pclass"].astype(str)


def criar_features(df: pd.DataFrame, referencia: pd.DataFrame | None = None) -> pd.DataFrame:
    """Colunas do Kaggle -> colunas do modelo.

    Tudo é por linha, exceto o tamanho do grupo (`TamBilhete`, `TamFamilia`), contado em `df` mais
    `referencia` (passe o outro arquivo: treino e teste contam o mesmo conjunto). Isso só usa colunas
    de entrada, nunca o rótulo, então não vaza a resposta.
    """
    saida = pd.DataFrame(index=df.index)
    saida["Sex"] = df["Sex"]
    saida["Classe"] = df["Pclass"].astype(str)
    saida["Embarked"] = df["Embarked"]
    saida["Age"] = df["Age"]
    saida["IdadeFaltante"] = df["Age"].isna().astype(int)
    saida["LogFare"] = np.log1p(df["Fare"])
    saida["SibSp"] = df["SibSp"]
    saida["Parch"] = df["Parch"]
    saida["TamanhoFamilia"] = df["SibSp"] + df["Parch"] + 1
    saida["Sozinho"] = (saida["TamanhoFamilia"] == 1).astype(int)
    saida["Titulo"] = df["Name"].map(extrair_titulo)
    # Deck = primeira letra da cabine; sem cabine vira "U" (desconhecido).
    saida["Deck"] = df["Cabin"].astype(object).fillna("U").astype(str).str[0]

    colunas = ["Name", "Ticket", "Pclass"]
    todos = df[colunas] if referencia is None else pd.concat([df[colunas], referencia[colunas]])
    bilhetes_todos, familias_todas = _chaves(todos)
    bilhete, familia = _chaves(df)
    saida["ChaveBilhete"] = bilhete
    saida["ChaveFamilia"] = familia
    saida["TamBilhete"] = bilhete.map(bilhetes_todos.value_counts()).astype(int)
    saida["TamFamilia"] = familia.map(familias_todas.value_counts()).astype(int)
    return saida


def preparar_treino(
    treino: pd.DataFrame, teste: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Features do treino (tamanho de grupo contado com o teste, se dado) e o rótulo."""
    return criar_features(treino, teste), treino["Survived"].astype(int)


class ImputaIdade(BaseEstimator, TransformerMixin):
    """Preenche `Age` com a mediana do grupo (Titulo, Classe), aprendida só em `fit`.

    Se o grupo não existe no treino, cai para a mediana do título e depois para a global.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.mediana_global_ = float(X["Age"].median())
        self.por_titulo_ = X.groupby("Titulo")["Age"].median().to_dict()
        self.por_grupo_ = X.groupby(["Titulo", "Classe"])["Age"].median().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        grupo = pd.Series(
            [self.por_grupo_.get(chave) for chave in zip(X["Titulo"], X["Classe"])],
            index=X.index,
            dtype=float,
        )
        titulo = X["Titulo"].map(self.por_titulo_).astype(float)
        X["Age"] = X["Age"].fillna(grupo).fillna(titulo).fillna(self.mediana_global_)
        return X


def montar_colunas(extras: tuple[str, ...] | list[str] = ()) -> ColumnTransformer:
    """`extras`: colunas numéricas a mais (tamanho de grupo, sobrevivência do grupo)."""
    numericas = Pipeline(
        [("imputar", SimpleImputer(strategy="median")), ("escala", StandardScaler())]
    )
    categoricas = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numericas, [*COLUNAS_NUMERICAS, *extras]),
            ("cat", categoricas, COLUNAS_CATEGORICAS),
        ]
    )
