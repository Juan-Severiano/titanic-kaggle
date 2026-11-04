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


def extrair_titulo(nome: str) -> str:
    """'Braund, Mr. Owen Harris' -> 'Mr'. Títulos raros viram 'Raro'."""
    achado = re.search(r",\s*([^.]+)\.", str(nome))
    titulo = achado.group(1).strip() if achado else "Raro"
    titulo = TITULOS_EQUIVALENTES.get(titulo, titulo)
    return titulo if titulo in TITULOS_COMUNS else "Raro"


def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Colunas do Kaggle -> colunas do modelo. Sem olhar outras linhas."""
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
    saida["Deck"] = df["Cabin"].fillna("U").astype(str).str[0]
    return saida


def preparar_treino(treino: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return criar_features(treino), treino["Survived"].astype(int)


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


def montar_colunas() -> ColumnTransformer:
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
            ("num", numericas, COLUNAS_NUMERICAS),
            ("cat", categoricas, COLUNAS_CATEGORICAS),
        ]
    )
