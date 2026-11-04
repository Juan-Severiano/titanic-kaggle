"""Modelos e avaliação por validação cruzada estratificada repetida."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .features import ImputaIdade, montar_colunas

SEMENTE = 42
BASELINE = "regra_genero"
NOMES = (BASELINE, "logistica", "floresta", "boosting")


class RegraGenero(ClassifierMixin, BaseEstimator):
    """Baseline: mulher sobrevive, homem não. Não aprende nada."""

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X) -> np.ndarray:
        return (X["Sex"] == "female").astype(int).to_numpy()


def _estimador(nome: str) -> BaseEstimator:
    if nome == "logistica":
        return LogisticRegression(max_iter=1000)
    if nome == "floresta":
        return RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, random_state=SEMENTE, n_jobs=-1
        )
    if nome == "boosting":
        return HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=200,
            l2_regularization=1.0,
            random_state=SEMENTE,
        )
    raise ValueError(f"Modelo desconhecido: {nome!r}. Use um de {', '.join(NOMES)}.")


def construir(nome: str) -> BaseEstimator:
    """Modelo pronto para receber a saída de `criar_features`."""
    if nome == BASELINE:
        return RegraGenero()
    return Pipeline(
        [
            ("idade", ImputaIdade()),
            ("colunas", montar_colunas()),
            ("modelo", _estimador(nome)),
        ]
    )


def avaliar(
    X: pd.DataFrame,
    y: pd.Series,
    nomes: tuple[str, ...] = NOMES,
    dobras: int = 5,
    repeticoes: int = 5,
) -> pd.DataFrame:
    """Acurácia por validação cruzada, com as mesmas dobras para todos os modelos.

    `ganho` é a média da diferença pareada contra a regra de gênero, dobra a dobra.
    """
    cv = RepeatedStratifiedKFold(n_splits=dobras, n_repeats=repeticoes, random_state=SEMENTE)
    notas = {
        nome: cross_val_score(construir(nome), X, y, cv=cv, scoring="accuracy")
        for nome in dict.fromkeys((BASELINE, *nomes))
    }
    linhas = [
        {
            "modelo": nome,
            "media": float(valores.mean()),
            "desvio": float(valores.std(ddof=1)),
            "ganho": float((valores - notas[BASELINE]).mean()),
        }
        for nome, valores in notas.items()
        if nome in nomes
    ]
    return pd.DataFrame(linhas).sort_values("media", ascending=False, ignore_index=True)
