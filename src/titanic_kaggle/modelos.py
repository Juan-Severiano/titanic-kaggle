"""Modelos e avaliação por validação cruzada estratificada repetida."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .features import COLUNAS_TAMANHO_GRUPO, ImputaIdade, montar_colunas
from .grupos import COLUNAS_SOBREVIVENCIA_GRUPO, SobrevivenciaDoGrupo

SEMENTE = 42
BASELINE = "regra_genero"
FAMILIAS = ("logistica", "floresta", "boosting")
# "" = sem grupos; "+tam" = tamanho do bilhete/família (sem rótulo); "+grupo" = tamanho + o que
# aconteceu com os outros do grupo (codificado sem vazar o rótulo da própria linha).
VARIANTES = ("", "+tam", "+grupo")
NOMES = (BASELINE, *(f"{familia}{variante}" for familia in FAMILIAS for variante in VARIANTES))


class RegraGenero(ClassifierMixin, BaseEstimator):
    """Baseline: mulher sobrevive, homem não. Não aprende nada."""

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X) -> np.ndarray:
        return (X["Sex"] == "female").astype(int).to_numpy()


def _estimador(familia: str) -> BaseEstimator:
    if familia == "logistica":
        return LogisticRegression(max_iter=1000)
    if familia == "floresta":
        return RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, random_state=SEMENTE, n_jobs=-1
        )
    if familia == "boosting":
        return HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=200,
            l2_regularization=1.0,
            random_state=SEMENTE,
        )
    raise ValueError(f"Modelo desconhecido: {familia!r}. Use um de {', '.join(NOMES)}.")


def separar_nome(nome: str) -> tuple[str, str]:
    """'logistica+grupo' -> ('logistica', '+grupo')."""
    familia, mais, variante = nome.partition("+")
    return familia, mais + variante


def construir(nome: str) -> BaseEstimator:
    """Modelo pronto para receber a saída de `criar_features`."""
    if nome == BASELINE:
        return RegraGenero()
    familia, variante = separar_nome(nome)
    if variante not in VARIANTES:
        raise ValueError(f"Modelo desconhecido: {nome!r}. Use um de {', '.join(NOMES)}.")
    passos: list[tuple[str, BaseEstimator]] = [("idade", ImputaIdade())]
    extras: list[str] = []
    if variante in ("+tam", "+grupo"):
        extras += COLUNAS_TAMANHO_GRUPO
    if variante == "+grupo":
        passos.append(("grupo", SobrevivenciaDoGrupo()))
        extras += COLUNAS_SOBREVIVENCIA_GRUPO
    passos += [("colunas", montar_colunas(extras)), ("modelo", _estimador(familia))]
    return Pipeline(passos)


def notas_por_dobra(
    X: pd.DataFrame,
    y: pd.Series,
    nomes: tuple[str, ...] = NOMES,
    dobras: int = 5,
    repeticoes: int = 5,
) -> dict[str, np.ndarray]:
    """Acurácia de cada modelo em cada dobra, com as MESMAS dobras para todos (mesma semente).

    A regra de gênero entra sempre, para servir de referência.
    """
    cv = RepeatedStratifiedKFold(n_splits=dobras, n_repeats=repeticoes, random_state=SEMENTE)
    return {
        nome: cross_val_score(construir(nome), X, y, cv=cv, scoring="accuracy")
        for nome in dict.fromkeys((BASELINE, *nomes))
    }


def resumir(notas: dict[str, np.ndarray], nomes: tuple[str, ...] = NOMES) -> pd.DataFrame:
    """Média, desvio entre dobras e ganho pareado contra a regra de gênero."""
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


def efeito_dos_grupos(notas: dict[str, np.ndarray]) -> pd.DataFrame:
    """Cada variante com grupos contra o mesmo modelo sem grupos, dobra a dobra."""
    linhas = []
    for nome, valores in notas.items():
        familia, variante = separar_nome(nome)
        if variante == "" or familia not in notas:
            continue
        diferenca = valores - notas[familia]
        linhas.append(
            {
                "modelo": nome,
                "contra": familia,
                "diferenca": float(diferenca.mean()),
                "melhor": int((diferenca > 0).sum()),
                "igual": int((diferenca == 0).sum()),
                "pior": int((diferenca < 0).sum()),
            }
        )
    return pd.DataFrame(linhas, columns=["modelo", "contra", "diferenca", "melhor", "igual", "pior"])


def avaliar(
    X: pd.DataFrame,
    y: pd.Series,
    nomes: tuple[str, ...] = NOMES,
    dobras: int = 5,
    repeticoes: int = 5,
) -> pd.DataFrame:
    """Acurácia por validação cruzada, com as mesmas dobras para todos os modelos."""
    return resumir(notas_por_dobra(X, y, nomes, dobras, repeticoes), nomes)
