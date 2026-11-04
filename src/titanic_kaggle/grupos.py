"""Sobrevivência do grupo (bilhete e família), calculada sem vazar o rótulo da própria linha.

Quem viaja junto tende a ter o mesmo destino. Para cada passageiro olhamos o que aconteceu com os
OUTROS passageiros do mesmo bilhete e da mesma família que têm rótulo conhecido.

Cuidado central: no treino, a codificação de cada linha usa só os rótulos de OUTRAS dobras internas
(`fit_transform`), exatamente como o teste do Kaggle vê o treino inteiro. Codificar o treino com o
próprio rótulo, ou por "leave-one-out", faria o modelo decorar a resposta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

CHAVES = {"Bilhete": "ChaveBilhete", "Familia": "ChaveFamilia"}
COLUNAS_SOBREVIVENCIA_GRUPO = [
    "TaxaBilhete",
    "ConhecidosBilhete",
    "TaxaFamilia",
    "ConhecidosFamilia",
]


class SobrevivenciaDoGrupo(BaseEstimator, TransformerMixin):
    """Adiciona, por grupo, a taxa de sobrevivência suavizada e quantos rótulos entraram.

    `Taxa` = (sobreviventes + suavizacao * prior) / (conhecidos + suavizacao); grupo sem rótulo
    conhecido recebe o prior (taxa geral das linhas usadas).
    """

    def __init__(self, dobras: int = 5, suavizacao: float = 1.0, semente: int = 42):
        self.dobras = dobras
        self.suavizacao = suavizacao
        self.semente = semente

    @staticmethod
    def _estatisticas(X: pd.DataFrame, y: np.ndarray) -> dict[str, pd.DataFrame]:
        return {
            nome: pd.DataFrame({"k": X[coluna].to_numpy(), "y": y})
            .groupby("k")["y"]
            .agg(["sum", "count"])
            for nome, coluna in CHAVES.items()
        }

    def _codificar(
        self, X: pd.DataFrame, estatisticas: dict[str, pd.DataFrame], prior: float
    ) -> dict[str, np.ndarray]:
        saida = {}
        for nome, coluna in CHAVES.items():
            tabela = estatisticas[nome]
            soma = X[coluna].map(tabela["sum"]).fillna(0.0).to_numpy(dtype=float)
            conhecidos = X[coluna].map(tabela["count"]).fillna(0.0).to_numpy(dtype=float)
            saida[f"Taxa{nome}"] = (soma + self.suavizacao * prior) / (conhecidos + self.suavizacao)
            saida[f"Conhecidos{nome}"] = conhecidos
        return saida

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, dtype=float)
        self.prior_ = float(y.mean())
        self.estatisticas_ = self._estatisticas(X, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y=None, **parametros) -> pd.DataFrame:
        """Codificação do TREINO: cada linha só enxerga rótulos de outras dobras internas."""
        y = np.asarray(y, dtype=float)
        self.fit(X, y)
        novas = {coluna: np.full(len(X), np.nan) for coluna in COLUNAS_SOBREVIVENCIA_GRUPO}
        divisor = KFold(self.dobras, shuffle=True, random_state=self.semente)  # não olha para y
        for dentro, fora in divisor.split(X):
            estatisticas = self._estatisticas(X.iloc[dentro], y[dentro])
            parte = self._codificar(X.iloc[fora], estatisticas, float(y[dentro].mean()))
            for coluna in novas:
                novas[coluna][fora] = parte[coluna]
        saida = X.copy()
        for coluna, valores in novas.items():
            saida[coluna] = valores
        return saida

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Codificação de linhas NOVAS (validação, teste): usa todos os rótulos do treino."""
        saida = X.copy()
        for coluna, valores in self._codificar(X, self.estatisticas_, self.prior_).items():
            saida[coluna] = valores
        return saida
