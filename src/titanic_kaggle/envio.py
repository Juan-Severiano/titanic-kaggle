"""Geração e validação do CSV de envio (`PassengerId,Survived`)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .features import criar_features, preparar_treino
from .modelos import construir

COLUNAS_ENVIO = ["PassengerId", "Survived"]


def gerar_envio(nome: str, treino: pd.DataFrame, teste: pd.DataFrame) -> pd.DataFrame:
    """Treina `nome` em todo o treino e prevê o teste."""
    X, y = preparar_treino(treino, teste)
    modelo = construir(nome).fit(X, y)
    previsao = modelo.predict(criar_features(teste, treino))
    return pd.DataFrame(
        {"PassengerId": teste["PassengerId"].to_numpy(), "Survived": previsao.astype(int)}
    )


def validar_envio(envio: pd.DataFrame, teste: pd.DataFrame) -> None:
    """Levanta ValueError se o CSV não está no formato que o Kaggle aceita."""
    if list(envio.columns) != COLUNAS_ENVIO:
        raise ValueError(f"Colunas devem ser {COLUNAS_ENVIO}, vieram {list(envio.columns)}.")
    if len(envio) != len(teste):
        raise ValueError(f"Esperava {len(teste)} linhas, vieram {len(envio)}.")
    if envio["PassengerId"].duplicated().any():
        raise ValueError("PassengerId repetido.")
    if not envio["PassengerId"].reset_index(drop=True).equals(
        teste["PassengerId"].reset_index(drop=True)
    ):
        raise ValueError("Os PassengerId não batem com o test.csv (ou estão fora de ordem).")
    if envio["Survived"].isna().any() or not envio["Survived"].isin([0, 1]).all():
        raise ValueError("Survived deve ser 0 ou 1, sem vazios.")


def salvar_envio(envio: pd.DataFrame, caminho: Path) -> Path:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    envio.to_csv(caminho, index=False)
    return caminho
