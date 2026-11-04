"""Primeira olhada nos dados: faltantes e taxa de sobrevivência por grupo."""

from __future__ import annotations

import pandas as pd

from .features import criar_features


def _taxa(tabela: pd.DataFrame, coluna: str) -> pd.DataFrame:
    grupos = tabela.groupby(coluna)["Survived"].agg(passageiros="size", sobreviveram="mean")
    return grupos.sort_values("sobreviveram", ascending=False)


def resumo(treino: pd.DataFrame, teste: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Tabelas prontas para imprimir: faltantes e taxa de sobrevivência por grupo."""
    features = criar_features(treino)
    tabela = features.assign(Survived=treino["Survived"].astype(int))
    tabela["FaixaEtaria"] = pd.cut(
        treino["Age"], [0, 12, 18, 35, 60, 120], labels=["0-12", "13-18", "19-35", "36-60", "61+"]
    ).astype(object).fillna("sem idade")
    tabela["Familia"] = pd.cut(
        features["TamanhoFamilia"], [0, 1, 4, 100], labels=["sozinho", "2-4", "5+"]
    ).astype(str)
    faltantes = pd.DataFrame(
        {
            "treino": treino.isna().sum(),
            # `Survived` não existe no teste: reindex dá NaN e vira 0 (não é faltante de verdade).
            "teste": teste.isna().sum().reindex(treino.columns).fillna(0).astype(int),
        }
    )
    faltantes = faltantes[(faltantes["treino"] > 0) | (faltantes["teste"] > 0)]
    return {
        "faltantes": faltantes,
        "sexo": _taxa(tabela, "Sex"),
        "classe": _taxa(tabela, "Classe"),
        "titulo": _taxa(tabela, "Titulo"),
        "faixa etária": _taxa(tabela, "FaixaEtaria"),
        "família": _taxa(tabela, "Familia"),
    }
