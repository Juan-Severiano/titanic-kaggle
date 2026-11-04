"""Dados SINTÉTICOS com as colunas do Kaggle. Não têm relação com os dados reais da competição."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def sintetico(n: int = 300, semente: int = 0, com_rotulo: bool = True, primeiro_id: int = 1):
    rng = np.random.default_rng(semente)
    sexo = rng.choice(["male", "female"], n, p=[0.64, 0.36])
    classe = rng.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])
    idade = rng.normal(30, 13, n).clip(1, 80).round(0)
    idade[rng.random(n) < 0.2] = np.nan
    titulo = np.where(
        sexo == "female",
        rng.choice(["Mrs", "Miss"], n),
        np.where(idade < 13, "Master", "Mr"),
    )
    tarifa = np.select([classe == 1, classe == 2], [80.0, 20.0], 10.0) * rng.lognormal(0, 0.4, n)
    tarifa[rng.random(n) < 0.02] = np.nan
    cabine = np.where((classe == 1) & (rng.random(n) < 0.7), "C85", None)
    embarque = rng.choice(["S", "C", "Q"], n, p=[0.72, 0.19, 0.09]).astype(object)
    embarque[rng.random(n) < 0.01] = None
    df = pd.DataFrame(
        {
            "PassengerId": np.arange(primeiro_id, primeiro_id + n),
            "Pclass": classe,
            "Name": [f"Sobrenome{i}, {t}. Nome{i}" for i, t in enumerate(titulo)],
            "Sex": sexo,
            "Age": idade,
            "SibSp": rng.poisson(0.5, n),
            "Parch": rng.poisson(0.4, n),
            "Ticket": [f"T{i}" for i in range(n)],
            "Fare": tarifa,
            "Cabin": cabine,
            "Embarked": embarque,
        }
    )
    if com_rotulo:
        prob = 0.15 + 0.55 * (sexo == "female") + 0.15 * (classe == 1) - 0.1 * (classe == 3)
        df.insert(1, "Survived", (rng.random(n) < prob.clip(0.02, 0.98)).astype(int))
    return df


@pytest.fixture
def treino():
    return sintetico(300, semente=1)


@pytest.fixture
def teste():
    return sintetico(120, semente=2, com_rotulo=False, primeiro_id=1000)


@pytest.fixture
def pasta_dados(tmp_path, treino, teste):
    treino.to_csv(tmp_path / "train.csv", index=False)
    teste.to_csv(tmp_path / "test.csv", index=False)
    return tmp_path
