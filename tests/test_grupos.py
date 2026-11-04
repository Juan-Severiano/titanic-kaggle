import numpy as np
import pandas as pd
import pytest

from titanic_kaggle.grupos import COLUNAS_SOBREVIVENCIA_GRUPO, SobrevivenciaDoGrupo


def _quadro(bilhetes, familias=None):
    familias = familias or [f"F{i}" for i in range(len(bilhetes))]
    return pd.DataFrame({"ChaveBilhete": bilhetes, "ChaveFamilia": familias})


def test_transform_usa_todos_os_rotulos_do_treino():
    X = _quadro(["A", "A", "A", "B"])
    y = np.array([1, 1, 0, 1])
    transformador = SobrevivenciaDoGrupo(suavizacao=1.0).fit(X, y)
    novo = _quadro(["A", "Z"])
    saida = transformador.transform(novo)

    prior = 0.75
    assert set(COLUNAS_SOBREVIVENCIA_GRUPO) <= set(saida.columns)
    assert saida["ConhecidosBilhete"].tolist() == [3, 0]
    assert saida["TaxaBilhete"].iloc[0] == pytest.approx((2 + prior) / (3 + 1))
    assert saida["TaxaBilhete"].iloc[1] == pytest.approx(prior)  # grupo nunca visto -> prior
    assert list(novo.columns) == ["ChaveBilhete", "ChaveFamilia"]  # a entrada não foi alterada


def test_fit_transform_nao_usa_o_proprio_rotulo():
    rng = np.random.default_rng(0)
    n = 200
    X = _quadro([f"T{i // 3}" for i in range(n)], [f"F{i // 2}" for i in range(n)])
    y = rng.integers(0, 2, n)
    base = SobrevivenciaDoGrupo().fit_transform(X, y)
    for i in (0, 7, 50, 199):
        invertido = y.copy()
        invertido[i] = 1 - invertido[i]
        outro = SobrevivenciaDoGrupo().fit_transform(X, invertido)
        for coluna in COLUNAS_SOBREVIVENCIA_GRUPO:
            assert outro[coluna].iloc[i] == pytest.approx(base[coluna].iloc[i])


def test_codificar_o_treino_com_transform_vazaria_o_rotulo():
    # Mostra por que existe `fit_transform`: `fit` + `transform` nas mesmas linhas depende do
    # próprio rótulo (o que treinaria o modelo para decorar a resposta).
    X = _quadro(["A", "A", "A", "A"])
    y = np.array([1, 0, 0, 0])
    com_rotulo = SobrevivenciaDoGrupo().fit(X, y).transform(X)["TaxaBilhete"].iloc[0]
    invertido = np.array([0, 0, 0, 0])
    sem_rotulo = SobrevivenciaDoGrupo().fit(X, invertido).transform(X)["TaxaBilhete"].iloc[0]
    assert com_rotulo != pytest.approx(sem_rotulo)


def test_fit_transform_e_deterministico_e_nao_deixa_nan():
    rng = np.random.default_rng(1)
    X = _quadro([f"T{i % 40}" for i in range(120)])
    y = rng.integers(0, 2, 120)
    a = SobrevivenciaDoGrupo().fit_transform(X, y)
    b = SobrevivenciaDoGrupo().fit_transform(X, y)
    pd.testing.assert_frame_equal(a, b)
    assert not a[COLUNAS_SOBREVIVENCIA_GRUPO].isna().any().any()
    assert a["TaxaBilhete"].between(0, 1).all()
