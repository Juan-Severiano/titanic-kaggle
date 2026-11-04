import numpy as np
import pandas as pd
import pytest

from titanic_kaggle.features import (
    COLUNAS_CATEGORICAS,
    COLUNAS_NUMERICAS,
    ImputaIdade,
    criar_features,
    extrair_titulo,
    preparar_treino,
)


@pytest.mark.parametrize(
    ("nome", "esperado"),
    [
        ("Braund, Mr. Owen Harris", "Mr"),
        ("Cumings, Mrs. John Bradley (Florence Briggs Thayer)", "Mrs"),
        ("Heikkinen, Miss. Laina", "Miss"),
        ("Palsson, Master. Gosta Leonard", "Master"),
        ("Mlle, Mlle. Qualquer", "Miss"),
        ("Fulano, Ms. Alguma", "Miss"),
        ("Beltrano, Mme. Alguma", "Mrs"),
        ("Rothes, the Countess. of (Lucy Noel Martha)", "Raro"),
        ("Minahan, Dr. William Edward", "Raro"),
        ("sem formato", "Raro"),
    ],
)
def test_extrair_titulo(nome, esperado):
    assert extrair_titulo(nome) == esperado


def test_criar_features_colunas_e_valores():
    df = pd.DataFrame(
        {
            "Pclass": [3, 1],
            "Name": ["Braund, Mr. Owen", "Futrelle, Mrs. Jacques"],
            "Sex": ["male", "female"],
            "Age": [22.0, np.nan],
            "SibSp": [1, 0],
            "Parch": [0, 0],
            "Fare": [7.25, 0.0],
            "Cabin": [np.nan, "C85"],
            "Embarked": ["S", "C"],
        }
    )
    f = criar_features(df)
    assert set(COLUNAS_NUMERICAS + COLUNAS_CATEGORICAS) <= set(f.columns)
    assert f["TamanhoFamilia"].tolist() == [2, 1]
    assert f["Sozinho"].tolist() == [0, 1]
    assert f["IdadeFaltante"].tolist() == [0, 1]
    assert f["Deck"].tolist() == ["U", "C"]
    assert f["Classe"].tolist() == ["3", "1"]
    assert f["Titulo"].tolist() == ["Mr", "Mrs"]
    assert f["LogFare"].iloc[1] == 0.0


def test_criar_features_nao_muda_o_original(treino):
    antes = treino.copy()
    criar_features(treino)
    pd.testing.assert_frame_equal(treino, antes)


def test_preparar_treino_separa_o_rotulo(treino):
    X, y = preparar_treino(treino)
    assert "Survived" not in X.columns
    assert len(X) == len(y) == len(treino)
    assert set(y.unique()) <= {0, 1}


def test_imputa_idade_usa_so_a_mediana_do_treino():
    treino = pd.DataFrame(
        {
            "Titulo": ["Mr", "Mr", "Miss"],
            "Classe": ["3", "3", "1"],
            "Age": [20.0, 30.0, 40.0],
        }
    )
    outro = pd.DataFrame(
        {
            "Titulo": ["Mr", "Miss", "Raro", "Mr"],
            "Classe": ["3", "1", "2", "1"],
            "Age": [np.nan, np.nan, np.nan, 99.0],
        }
    )
    imp = ImputaIdade().fit(treino)
    saida = imp.transform(outro)
    # (Mr, 3) -> 25; (Miss, 1) -> 40; Raro nunca visto -> mediana global (30); idade presente não muda
    assert saida["Age"].tolist() == [25.0, 40.0, 30.0, 99.0]
    assert outro["Age"].isna().sum() == 3  # a entrada não foi alterada


def test_imputa_idade_grupo_novo_cai_para_o_titulo():
    treino = pd.DataFrame(
        {"Titulo": ["Mr", "Mr"], "Classe": ["3", "3"], "Age": [20.0, 40.0]}
    )
    novo = pd.DataFrame({"Titulo": ["Mr"], "Classe": ["1"], "Age": [np.nan]})
    assert ImputaIdade().fit(treino).transform(novo)["Age"].iloc[0] == 30.0
