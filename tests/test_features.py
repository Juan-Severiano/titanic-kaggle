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
            "Ticket": ["A/5 21171", "PC 17599"],
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


def _quadro(tickets, nomes, classes):
    n = len(tickets)
    return pd.DataFrame(
        {
            "Pclass": classes,
            "Name": nomes,
            "Sex": ["male"] * n,
            "Age": [30.0] * n,
            "SibSp": [0] * n,
            "Parch": [0] * n,
            "Ticket": tickets,
            "Fare": [10.0] * n,
            "Cabin": [np.nan] * n,
            "Embarked": ["S"] * n,
        }
    )


def test_tamanho_do_grupo_conta_treino_e_teste_juntos():
    treino = _quadro(["A", "B"], ["Silva, Mr. Ana", "Souza, Mr. Bia"], [3, 3])
    teste = _quadro(["A", "C"], ["Silva, Mrs. Cleo", "Lima, Mr. Dan"], [3, 1])
    sozinho = criar_features(treino)
    junto = criar_features(treino, teste)
    assert sozinho["TamBilhete"].tolist() == [1, 1]
    assert junto["TamBilhete"].tolist() == [2, 1]  # o bilhete A também aparece no teste
    assert junto["TamFamilia"].tolist() == [2, 1]  # Silva|3 aparece nos dois
    # visto do outro lado, a contagem é a mesma
    assert criar_features(teste, treino)["TamBilhete"].tolist() == [2, 1]
    assert junto["ChaveFamilia"].tolist() == ["Silva|3", "Souza|3"]


def test_classe_diferente_separa_familias_com_o_mesmo_sobrenome():
    df = _quadro(["A", "B"], ["Silva, Mr. Ana", "Silva, Mr. Bia"], [1, 3])
    assert criar_features(df)["TamFamilia"].tolist() == [1, 1]


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
