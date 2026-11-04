import numpy as np
import pandas as pd
import pytest

from titanic_kaggle.cli import main
from titanic_kaggle.envio import gerar_envio, validar_envio
from titanic_kaggle.features import criar_features, preparar_treino
from titanic_kaggle.modelos import BASELINE, NOMES, avaliar, construir


@pytest.mark.parametrize("nome", NOMES)
def test_modelo_treina_e_preve_01(nome, treino, teste):
    X, y = preparar_treino(treino)
    previsao = construir(nome).fit(X, y).predict(criar_features(teste))
    assert len(previsao) == len(teste)
    assert set(np.unique(previsao)) <= {0, 1}


def test_nome_desconhecido_da_erro_claro():
    with pytest.raises(ValueError, match="Modelo desconhecido"):
        construir("xgboost")


def test_baseline_e_a_regra_de_genero(treino):
    X, y = preparar_treino(treino)
    previsao = construir(BASELINE).fit(X, y).predict(X)
    assert previsao.tolist() == (treino["Sex"] == "female").astype(int).tolist()


def test_avaliar_tabela_e_ordem(treino):
    X, y = preparar_treino(treino)
    tabela = avaliar(X, y, ("logistica", BASELINE), repeticoes=2)
    assert list(tabela.columns) == ["modelo", "media", "desvio", "ganho"]
    assert tabela["media"].is_monotonic_decreasing
    base = tabela.set_index("modelo").loc[BASELINE]
    assert base["ganho"] == pytest.approx(0.0)
    assert 0.5 < base["media"] < 1.0


def test_avaliar_e_deterministico(treino):
    X, y = preparar_treino(treino)
    a = avaliar(X, y, ("logistica",), repeticoes=2)
    b = avaliar(X, y, ("logistica",), repeticoes=2)
    pd.testing.assert_frame_equal(a, b)


def test_modelo_aprende_o_sinal_dos_dados_sinteticos(treino):
    # Nos dados sintéticos mulher e 1ª classe sobrevivem mais; a logística tem que enxergar isso.
    X, y = preparar_treino(treino)
    tabela = avaliar(X, y, ("logistica",), repeticoes=2).set_index("modelo")
    assert tabela.loc["logistica", "media"] > 0.65


def test_envio_valido(treino, teste):
    envio = gerar_envio("logistica", treino, teste)
    validar_envio(envio, teste)
    assert list(envio.columns) == ["PassengerId", "Survived"]
    assert envio["PassengerId"].tolist() == teste["PassengerId"].tolist()


def test_validar_envio_recusa_formatos_errados(treino, teste):
    bom = gerar_envio("regra_genero", treino, teste)

    with pytest.raises(ValueError, match="Colunas"):
        validar_envio(bom.rename(columns={"Survived": "Sobreviveu"}), teste)
    with pytest.raises(ValueError, match="linhas"):
        validar_envio(bom.iloc[:-1], teste)
    with pytest.raises(ValueError, match="repetido"):
        validar_envio(bom.assign(PassengerId=bom["PassengerId"].iloc[0]), teste)
    with pytest.raises(ValueError, match="não batem"):
        validar_envio(bom.assign(PassengerId=bom["PassengerId"] + 1), teste)
    with pytest.raises(ValueError, match="0 ou 1"):
        validar_envio(bom.assign(Survived=2), teste)
    vazio = bom.astype({"Survived": float})
    vazio.loc[0, "Survived"] = np.nan
    with pytest.raises(ValueError, match="0 ou 1"):
        validar_envio(vazio, teste)


def test_cli_gerar_escreve_csv_sem_enviar(pasta_dados, tmp_path, capsys):
    saida = tmp_path / "saidas" / "submission.csv"
    codigo = main(
        ["gerar", "--dados", str(pasta_dados), "--modelo", "logistica", "--saida", str(saida)]
    )
    assert codigo == 0
    lido = pd.read_csv(saida)
    assert list(lido.columns) == ["PassengerId", "Survived"]
    assert len(lido) == 120
    assert "Nada foi enviado" in capsys.readouterr().out


def test_cli_avaliar_e_resumo_rodam(pasta_dados, capsys):
    assert main(["avaliar", "--dados", str(pasta_dados), "--repeticoes", "1",
                 "--modelos", "regra_genero", "logistica"]) == 0
    assert "logistica" in capsys.readouterr().out
    assert main(["resumo", "--dados", str(pasta_dados)]) == 0
    saida = capsys.readouterr().out
    assert "== sexo" in saida and "== faltantes" in saida


def test_cli_sem_dados_explica_como_baixar(tmp_path, capsys):
    assert main(["resumo", "--dados", str(tmp_path / "nao-existe")]) == 2
    assert "Join Competition" in capsys.readouterr().err
