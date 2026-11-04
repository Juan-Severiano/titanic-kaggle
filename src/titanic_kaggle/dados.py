"""Leitura dos arquivos da competição (`train.csv`, `test.csv`)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
DADOS = RAIZ / "data"
SAIDAS = RAIZ / "saidas"

COMO_BAIXAR = (
    "Entre na competição no site (botão Join Competition, aceitando as regras) e rode:\n"
    "  uvx --with six kaggle competitions download -c titanic -p data\n"
    "  unzip -o data/titanic.zip -d data"
)


class DadosAusentes(FileNotFoundError):
    """Os CSVs da competição não estão na pasta de dados."""


def carregar(pasta: Path = DADOS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devolve (treino, teste) como saem do Kaggle, sem nenhuma alteração."""
    treino, teste = pasta / "train.csv", pasta / "test.csv"
    faltando = [p.name for p in (treino, teste) if not p.is_file()]
    if faltando:
        raise DadosAusentes(f"Faltam {', '.join(faltando)} em {pasta}.\n{COMO_BAIXAR}")
    return pd.read_csv(treino), pd.read_csv(teste)
