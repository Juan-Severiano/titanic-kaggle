"""Linha de comando: `titanic resumo | avaliar | gerar`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .dados import DADOS, SAIDAS, DadosAusentes, carregar
from .envio import gerar_envio, salvar_envio, validar_envio
from .explorar import resumo
from .features import preparar_treino
from .modelos import NOMES, avaliar


def _formatar(tabela) -> str:
    return tabela.to_string(float_format=lambda v: f"{v:.4f}")


def _resumo(args: argparse.Namespace) -> int:
    treino, teste = carregar(args.dados)
    print(f"treino: {len(treino)} linhas | teste: {len(teste)} linhas\n")
    for titulo, tabela in resumo(treino, teste).items():
        print(f"== {titulo}")
        print(_formatar(tabela), end="\n\n")
    return 0


def _avaliar(args: argparse.Namespace) -> int:
    treino, _ = carregar(args.dados)
    X, y = preparar_treino(treino)
    tabela = avaliar(X, y, tuple(args.modelos), repeticoes=args.repeticoes)
    print(f"Acurácia em validação cruzada (5 dobras x {args.repeticoes} repetições):\n")
    print(_formatar(tabela.set_index("modelo")))
    print(
        "\n'desvio' é entre dobras; 'ganho' é a diferença para a regra de gênero. "
        "O placar do Kaggle usa só o teste e costuma ficar abaixo da validação."
    )
    return 0


def _gerar(args: argparse.Namespace) -> int:
    treino, teste = carregar(args.dados)
    nome = args.modelo
    if nome == "auto":
        X, y = preparar_treino(treino)
        nome = str(avaliar(X, y, repeticoes=args.repeticoes).iloc[0]["modelo"])
        print(f"Modelo escolhido pela validação cruzada: {nome}")
    envio = gerar_envio(nome, treino, teste)
    validar_envio(envio, teste)
    caminho = salvar_envio(envio, args.saida)
    print(
        f"Gerei {caminho} ({len(envio)} linhas, {int(envio['Survived'].sum())} previstos "
        "como sobreviventes). Nada foi enviado ao Kaggle."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="titanic", description=__doc__)
    sub = parser.add_subparsers(dest="comando", required=True)

    def comum(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dados", type=Path, default=DADOS, help="pasta com train.csv e test.csv")

    p = sub.add_parser("resumo", help="faltantes e taxa de sobrevivência por grupo")
    comum(p)
    p.set_defaults(func=_resumo)

    p = sub.add_parser("avaliar", help="acurácia por validação cruzada")
    comum(p)
    p.add_argument("--modelos", nargs="+", choices=NOMES, default=list(NOMES))
    p.add_argument("--repeticoes", type=int, default=5)
    p.set_defaults(func=_avaliar)

    p = sub.add_parser("gerar", help="gera o CSV de envio (não envia ao Kaggle)")
    comum(p)
    p.add_argument("--modelo", choices=(*NOMES, "auto"), default="auto")
    p.add_argument("--repeticoes", type=int, default=5)
    p.add_argument("--saida", type=Path, default=SAIDAS / "submission.csv")
    p.set_defaults(func=_gerar)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DadosAusentes as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
