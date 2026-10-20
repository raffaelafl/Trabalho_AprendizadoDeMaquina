
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURAÇÕES (altere aqui, sem mexer nas funções)
# =============================================================================

ARQUIVO_ENTRADA = "ProgramaRodofaunaAtropelada.xlsx"
ARQUIVO_SAIDA = "dados_engenharia.xlsx"
ABA_ENTRADA = 0  # índice ou nome da aba (0 = primeira aba)

# Filtro de registros
COLUNA_FILTRO = "silvestre_"
VALOR_FILTRO = "silvestre"

# Colunas removidas por não serem atributos preditores (requisito do projeto)
COLUNAS_REMOVER = [
    "id", "silvestre_", "campanha", "periodo", "hora",
    "sexo", "posicao", "ordem", "familia", "nome_popul",
]

# Colunas puramente técnicas (índice do ArcGIS/shapefile). Não descrevem o
# evento e, por serem únicas por linha, impedem a detecção de duplicatas.
# Elas são ignoradas na busca de duplicatas e removidas junto com as demais.
# Para MANTER essa coluna na saída, esvazie a lista: COLUNAS_TECNICAS = []
COLUNAS_TECNICAS = ["OBJECTID"]

# Coluna de data e seu formato no arquivo (ex.: 05/04/10 -> "%d/%m/%y")
COLUNA_DATA = "data_dt"
FORMATO_DATA = "%d/%m/%y"

# Coordenadas UTM (metros)
COLUNA_X = "x"
COLUNA_Y = "y"

# Tamanhos de célula do grid (metros)
TAMANHO_GRID_FINO_M = 500
TAMANHO_GRID_GROSSO_M = 1000

# Resolução usada para gerar o "id_local" (metros). Por padrão, igual ao
# grid fino: cada célula de 500 m recebe um código inteiro único.
TAMANHO_ID_LOCAL_M = 500

# Meses do período chuvoso no Distrito Federal (outubro a abril)
MESES_CHUVOSOS = {10, 11, 12, 1, 2, 3, 4}

# Colunas mínimas exigidas na entrada
COLUNAS_OBRIGATORIAS = [COLUNA_FILTRO, COLUNA_DATA, COLUNA_X, COLUNA_Y]


# =============================================================================
# 1. LEITURA
# =============================================================================

def carregar_dados(caminho, aba=ABA_ENTRADA):
    """Lê a planilha Excel e devolve um DataFrame.

    Trata os erros mais comuns: arquivo inexistente, extensão errada,
    arquivo corrompido/aberto por outro programa, aba inexistente,
    planilha vazia e colunas obrigatórias ausentes.
    """
    caminho = Path(caminho)

    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {caminho}")
    if caminho.suffix.lower() not in (".xlsx", ".xlsm"):
        raise ValueError(f"Formato não suportado ({caminho.suffix}). Use .xlsx.")

    try:
        df = pd.read_excel(caminho, sheet_name=aba, engine="openpyxl")
    except PermissionError as erro:
        raise PermissionError(
            f"Sem permissão para ler '{caminho}'. Feche o arquivo no Excel."
        ) from erro
    except ValueError as erro:  # aba inexistente
        raise ValueError(f"Não foi possível ler a aba '{aba}': {erro}") from erro
    except Exception as erro:  # arquivo corrompido, zip inválido etc.
        raise RuntimeError(f"Falha ao ler '{caminho}': {erro}") from erro

    if df.empty:
        raise ValueError("A planilha de entrada está vazia.")

    faltantes = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltantes:
        raise KeyError(f"Colunas obrigatórias ausentes na planilha: {faltantes}")

    # Remove espaços acidentais nos nomes das colunas
    df.columns = [str(c).strip() for c in df.columns]
    return df


# =============================================================================
# 2. FILTRO
# =============================================================================

def filtrar_silvestres(df):
    """Mantém apenas os registros em que COLUNA_FILTRO == VALOR_FILTRO.

    A comparação ignora espaços e diferenças de maiúsculas/minúsculas.
    """
    valores = df[COLUNA_FILTRO].astype(str).str.strip().str.lower()
    filtrado = df.loc[valores == VALOR_FILTRO.lower()].copy()

    if filtrado.empty:
        raise ValueError(
            f"Nenhum registro com {COLUNA_FILTRO} == '{VALOR_FILTRO}'."
        )
    return filtrado.reset_index(drop=True)


# =============================================================================
# 3. DUPLICATAS
# =============================================================================

def remover_duplicatas(df):
    """Remove linhas duplicadas e devolve (df_sem_duplicatas, qtd_removida).

    As colunas técnicas (ex.: OBJECTID) são ignoradas na comparação, pois
    são únicas por linha e mascarariam as duplicatas reais.
    Mantém a primeira ocorrência de cada registro.
    """
    colunas_comparacao = [c for c in df.columns if c not in COLUNAS_TECNICAS]
    antes = len(df)
    df = df.drop_duplicates(subset=colunas_comparacao, keep="first")
    return df.reset_index(drop=True), antes - len(df)


# =============================================================================
# 4. REMOÇÃO DE COLUNAS
# =============================================================================

def remover_colunas(df):
    """Remove colunas não preditoras e devolve (df, lista_removidas).

    Colunas inexistentes são ignoradas (sem erro), o que torna o código
    tolerante a pequenas variações da planilha.
    """
    alvo = COLUNAS_REMOVER + COLUNAS_TECNICAS
    removidas = [c for c in alvo if c in df.columns]
    return df.drop(columns=removidas), removidas


# =============================================================================
# 5. ATRIBUTOS TEMPORAIS
# =============================================================================

def _estacao_hemisferio_sul(datas):
    """Classifica cada data na estação do ano (hemisfério sul).

    Datas aproximadas (início fixo, desconsiderando variação de 1 dia):
        Verão     : 21/dez a 20/mar
        Outono    : 21/mar a 20/jun
        Inverno   : 21/jun a 22/set
        Primavera : 23/set a 20/dez
    """
    mmdd = datas.dt.month * 100 + datas.dt.day  # ex.: 21 de dezembro -> 1221
    condicoes = [
        (mmdd >= 1221) | (mmdd <= 320),
        (mmdd >= 321) & (mmdd <= 620),
        (mmdd >= 621) & (mmdd <= 922),
    ]
    return np.select(condicoes, ["Verão", "Outono", "Inverno"], default="Primavera")


def criar_atributos_temporais(df):
    """Converte COLUNA_DATA para datetime e cria os atributos temporais.

    Atenção: se já existir uma coluna "mes" (ex.: nome do mês em texto),
    ela é SUBSTITUÍDA pelo mês numérico (1 a 12).
    Linhas com data inválida geram NaT/NaN e aparecem no relatório de nulos.
    """
    df = df.copy()
    df[COLUNA_DATA] = pd.to_datetime(
        df[COLUNA_DATA], format=FORMATO_DATA, errors="coerce"
    )

    invalidas = int(df[COLUNA_DATA].isna().sum())
    if invalidas:
        print(f"[AVISO] {invalidas} data(s) inválida(s) em '{COLUNA_DATA}' "
              f"(formato esperado: {FORMATO_DATA}).")

    data = df[COLUNA_DATA]
    df["mes"] = data.dt.month
    df["trimestre"] = data.dt.quarter
    df["semestre"] = np.where(data.dt.month <= 6, 1, 2)
    df["dia_do_mes"] = data.dt.day
    df["dia_do_ano"] = data.dt.dayofyear
    df["semana_do_ano"] = data.dt.isocalendar().week.astype("Int64")
    df["fim_de_semana"] = (data.dt.dayofweek >= 5).astype(int)  # sáb=5, dom=6
    df["estacao_do_ano"] = _estacao_hemisferio_sul(data)
    df["periodo_chuvoso"] = data.dt.month.isin(MESES_CHUVOSOS).astype(int)
    df["ano_bissexto"] = data.dt.is_leap_year.astype(int)

    # Linhas sem data: os atributos derivados não podem ter valor
    sem_data = data.isna()
    if sem_data.any():
        df.loc[sem_data, ["fim_de_semana", "periodo_chuvoso", "ano_bissexto",
                          "estacao_do_ano", "semestre"]] = np.nan
    return df


# =============================================================================
# 6. ATRIBUTOS ESPACIAIS
# =============================================================================

def _rotulo_grid(x, y, tamanho):
    """Devolve o rótulo 'X_Y' do canto inferior-esquerdo da célula do grid.

    Ex.: x=219467, y=8273815, tamanho=500 -> '219000_8273500'.
    """
    gx = (np.floor(x / tamanho) * tamanho).astype("int64")
    gy = (np.floor(y / tamanho) * tamanho).astype("int64")
    return gx.astype(str) + "_" + gy.astype(str)


def criar_atributos_espaciais(df):
    """Cria grid_500m, grid_1000m, ocorrencias_grid, densidade_grid e id_local.

    - grid_*       : célula quadrada que contém o ponto (x, y).
    - ocorrencias  : nº de registros na célula de 500 m (inclui o próprio).
    - densidade    : ocorrências por km² (área da célula = lado² em km²).
    - id_local     : código inteiro único por célula (ver TAMANHO_ID_LOCAL_M).

    Deve ser chamada DEPOIS de remover duplicatas, para que a contagem
    não seja inflada por registros repetidos.
    """
    df = df.copy()
    x = pd.to_numeric(df[COLUNA_X], errors="coerce")
    y = pd.to_numeric(df[COLUNA_Y], errors="coerce")

    nome_fino = f"grid_{TAMANHO_GRID_FINO_M}m"
    nome_grosso = f"grid_{TAMANHO_GRID_GROSSO_M}m"

    df[nome_fino] = _rotulo_grid(x, y, TAMANHO_GRID_FINO_M)
    df[nome_grosso] = _rotulo_grid(x, y, TAMANHO_GRID_GROSSO_M)

    # Coordenadas ausentes não formam grid válido
    sem_coord = x.isna() | y.isna()
    df.loc[sem_coord, [nome_fino, nome_grosso]] = np.nan

    # Quantidade de registros por célula fina (NaN é ignorado pelo groupby)
    df["ocorrencias_grid"] = df.groupby(nome_fino)[nome_fino].transform("size")

    # Densidade = ocorrências / área da célula em km²
    area_km2 = (TAMANHO_GRID_FINO_M / 1000.0) ** 2
    df["densidade_grid"] = (df["ocorrencias_grid"] / area_km2).round(4)

    # id_local: código inteiro sequencial para cada região única
    rotulo_local = _rotulo_grid(x, y, TAMANHO_ID_LOCAL_M)
    rotulo_local[sem_coord] = np.nan
    df["id_local"] = pd.factorize(rotulo_local)[0] + 1  # 1..N
    df.loc[sem_coord, "id_local"] = np.nan
    df["id_local"] = df["id_local"].astype("Int64")
    return df


# =============================================================================
# 7. RELATÓRIO
# =============================================================================

def gerar_relatorio(df_final, estatisticas):
    """Imprime o relatório final do processamento.

    `estatisticas` é um dicionário preenchido por main() com:
        n_inicial, n_filtrado, n_duplicatas, colunas_originais,
        colunas_removidas.
    """
    criadas = [c for c in df_final.columns if c not in estatisticas["colunas_originais"]]
    # "mes" já existia como texto e foi recriado como número: informar
    recriadas = [c for c in df_final.columns
                 if c in estatisticas["colunas_originais"] and c == "mes"]

    linha = "=" * 64
    print(f"\n{linha}\n RELATÓRIO DE ENGENHARIA DE ATRIBUTOS - RODOFAUNA\n{linha}")
    print(f" Registros iniciais              : {estatisticas['n_inicial']}")
    print(f" Registros após filtro           : {estatisticas['n_filtrado']}")
    print(f" Duplicatas removidas            : {estatisticas['n_duplicatas']}")
    print(f" Registros na base final         : {len(df_final)}")
    print(f" Colunas na base final           : {df_final.shape[1]}")

    print("\n Valores nulos por coluna:")
    nulos = df_final.isna().sum()
    for coluna, qtd in nulos.items():
        print(f"   {coluna:<20} {int(qtd)}")

    print(f"\n Colunas criadas ({len(criadas)}):")
    print("   " + ", ".join(criadas))
    if recriadas:
        print(f"\n Colunas recriadas (já existiam, valor substituído): "
              f"{', '.join(recriadas)}")
    print(f"\n Colunas removidas ({len(estatisticas['colunas_removidas'])}):")
    print("   " + ", ".join(estatisticas["colunas_removidas"]))
    print(linha)


# =============================================================================
# 8. SAÍDA
# =============================================================================

def salvar_dados(df, caminho):
    """Salva o DataFrame em .xlsx (cabeçalho em negrito e colunas ajustadas)."""
    caminho = Path(caminho)
    try:
        with pd.ExcelWriter(caminho, engine="openpyxl") as escritor:
            df.to_excel(escritor, index=False, sheet_name="dados_engenharia")
            planilha = escritor.sheets["dados_engenharia"]

            from openpyxl.styles import Font
            planilha.freeze_panes = "A2"
            for celula in planilha[1]:
                celula.font = Font(bold=True)
            # Largura de cada coluna conforme o maior texto (amostra de 500 linhas)
            for i, coluna in enumerate(df.columns, start=1):
                maior = max(len(str(coluna)),
                            df[coluna].astype(str).head(500).str.len().max())
                letra = planilha.cell(row=1, column=i).column_letter
                planilha.column_dimensions[letra].width = min(maior + 2, 30)

                # Colunas de data: formato dd/mm/aaaa e largura suficiente
                # (evita a exibição "########" no Excel)
                if pd.api.types.is_datetime64_any_dtype(df[coluna]):
                    for linha in range(2, len(df) + 2):
                        planilha.cell(row=linha, column=i).number_format = "DD/MM/YYYY"
                    planilha.column_dimensions[letra].width = 14
    except PermissionError as erro:
        raise PermissionError(
            f"Não foi possível gravar '{caminho}'. Feche-o se estiver aberto."
        ) from erro
    print(f"\n Arquivo salvo em: {caminho.resolve()}")


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

def main():
    """Executa o pipeline completo de engenharia de atributos."""
    parser = argparse.ArgumentParser(description="Engenharia de atributos RodoFauna")
    parser.add_argument("-i", "--entrada", default=ARQUIVO_ENTRADA,
                        help="planilha .xlsx de entrada")
    parser.add_argument("-o", "--saida", default=ARQUIVO_SAIDA,
                        help="planilha .xlsx de saída")
    args = parser.parse_args()

    try:
        df = carregar_dados(args.entrada)
        colunas_originais = list(df.columns)
        n_inicial = len(df)

        df = filtrar_silvestres(df)
        n_filtrado = len(df)

        df, n_duplicatas = remover_duplicatas(df)
        df, colunas_removidas = remover_colunas(df)
        df = criar_atributos_temporais(df)
        df = criar_atributos_espaciais(df)

        estatisticas = {
            "n_inicial": n_inicial,
            "n_filtrado": n_filtrado,
            "n_duplicatas": n_duplicatas,
            "colunas_originais": colunas_originais,
            "colunas_removidas": colunas_removidas,
        }
        gerar_relatorio(df, estatisticas)
        salvar_dados(df, args.saida)

    except (FileNotFoundError, KeyError, ValueError, PermissionError, RuntimeError) as erro:
        print(f"\n[ERRO] {erro}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()