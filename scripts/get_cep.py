from pathlib import Path
import pandas as pd
from scripts.labels import *

# set up paths
ROOT_DIR = Path.cwd()
DATA_DIR = ROOT_DIR / "data"

cep_raw_df = pd.read_csv(DATA_DIR / "cleaned" / "cep_no_na.csv")[[
    'encuesta_a', # año de encuesta
    'encuesta_m', # mes de encuesta
    'encuesta', # numero de encuesta
    'iden_pol_1', # identidad politica
    'region_2', # region (62 a 81)
    'region_3', # region (82 a 89)
    'sexo', # sexp
    'pobreza_17_a',
    'pobreza_17_b',
    'gse', # nivel socioeconomico
    'esc_nivel_1_b', # escolaridad (58 a 84))
    'esc_nivel_1_c', # escolaridad (85 a 89)
    'edad', # edad
    'zona_u_r', # zona rural/urbana
    'constitucion_1', # constitucion aprueba/rechaza
    ]]

# def clean_row(row):
#     return row.replace({r'^\s*<NA>\s*$': pd.NA, r'^\s*$': pd.NA}, regex=True).dropna()


def process_column(df: pd.DataFrame,
                 col: str,
                 labels: dict = None,
                 rename: str = None):
    
    df[col] = pd.to_numeric(cep_raw_df[col], errors='coerce').astype('Int64').astype(str)
    
    if labels:
        df[col] = df[col].replace(labels)

    if rename:
        df = df.rename(columns={col: rename})
    
    return df

def simplify_columns(
        df: pd.DataFrame,
        col1: str,
        col2: str,
        new_col: str
        ):
    
    a = df[col1].replace({r"^\s*<NA>\s*$": pd.NA}, regex=True).astype("string")
    b = df[col2].replace({r"^\s*<NA>\s*$": pd.NA}, regex=True).astype("string")

    df[new_col] = a.combine_first(b)
    
    return df

def get_cep_clean():

    cep_df = cep_raw_df.copy()[[
        'encuesta_a',
        'encuesta_m',
        'encuesta',
    ]]

    cep_df = process_column(cep_df, 'zona_u_r', labels_zone, rename='zone')
    cep_df = process_column(cep_df, 'iden_pol_1', labels_pol, rename='pol')
    cep_df = process_column(cep_df, 'region_2', labels_region_2)
    cep_df = process_column(cep_df, 'region_3', labels_region_3)
    cep_df = process_column(cep_df, 'sexo', labels_sex, rename='sex')
    cep_df = process_column(cep_df, 'gse', labels_gse)
    cep_df = process_column(cep_df, 'esc_nivel_1_b', labels_esc_b)
    cep_df = process_column(cep_df, 'esc_nivel_1_c', labels_esc_c)
    cep_df = process_column(cep_df, 'edad')
    cep_df = process_column(cep_df, 'constitucion_1', labels_constitucion, rename='constitucion')
    
    cep_df = simplify_columns(cep_df, 'esc_nivel_1_b', 'esc_nivel_1_c', 'esc')
    cep_df = simplify_columns(cep_df, 'region_2', 'region_3', 'region')

    return cep_df