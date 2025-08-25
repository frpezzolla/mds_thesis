#%% setup run
import json
from pathlib import Path
import pandas as pd
import torch, torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import (AutoTokenizer,AutoModelForCausalLM)
import re
import ast
import argparse
from scripts.get_cep import get_cep_clean
from scripts.prompt import build_context
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


from transformers import logging
logging.set_verbosity_error()

# set up paths
ROOT_DIR = Path.cwd()
DATA_DIR = ROOT_DIR / "data"

print(ROOT_DIR)


def clean_row(row):
    return row.replace({r'^\s*<NA>\s*$': pd.NA, r'^\s*$': pd.NA}, regex=True).dropna()

# clean = lambda s: s.replace({r"^\s*<NA>\s*$": pd.NA, r"^\s*$": pd.NA}, regex=True).dropna().astype("string")

def fix_common_json_errors(text):
    # Remove leading/trailing code block markers (if any)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.I)
        text = text.rstrip("`").strip()
    
    # Replace ; or . at end of key-value lines with commas, *but not inside string values*
    # Only between fields: a quote, then ; or ., then whitespace/newline, then next key
    # Use a regex for this (doesn't catch all cases but covers most LLM output):
    text = re.sub(r'"\s*[;.] *\n?\s*"', '",\n"', text)
    # And at the end of the first key-value (if the next field doesn't start with a quote):
    text = re.sub(r'"\s*[;.] *\n?\s*(\w)', r'",\n"\1', text)
    
    # Replace any ; or . directly before a quote (common error):
    text = re.sub(r'([;.] *)(")', r',\2', text)

    # Remove trailing commas before closing curly brace (also common):
    text = re.sub(r',\s*}', '}', text)
    
    # Ensure all keys/values are wrapped in double quotes (for JSON)
    # If you are OK with ast.literal_eval, skip this, but for JSON:
    # text = re.sub(r"'", '"', text)  # careful, could break inner apostrophes

    return text


#%% run simulation

# from pydantic import BaseModel

# class Response(BaseModel):
#     reasoning: str
#     answer: str


def main(args):
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key= os.getenv('HF_TOKEN'),
    )

    cep_df = get_cep_clean()
    cep_df['constitucion'] = cep_df['constitucion'].replace({r"^\s*<NA>\s*$": pd.NA}, regex=True).astype("string")
    cep_df = cep_df.dropna(subset=['constitucion'])

    if args.n_sims == -1:
        args.n_sims = len(cep_df)

    rows = []
    malformed = []
    
    for _,row in cep_df.iloc[:args.n_sims].iterrows():
        context = build_context(row)

        messages = [ 
                {"role": "system", "content": context}, 
                {"role": "user", "content": ( 
                    "Pregunta: <¿Usted aprueba o rechaza una nueva Constitución?> " 
                    "Alternativas: [Apruebo, Rechazo, Ninguna]. " 
                    'Responde en formato JSON con las siguientes llaves: "razonamiento" (en máximo 100 palabras), donde explicas tu razonamiento para responder la pregunta, y "respuesta", que muestra la alternativa que elegiste para responder a la pregunta.' 
                    )
                }
            
        ]

        completion = client.chat.completions.create(
            model = args.model,
            messages = messages,
            # response_format = {
            #     "type": "json_schema",
            #     "json_schema": {
            #         "reasoning": "Response",
            #         "schema": Response.model_json_schema(),
            #         "strict": True
            #     }
            # }
        )

        content = completion.choices[0].message.content

        print(content)
        print(row.constitucion)

        print(50*"-")

        match = re.search(r'\{[\s\S]*?\}', content)
        cleaned_text = match.group()

        result = json.loads(cleaned_text)

        print(result.get('respuesta'))

def set_model(arg: str):
    if arg == 'LLaMA-3.2-1B-Instruct':
        return "meta-llama/Llama-3.2-1B-Instruct"
    elif arg == 'LLaMA-3.2-3B-Instruct':
        return "meta-llama/Llama-3.2-3B-Instruct"
    elif arg == 'Kimi-K2-Instruct':
        return 'moonshotai/Kimi-K2-Instruct'
    elif arg == 'Qwen-4B-Thinking':
        return 'Qwen/Qwen3-4B-Thinking-2507'
    else:
        return "meta-llama/Llama-3.2-1B-Instruct"
    
def set_device(arg: str):
    if arg in ['0','1']:
        return f"cuda:{arg}"
    else:
        raise ValueError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=set_model, nargs='?', default='LLaMA-3.2-1B-Instruct')
    parser.add_argument('-o', '--output', type=Path, nargs='?', default=None)
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('-n', '--n-sims', type=int, default=-1)
    parser.add_argument('--device', type=set_device, default='0')

    args = parser.parse_args()

    print(vars(args))

    main(
        args,
    )