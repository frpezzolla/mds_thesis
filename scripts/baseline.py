#%% setup run
import json
from pathlib import Path
import pandas as pd
import torch, torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import (AutoTokenizer,AutoModelForCausalLM)
import re
import ast

from transformers import logging
logging.set_verbosity_error()

# set up paths
ROOT_DIR = Path.cwd()
DATA_DIR = ROOT_DIR / "data"

print(ROOT_DIR)
from scripts.get_cep import get_cep_clean


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

cep_df = get_cep_clean()
cep_df['constitucion'] = cep_df['constitucion'].replace({r"^\s*<NA>\s*$": pd.NA}, regex=True).astype("string")
cep_df = cep_df.dropna(subset=['constitucion'])

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda"
N_SIMS = len(cep_df)
BATCH_SIZE = 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

model.eval()

rows = []
malformed = []
for batch_start in tqdm(range(0, N_SIMS, BATCH_SIZE), desc='simulating'):

    batch_samples = cep_df.iloc[batch_start:batch_start+BATCH_SIZE]
    
    # batch_contexts = [
    #     f"You are a Chilean survey respondent with the right to vote participating in a national opinion survey. You are {sample.edad} year old {sample.sex}. Your live in the {sample.region} region of Chile, in {sample.zone} area. Your socioeconomic level is {sample.gse}. Your education level is {sample.esc}. On the political scale, you identify as {sample.pol}."
    #     "The question will be delimited with the following symbols <>, and the response alternatives by square brackets []." for _,sample in batch_samples.iterrows()
    # ]

    # batch_messages = [
    #     [
    #         {"role": "system", "content": context},
    #         {"role": "user","content": (
    #             "Question: <Which of the following options did you vote for in the Chilean exit plebiscite for the new constitution in September 2022?>"
    #             "Possible answers: [Approve, Reject, None]"
    #             "Deliver a single JSON object with the following keys: 'explanation' (IN MAXIMUM 100 WORDS) where you will write your reasoning to answer the question, and 'answer' that shows the alternative chosen to answer that question. Answer as honestly as possible."
    #         )},
    #         {"role": "assistant", "content": assistant_prefix}
    #         ] for context in batch_contexts
    #     ]

    batch_contexts = [
        f"Eres una persona chilena habilitada para votar que participa en una encuesta nacional de opinión. Eres {sample.sex} de {sample.edad} años. Vives en la región {sample.region} de Chile, en una zona {sample.zone}. Tu nivel socioeconómico es {sample.gse}. Tu nivel educativo es {sample.esc}. En la escala política, te identificas como {sample.pol}."
        "Responde a la siguiente pregunta, delimitada por los símbolos <>, de acyerdi con las alternativas de respuesta delimitadas por corchetes []." for _, sample in batch_samples.iterrows()
    ]

    batch_messages = [ 
        [ 
            {"role": "system", "content": context}, 
            {"role": "user", "content": ( 
                "Pregunta: <¿Por cuál de las siguientes opciones votaste en el plebiscito de salida para la nueva constitución en Chile en septiembre de 2022?> " 
                "Alternativas: [Apruebo, Rechazo, Ninguna]. " 
                'Responde en formato JSON con las siguientes llaves: "razonamiento" (en máximo 100 palabras), donde explicas tu razonamiento para responder la pregunta, y "respuesta", que muestra la alternativa que elegiste para responder a la pregunta.' 
                )
            }
        ] for context in batch_contexts 
    ]

    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer.apply_chat_template(
        batch_messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_attention_mask=True,
        padding=True,
        truncation=True,
    )
    
    attention_mask = torch.ones_like(inputs)
    inputs = {"input_ids": inputs.to(DEVICE), "attention_mask": attention_mask.to(DEVICE)}


    start = inputs['input_ids'].shape[0]

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=500,
            # temperature=1.,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
for i, (gen_text, sample) in enumerate(zip(gen_texts, batch_samples.itertuples())):
    match = re.search(r'\{[\s\S]*?\}', gen_text)
    clean_text = match.group() if match else "{}"
    # Clean common JSON errors
    clean_text = fix_common_json_errors(clean_text)
    try:
        result = json.loads(clean_text)
    except Exception:
        try:
            result = ast.literal_eval(clean_text)
        except Exception:
            malformed.append(gen_text)
            result = {"razonamiento": 'E', "respuesta": 'E'}

    explanation = result.get('razonamiento') or 'E'
    simulated_answer = result.get('respuesta') or 'E'
    real_answer = sample.constitucion

    rows.append({
        "simulation_id": batch_start + i,
        "real_answer": real_answer,
        "simulated_answer": simulated_answer,
        "explanation": explanation,
        "edad": sample.edad,
        "sex": sample.sex,
        "region": sample.region,
        "zone": sample.zone,
        "gse": sample.gse,
        "esc": sample.esc,
        "pol": sample.pol,
    })


answers = pd.DataFrame(rows).to_csv(DATA_DIR / 'generated' / 'simulated_baseline_.csv', index=False)
with open(DATA_DIR / 'generated' / 'malformed_answers.txt', 'w', encoding='utf-8') as f:
    for txt in malformed:
        f.write(txt)
        f.write('\n' + '-'*40 + '\n')