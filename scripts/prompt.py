from types import SimpleNamespace as SN

def build_context(row):
    context = ("Eres una persona chilena habilitada para votar que participa "
    "en una encuesta nacional de opinión. ")
    
    if row.sex and row.edad: 
        context += f"Eres {row.sex} de {row.edad} años. "
    elif row.sex and not row.edad:
        context += f"Eres {row.sex} "
    elif row.edad and not row.sex:
        context += f"Tienes {row.edad} años. "

    if row.region:
        context += f"Vives en la región {row.region} de Chile"
        if row.zone:
            context += f", en una zona {row.zone}"
        context += ". "
    
    if row.gse: context += f"Tu nivel socioeconómico es {row.gse}. "
    if row.esc: context += f"Tu nivel educativo es {row.esc}. "
    if row.pol: 
        context += f"En la escala política, te identificas como {row.pol}. "

    context += ("Responde a la siguiente pregunta, delimitada por los símbolos"
    " <>, de acuerdo con las alternativas de respuesta delimitadas por "
    "corchetes [].")
    
    return context

class Message:
    question = None
    choices = None
    
    def __str__():

        instruct = (
            'Responde en formato JSON con las siguientes llaves: '
            '"razonamiento" (en máximo 100 palabras), donde explicas tu '
            'razonamiento para responder la pregunta, y "respuesta", que '
            'muestra la alternativa que elegiste para responder a la pregunta.'
        )

        return self.question + self.choices + self.instruct

# batch_messages = [ 
#     [ 
#         {"role": "system", "content": context}, 
#         {"role": "user", "content": ( 
#             "Pregunta: <¿Usted aprueba o rechaza una nueva Constitución?> " 
#             "Alternativas: [Apruebo, Rechazo, Ninguna]. " 
#             'Responde en formato JSON con las siguientes llaves: "razonamiento" (en máximo 100 palabras), donde explicas tu razonamiento para responder la pregunta, y "respuesta", que muestra la alternativa que elegiste para responder a la pregunta.' 
#             )
#         }
#     ] for context in batch_contexts 
# ]