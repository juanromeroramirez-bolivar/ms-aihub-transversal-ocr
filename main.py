import io
import os

from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser



app = FastAPI()
puerto = os.environ.get("PORT", 8080)



# Configuración de CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ocr-vision")
async def ocr_vision(
    api_key: str = Form(default='SB_XF456FNDKSF'),
    tipo_documento: str = Form(default='nota de un clausulado de un seguro'),
    contexto: str = Form(default='Eres un analista de seguros.'), 
    instruccion: str = Form(default='Allí encontraras una tabla donde estan las coberturas y los topes de cada cobertura (montos de dinero en pesos colombianos). Extrae todos los valores, no ignores nada.'),
    tags: str = Form(default='es una cobertura (fila) y el valor es el texto del tope (valor asegurable, el intervalo que te indica el archivo)'),
    file: UploadFile = File(...)):


    os.environ["GOOGLE_API_KEY"] = api_key

    llm = ChatGoogleGenerativeAI(model="gemini-pro-vision", temperature=0.0)
    parser = JsonOutputParser()
    chain =  llm | parser
    print('Chain instanciada')

    # File processing
    imagen_bytes = await file.read()
    img = Image.open(io.BytesIO(imagen_bytes))
    print("Imagen leida")

    # Prompt
    prompt = f"""
    Contexto: {contexto}
    Recibiras la imagen de {tipo_documento}
    Instrucción: {instruccion}
    No puedes inventar nada, solo puedes responder con lo que esta presente en la imagen, no asumas nada.
    Usa información unicamente de la(s) imagen(es) para el valor de cada tag.
    Donde cada tag del json {tags}
    """

    # Procesar las imágenes y obtener la descripción del siniestro
    hmessage = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt,
            },
            {"type": "image_url", "image_url": img},
        ]
    )

    print('Previo al procesamiento')
    result = chain.invoke([hmessage])
    print("Procesado")

    return result



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(puerto))
