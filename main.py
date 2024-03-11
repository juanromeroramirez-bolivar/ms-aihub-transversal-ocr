import io
import os

import pandas as pd


from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pdf2image import convert_from_path
from PyPDF2 import PdfFileWriter, PdfReader


app = FastAPI()
puerto = os.environ.get("PORT", 8080)
max_size = 4194304


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
    file: UploadFile = File(...),):


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




@app.post("/ocr-vision-pdf")
async def ocr_vision(
    api_key: str = Form(default='SB_XF456FNDKSF'),
    tipo_documento: str = Form(default='notas de un clausulado'),
    contexto: str = Form(default='Eres un analista de seguros.'), 
    instruccion: str = Form(default='Allí encontraras en alguna página una tabla donde estan las coberturas y los topes de cada cobertura (montos de dinero en pesos colombianos). Extrae todos los valores, no ignores nada.'),
    tags: str = Form(default='es una cobertura (fila) y el valor es el texto del tope (valor asegurable, el intervalo que te indica el archivo)'),
    file: UploadFile = File(...),
    pagina_inicial: Optional[str] = None,
    pagina_final: Optional[str] = None):


    os.environ["GOOGLE_API_KEY"] = api_key
    llm = ChatGoogleGenerativeAI(model="gemini-pro-vision", temperature=0.0)
    parser = JsonOutputParser()
    chain =  llm | parser
    print('Chain instanciada')


    temporary_file_path = os.path.join("/tmp", file.filename)
    contents = await file.read()

    with open(temporary_file_path, 'wb') as f:
        f.write(contents)



    if pagina_final and pagina_inicial:
        print('caso 1')
        pagina_final = int(pagina_final)
        pagina_inicial = int(pagina_inicial)
        pages = convert_from_path(pdf_path=temporary_file_path, first_page=pagina_inicial, last_page=pagina_final)
    else:
        print('caso 2')
        if pagina_inicial:
            print('caso 3')
            pagina_inicial = int(pagina_inicial)
            pages = convert_from_path(pdf_path=temporary_file_path, first_page=pagina_inicial)
        elif pagina_final:
             print('caso 4')
             pagina_final = int(pagina_final)
             pages = convert_from_path(pdf_path=temporary_file_path, last_page=pagina_final)
        else:
            print('caso 5')
            pages = convert_from_path(pdf_path=temporary_file_path)


    data = []
    for i, page in enumerate(pages):
        path = f"./page_{i + 1}.jpg"
        page.save(path, "JPEG")
        arreglo = {"type": "image_url", "image_url": path}
        data.append((path, i+1, arreglo, os.path.getsize(path)))

    df = pd.DataFrame(data, columns = ['path','page', 'call','size'])


    # Lista para almacenar los chunks
    chunks = []
    current_chunk = []
    current_size = 0


    for index, row in df.iterrows():
        imagen_path = row["path"]
        imagen_size = os.path.getsize(imagen_path)

        if current_size + imagen_size > max_size:
            chunks.append(current_chunk)
            # Reiniciar el chunk y el tamaño actual
            current_chunk = []
            current_size = 0

        current_chunk.append(row["call"])
        # Actualizar el tamaño actual del chunk
        current_size += imagen_size

    # Agregar el último chunk si queda algo
    if current_chunk:
        chunks.append(current_chunk)


    results = {}
    for i, chunk in enumerate(chunks):
        pagina = 0

        # Prompt
        prompt = f"""
        Contexto: {contexto}
        Recibiras una serie de imagenes de un documento de {tipo_documento}
        Tienes de la pagina {pagina} hasta la pagina {pagina + len(chunk)}
        Instrucción: {instruccion}
        No puedes inventar nada, solo puedes responder con lo que esta presente en la imagen, no asumas nada.
        Usa información unicamente de la(s) imagen(es) para el valor de cada tag.
        Donde cada tag del json {tags}
        """

        content_ = content_ = [
            {
                "type": "text",
                "text": prompt,
                    }]
            
        for img in chunk:
            content_.append(img)

        

        hmessage = HumanMessage(content=content_)
        result = chain.invoke([hmessage])
        results[i] = result
        pagina = pagina + len(chunk)

    return results





if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(puerto))
