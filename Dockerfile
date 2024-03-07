FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download es_core_news_md
EXPOSE 8080

# uvicorn main:app --host 0.0.0.0 --port 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]