# FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

FROM python:3.10

WORKDIR /app
COPY . .
RUN apt update && apt install -y poppler-utils
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

# uvicorn main:app --host 0.0.0.0 --port 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]