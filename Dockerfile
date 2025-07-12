FROM python:3.10-slim
WORKDIR /app

# Copia as dependências e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia a pasta do modelo (que está localmente) para dentro da imagem
COPY cross-encoder-model/ ./cross-encoder-model/

# Copia o código da aplicação
COPY main.py .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]