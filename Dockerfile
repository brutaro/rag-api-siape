# Use uma imagem base oficial do Python
FROM python:3.10-slim

# Defina o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copie apenas o arquivo de dependências primeiro para otimizar o cache
COPY requirements.txt .

# Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt

# --- ETAPA DE DOWNLOAD DO MODELO ---
# Copie e execute o script de download para salvar o modelo na imagem
COPY download_model.py .
RUN python download_model.py
# --- FIM DA ETAPA DE DOWNLOAD ---

# Copie o código da aplicação (o main.py)
COPY main.py .

# Exponha a porta em que a aplicação irá rodar
EXPOSE 8000

# Comando para iniciar a aplicação quando o contêiner for executado
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]