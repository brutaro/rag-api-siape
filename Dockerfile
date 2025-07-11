# Use uma imagem base oficial do Python
FROM python:3.10-slim

# Defina o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copie o arquivo de dependências para o diretório de trabalho
COPY requirements.txt .

# Instale as dependências
# O --no-cache-dir reduz o tamanho da imagem
RUN pip install --no-cache-dir -r requirements.txt

# Copie o código da aplicação (o main.py) para o diretório de trabalho
COPY main.py .

# Exponha a porta em que a aplicação irá rodar (Uvicorn usa a 8000 por padrão)
EXPOSE 8000

# Comando para iniciar a aplicação quando o contêiner for executado
# O --host 0.0.0.0 torna a aplicação acessível de fora do contêiner
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]