import requests
import uuid
import sys

# URL da sua API no Railway
API_URL = "https://rag-api-siape-production.up.railway.app/query"

# Gera um ID de sessão único para este teste
session_id = f"test-session-{uuid.uuid4()}"

# Pega a pergunta dos argumentos da linha de comando ou usa uma padrão
question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Para que serve o comando >CAVADIRFEX?"

print("Enviando pergunta para a API na nuvem...")
print(f"URL: {API_URL}")
print(f"Pergunta: {question}")
print(f"Session ID: {session_id}")

# Cria o payload no formato correto, com 'question' e 'session_id'
payload = {
    "question": question,
    "session_id": session_id
}

try:
    # Faz a requisição POST para a API
    response = requests.post(API_URL, json=payload, stream=True)
    
    # Verifica se a resposta foi bem sucedida
    response.raise_for_status() 

    print("\n--- Resposta da Vivi IA (Streaming) ---")
    # Itera sobre a resposta em streaming
    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        # O separador de fontes vem no primeiro chunk
        if '---' in chunk:
            parts = chunk.split('---', 1)
            print(f"Fontes: {parts[0].strip()}")
            print(parts[1], end='', flush=True)
        else:
            print(chunk, end='', flush=True)
    print("\n-------------------------------------\n")


except requests.exceptions.HTTPError as http_err:
    print(f"\nERRO: Falha ao se conectar com a API: {http_err}")
    print(f"Detalhes: {response.text}") # Mostra mais detalhes do erro
except Exception as err:
    print(f"\nERRO: Ocorreu um erro inesperado: {err}")