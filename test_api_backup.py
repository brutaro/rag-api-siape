import requests
import json
import sys
import textwrap

API_URL = "https://rag-api-siape-production.up.railway.app/query"
question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Quais as regras para o auxílio-funeral?"

payload = {"question": question}

print(f"Enviando pergunta: {question}\n")

try:
    # Usamos stream=True para receber a resposta em partes
    with requests.post(API_URL, json=payload, stream=True, timeout=90) as response:
        response.raise_for_status()
        
        sources_received = False
        full_answer = ""
        
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if not sources_received and '---' in chunk:
                # Processa o primeiro chunk que contém as fontes
                parts = chunk.split('\n---\n', 1)
                sources_json_string = parts[0]
                remaining_chunk = parts[1]
                
                try:
                    sources_data = json.loads(sources_json_string)
                    print(f"Fontes Consultadas: {sources_data.get('sources')}\n")
                except json.JSONDecodeError:
                    print(f"Erro ao decodificar fontes: {sources_json_string}")

                full_answer += remaining_chunk
                print(remaining_chunk, end='', flush=True)
                sources_received = True
            elif sources_received:
                # Processa os chunks seguintes da resposta da IA
                full_answer += chunk
                print(chunk, end='', flush=True)

    print("\n\n--- Fim da Resposta ---")

except requests.exceptions.RequestException as e:
    print(f"\nERRO: Falha na conexão com a API. {e}")