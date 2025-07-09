import requests
import json

# --- CONFIGURAÇÃO ---
API_URL = "https://rag-api-siape-production.up.railway.app/query"
QUESTION = "Para que serve o comando >CAVADIRFEX?"

# --- EXECUÇÃO DA CHAMADA À API ---
print(f"Enviando pergunta para a API na nuvem...")
print(f"URL: {API_URL}")
print(f"Pergunta: {QUESTION}")

payload = {"question": QUESTION}
headers = {"Content-Type": "application/json"}

try:
    # Usamos stream=True para lidar com a resposta como um fluxo
    with requests.post(API_URL, headers=headers, json=payload, stream=True) as response:
        response.raise_for_status()
        
        buffer = ""
        sources_found = False
        answer_started = False

        # Itera sobre os pedaços (chunks) da resposta conforme eles chegam
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            buffer += chunk

            # Procura pelo nosso separador '---' para separar as fontes da resposta
            if not sources_found and "\n---\n" in buffer:
                parts = buffer.split("\n---\n", 1)
                json_part = parts[0]
                buffer = parts[1]  # O resto é o início da resposta da IA

                try:
                    sources_data = json.loads(json_part)
                    print("\n" + "="*50)
                    print("FONTES CONSULTADAS:")
                    print(sources_data.get("sources", "Nenhuma fonte encontrada."))
                    print("="*50)
                except json.JSONDecodeError:
                    print("ERRO: Não foi possível decodificar o JSON das fontes.")
                
                sources_found = True

            # Imprime a resposta da IA conforme ela chega
            if sources_found:
                if not answer_started:
                    print("\nRESPOSTA DA VIVI IA:")
                    print("="*50)
                    answer_started = True
                
                print(buffer, end="", flush=True)
                buffer = ""  # Limpa o buffer depois de imprimir

        # Imprime qualquer parte final que sobrou no buffer
        if buffer:
            print(buffer, end="", flush=True)
        print() # Adiciona uma nova linha no final

except requests.exceptions.RequestException as e:
    print(f"\nERRO: Falha ao se conectar com a API: {e}")

