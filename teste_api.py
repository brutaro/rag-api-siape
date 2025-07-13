import requests
import json
import sys

# --- CONFIGURAÇÃO ---
# URL da sua API. Railway geralmente expõe na porta 443 (HTTPS), então não é preciso especificar a porta 8080.
API_URL = "https://rag-api-siape-production.up.railway.app/query"

def query_rag_api(question: str):
    """
    Envia uma pergunta para a API RAG, processa a resposta de streaming
    e exibe as fontes e a resposta da IA em tempo real.
    """
    print("-" * 50)
    print(f"Enviando pergunta: \"{question}\"")
    print(f"Para a API em: {API_URL}")
    print("-" * 50)

    # Payload da requisição POST
    payload = {
        "question": question,
        "top_k_initial": 50, # Valor padrão do seu código
        "top_k_final": 7     # Valor padrão do seu código
    }

    try:
        # Inicia a requisição com streaming ativado
        with requests.post(API_URL, json=payload, stream=True, timeout=300) as response:
            # Verifica se a requisição foi bem-sucedida (código 2xx)
            response.raise_for_status()

            header_buffer = b''
            header_processed = False
            separator = b'\n---\n'

            print("\nAguardando resposta da IA...\n")

            # Itera sobre os chunks da resposta de streaming
            for chunk in response.iter_content(chunk_size=None):
                if not chunk:
                    continue

                if not header_processed:
                    header_buffer += chunk
                    # Procura pelo separador no buffer acumulado
                    if separator in header_buffer:
                        header_part, body_part = header_buffer.split(separator, 1)
                        
                        try:
                            # Decodifica e processa o cabeçalho com as fontes
                            sources_data = json.loads(header_part.decode('utf-8'))
                            sources = sources_data.get("sources", [])
                            
                            print("=" * 20)
                            print("Fontes Encontradas:")
                            if sources:
                                for source in sources:
                                    print(f"- {source}")
                            else:
                                print("Nenhuma fonte encontrada.")
                            print("=" * 20)
                            print("\nResposta da IA:")

                        except json.JSONDecodeError:
                            print("[ERRO] Não foi possível decodificar o cabeçalho das fontes.")
                        
                        header_processed = True
                        # Imprime a primeira parte do corpo da resposta que já foi recebida
                        sys.stdout.write(body_part.decode('utf-8'))
                        sys.stdout.flush()
                else:
                    # Se o cabeçalho já foi processado, apenas imprime o conteúdo
                    sys.stdout.write(chunk.decode('utf-8'))
                    sys.stdout.flush()
        
        print("\n\n--- Fim da transmissão ---")

    except requests.exceptions.RequestException as e:
        print(f"\n[ERRO DE CONEXÃO] Não foi possível se conectar à API: {e}")
    except Exception as e:
        print(f"\n[ERRO INESPERADO] Ocorreu um erro: {e}")


if __name__ == "__main__":
    print("Cliente de teste para a API RAG com Streaming.")
    print("Digite 'sair' a qualquer momento para fechar o programa.")
    
    while True:
        user_question = input("\nDigite sua pergunta sobre o SIAPE: ")
        if user_question.lower() == 'sair':
            break
        if not user_question.strip():
            print("Por favor, digite uma pergunta válida.")
            continue
        
        query_rag_api(user_question)

