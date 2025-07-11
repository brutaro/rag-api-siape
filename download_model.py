from sentence_transformers import CrossEncoder
import os # Importe a biblioteca os

print("Iniciando o script de download de modelo...")

# REQUISITO: Ler o token do ambiente de build
token = os.getenv("HUGGING_FACE_HUB_TOKEN")

if not token:
    print("AVISO: Token do Hugging Face (HUGGING_FACE_HUB_TOKEN) não encontrado.")
else:
    print("Token do Hugging Face encontrado. Usando para autenticação.")

# Define o nome do modelo
model_name = 'sentence-transformers/ms-marco-MiniLM-L-6-v2'
# Define o caminho onde o modelo será salvo
save_path = './cross-encoder-model'

print(f"Baixando o modelo '{model_name}' para '{save_path}'...")

try:
    # REQUISITO: Passa o token para o construtor do CrossEncoder
    model = CrossEncoder(model_name, token=token)
    model.save(save_path)
    print("Download do modelo concluído com sucesso.")

except Exception as e:
    print(f"ERRO: Falha no download do modelo. Detalhes: {e}")
    # Levanta a exceção para fazer o build falhar e alertar sobre o problema.
    raise e