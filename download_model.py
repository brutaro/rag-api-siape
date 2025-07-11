from sentence_transformers import CrossEncoder
import os

# Define o nome do modelo
model_name = 'sentence-transformers/ms-marco-MiniLM-L-6-v2'
# Define o caminho onde o modelo será salvo dentro do container
save_path = './cross-encoder-model'

print(f"Baixando o modelo '{model_name}' para '{save_path}'...")

# Baixa o modelo e salva no caminho especificado
model = CrossEncoder(model_name)
model.save(save_path)

print("Download do modelo concluído com sucesso.")