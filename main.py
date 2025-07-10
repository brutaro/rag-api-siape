import os
import traceback
import asyncio
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware # <<<<< NOVA IMPORTAÇÃO >>>>>
from pydantic import BaseModel
from pinecone import Pinecone
from openai import OpenAI
from sentence_transformers import CrossEncoder
from typing import AsyncGenerator, List

# --- CONFIGURAÇÃO E INICIALIZAÇÃO (sem alterações) ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

print("Iniciando a API e carregando os modelos...")
try:
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        raise ValueError("Chaves de API não encontradas nas variáveis de ambiente.")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    client = OpenAI(api_key=OPENAI_API_KEY)
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    index = pc.Index("siape-procedimentos")
    print("Modelos e conexão com o índice estabelecidos com sucesso.")
except Exception as e:
    print(f"ERRO CRÍTICO NA INICIALIZAÇÃO: {e}")
    reranker_model = None
    index = None

print("API pronta para receber requisições.")
app = FastAPI(title="API de RAG com Streaming e Fontes")

# <<<<< INÍCIO DA CORREÇÃO DE CORS >>>>>
# Adiciona o middleware de CORS para permitir que o frontend se comunique com a API.
origins = ["*"] # Permite todas as origens. Para produção, restrinja a domínios específicos.

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Permite todos os métodos (GET, POST, etc.)
    allow_headers=["*"], # Permite todos os cabeçalhos
)
# <<<<< FIM DA CORREÇÃO DE CORS >>>>>


class QueryRequest(BaseModel):
    question: str
    top_k_initial: int = 50
    top_k_final: int = 7

# --- O RESTO DO CÓDIGO (get_context, stream_final_answer, process_query) ---
# --- CONTINUA EXATAMENTE O MESMO DE ANTES ---

def get_context(original_query: str, top_k: int):
    # ... (lógica de busca)
    multi_query_prompt = f"Sua tarefa é gerar 3 versões diferentes... Pergunta Original: \"{original_query}\""
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": multi_query_prompt}], temperature=0.3, max_tokens=200)
    raw_queries = response.choices[0].message.content.strip().split('\n')
    generated_queries = [q.strip() for q in raw_queries if q.strip()]
    embeddings_response = client.embeddings.create(input=generated_queries, model="text-embedding-3-small")
    query_embeddings = [data.embedding for data in embeddings_response.data]
    all_candidate_chunks = {}
    for query_embedding in query_embeddings:
        query_response = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        for match in query_response['matches']:
            if match['id'] not in all_candidate_chunks:
                all_candidate_chunks[match['id']] = match
    return list(all_candidate_chunks.values()) if all_candidate_chunks else None

async def stream_final_answer(final_prompt: str, sources: List[str]) -> AsyncGenerator[str, None]:
    try:
        sources_payload = json.dumps({"sources": sources})
        yield f"{sources_payload}\n---\n"
        stream = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": final_prompt}], temperature=0.2, stream=True)
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
                await asyncio.sleep(0.01)
    except Exception as e:
        print(f"Erro durante o streaming: {e}")
        yield "Desculpe, ocorreu um erro ao gerar a resposta."

@app.post("/query")
async def process_query(request: QueryRequest):
    try:
        if not reranker_model or not index:
            raise HTTPException(status_code=500, detail="API não foi inicializada corretamente.")
        candidate_matches = get_context(request.question, request.top_k_initial)
        if not candidate_matches:
            async def error_stream():
                yield json.dumps({"sources": []}) + "\n---\n"
                yield "A informação não foi encontrada na base de conhecimento (etapa de busca)."
            return StreamingResponse(error_stream(), media_type="text/plain; charset=utf-8")
        reranker_input_pairs = [[request.question, match['metadata']['text']] for match in candidate_matches]
        reranker_scores = reranker_model.predict(reranker_input_pairs)
        results_with_scores = list(zip(reranker_scores, candidate_matches))
        results_with_scores.sort(key=lambda x: x[0], reverse=True)
        top_results = results_with_scores[:request.top_k_final]
        final_context_chunks = [match['metadata']['text'] for score, match in top_results]
        final_context = "\n---\n".join(final_context_chunks)
        sources = list(set([match['metadata']['source'] for score, match in top_results]))
        final_prompt = f"""
        Assuma a persona de "Vivi IA"... (seu prompt completo aqui)
        Contexto Fornecido: --- {final_context} ---
        Pergunta do usuário: {request.question}
        """
        return StreamingResponse(stream_final_answer(final_prompt, sources), media_type="text/plain; charset=utf-8")
    except Exception as e:
        print(f"!!!!!!!!!!!! ERRO INESPERADO DURANTE O PROCESSAMENTO DA QUERY !!!!!!!!!!!!")
        traceback.print_exc()
        async def exception_stream():
            yield json.dumps({"sources": ["Erro no Servidor"]}) + "\n---\n"
            yield f"Ocorreu um erro interno no servidor: {e}"
        return StreamingResponse(exception_stream(), media_type="text/plain; charset=utf-8")
