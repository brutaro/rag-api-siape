# main.py (VERSÃO FINAL - FOCO EM BAIXO USO DE MEMÓRIA)

import os
import json
import logging
import httpx
import asyncio
import time
import gc # Garbage Collector
import psutil # Para monitorar uso de recursos

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from sentence_transformers.cross_encoder import CrossEncoder
from typing import List, AsyncGenerator, Set

# --- 1. CONFIGURAÇÃO INICIAL ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Vivi IA - RAG Service")

class QueryRequest(BaseModel):
    question: str

# --- 2. PROMPT E VARIÁVEIS DE AMBIENTE ---
VIVI_IA_SYSTEM_PROMPT = """
Você é Vivi IA, uma versão IA da Vivi. Especialista em gestão pública e no SIAPE, responde com precisão e objetividade. Fale na primeira pessoa, sendo direta e eficiente, mas sem tolerar preguiça ou falta de esforço. Suas respostas são EXTREMAMENTE estruturadas, fundamentadas nas normativas do SIAPE e seguem sua <voz>.

            <instrucoes>
            - SEMPRE siga suas etapas em <etapas>.
            - SEMPRE responda no mesmo idioma da pergunta.
            - O CONTEXTO FORNECIDO ABAIXO É A SUA ÚNICA BASE DE CONHECIMENTO.
            - NUNCA procure informações na internet ou fora do contexto.
            - NUNCA mencione os nomes dos arquivos da sua base de conhecimento.
            </instrucoes>

            <restricoes>
            - NUNCA responda perguntas fora de <foco>, retome para SIAPE e gestão pública.
            - Para suas explicações, escreva em parágrafos coesos e evite usar listas ou tópicos com marcadores (bullets). A formatação especial só é permitida ao citar textos de lei.
            - NUNCA realize tarefas operacionais, apenas oriente conforme o SIAPE.
            - NUNCA use hiperlinks.
            - Se a resposta para a pergunta do usuário não estiver no "Contexto Fornecido", responda apenas: "Vamos ao que interessa... Não encontrei a resposta para sua pergunta em minha base de conhecimento."
            </restricoes>

            <voz>
            - Vá direto ao ponto, sem rodeios.
            - Comece SEMPRE sua resposta com uma das seguintes frases, de forma aleatória: "Vamos ao que interessa...", "Analisando os dados enviados...", "Olha só o que temos aqui...", ou "Vamos conferir se está nos conformes...".
            - AO CITAR O TEXTO DE UMA LEI OU NORMATIVA, TRANSCREVA-O FIELMENTE, MANTENDO AS QUEBRAS DE LINHA E A ESTRUTURA ORIGINAL com seus incisos (I, II, a), b), etc.). Esta é a única exceção à regra de não usar listas.
            - Use CAPSLOCK para ÊNFASE em termos ou normativas relevantes.
            - Exemplo de como citar: "Esse procedimento segue o Artigo 132, que define como penalidade a demissão por 'crime contra a administração pública' (inciso I) e por 'improbidade administrativa' (inciso IV)."
            </voz>

            <foco>
            Gestão de Pessoas no Setor Público, Administração de Recursos Humanos, Procedimentos e Normativas do SIAPE, Rotinas de Cadastro e Pagamento, Benefícios e Direitos dos Servidores Públicos.
            </foco>
"""
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

if not OPENAI_API_KEY:
    raise ValueError("A variável de ambiente OPENAI_API_KEY deve ser configurada.")

# --- 3. FUNÇÃO DE LOG DE MEMÓRIA ---
def log_memory_usage(stage: str):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"[MEMORY] Stage: {stage} - RSS: {mem_info.rss / 1024 ** 2:.2f} MB")

# --- 4. CARREGAMENTO DO MODELO ---
cross_encoder = None

@app.on_event("startup")
def startup_event():
    global cross_encoder
    log_memory_usage("Pre-Startup")
    logger.info("Iniciando carregamento do modelo Cross-Encoder local...")
    try:
        model_path = './cross-encoder-model'
        cross_encoder = CrossEncoder(model_path, device='cpu')
        logger.info(f"✅ Modelo Cross-Encoder carregado com sucesso do caminho '{model_path}'.")
        log_memory_usage("Post-Model-Load")
    except Exception as e:
        logger.error(f"❌ Falha crítica ao carregar o Cross-Encoder local: {e}")
        raise RuntimeError(f"Não foi possível carregar o Cross-Encoder local: {e}")

# --- 5. FUNÇÕES DA PIPELINE RAG (COM OTIMIZAÇÃO DE MEMÓRIA) ---

async def enrich_and_generate_queries(query: str, client: httpx.AsyncClient) -> List[str]:
    # (Função sem alterações, já é leve)
    ...

async def retrieve_documents(query: str) -> List[str]:
    # (Função sem alterações, já é leve)
    ...

def rerank_documents_memory_efficient(original_query: str, documents: List[str], batch_size: int = 16) -> List[str]:
    """ Reranks documents in batches to save memory. """
    if not cross_encoder or not documents:
        return documents
    
    logger.info(f"Iniciando rerank eficiente de {len(documents)} documentos em lotes de {batch_size}...")
    
    scored_docs = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        model_input = [[original_query, doc] for doc in batch]
        
        log_memory_usage(f"Rerank-Predict-Batch-{i//batch_size}")
        scores = cross_encoder.predict(model_input, show_progress_bar=False)
        
        for doc, score in zip(batch, scores):
            scored_docs.append((score, doc))
        
        # Força a liberação de memória após cada lote
        gc.collect()

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    log_memory_usage("Post-Rerank-Sort")
    
    return [doc for score, doc in scored_docs]

async def stream_llm_response(query: str, context_docs: List[str], client: httpx.AsyncClient) -> AsyncGenerator[str, None]:
    # (Função sem alterações)
    ...

# --- 6. ENDPOINT PRINCIPAL COM CONTROLE DE RECURSOS ---
@app.post("/query")
async def handle_query_stream(request: QueryRequest):
    start_time = time.time()
    log_memory_usage("Request-Start")
    
    original_question = request.question
    logger.info(f"--- INICIANDO FLUXO COM CONTROLE DE MEMÓRIA: '{original_question}' ---")
    
    async with httpx.AsyncClient() as client:
        # Etapa 1: Enriquecimento
        queries_to_search = [original_question] + await enrich_and_generate_queries(original_question, client)
        log_memory_usage("Post-Enrichment")

        # Etapa 2: Busca de Documentos
        unique_docs: Set[str] = set()
        retrieval_tasks = [retrieve_documents(q) for q in queries_to_search]
        list_of_docs = await asyncio.gather(*retrieval_tasks)
        for doc_list in list_of_docs:
            unique_docs.update(doc_list)
        
        docs_list = list(unique_docs)
        unique_docs = set() # Libera a memória do set
        gc.collect()
        log_memory_usage("Post-Retrieval")

        # Etapa 3: Rerank com uso eficiente de memória
        reranked_docs = rerank_documents_memory_efficient(original_question, docs_list)
        docs_list = [] # Libera a memória da lista
        gc.collect()

        top_k_context = reranked_docs[:5]
        logger.info(f"Tempo total antes do streaming: {time.time() - start_time:.2f}s.")
        log_memory_usage("Pre-Streaming")
        
        # Etapa 4: Geração
        return StreamingResponse(
            stream_llm_response(original_question, top_k_context, client),
            media_type="text/event-stream"
        )