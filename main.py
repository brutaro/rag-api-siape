# main.py

import os
import json
import logging
import httpx
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from sentence_transformers.cross_encoder import CrossEncoder
from typing import List, AsyncGenerator, Set

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Inicialização da Aplicação FastAPI ---
app = FastAPI(title="Vivi IA - RAG Service")

# --- REQUISITO 2: Prompt da Persona "Vivi IA" ---
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

# --- Modelo de Dados (Pydantic) ---
class QueryRequest(BaseModel):
    question: str

# --- Configurações e Chaves de API ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

if not OPENAI_API_KEY:
    raise ValueError("A variável de ambiente OPENAI_API_KEY deve ser configurada.")

# --- Carregamento de Modelos na Inicialização ---
cross_encoder = None

@app.on_event("startup")
def startup_event():
    global cross_encoder
    logger.info("Iniciando carregamento do modelo Cross-Encoder...")
    try:
        # AQUI ESTÁ A CORREÇÃO: Adicionado o prefixo "sentence-transformers/"
        model_name = 'sentence-transformers/ms-marco-MiniLM-L-6-v2'
        cross_encoder = CrossEncoder(model_name, device='cpu')
        logger.info(f"✅ Modelo Cross-Encoder '{model_name}' carregado com sucesso.")
    except Exception as e:
        logger.error(f"❌ Falha crítica ao carregar o Cross-Encoder: {e}")
        raise RuntimeError(f"Não foi possível carregar o Cross-Encoder: {e}")

# --- Funções do Pipeline RAG ---

async def enrich_and_generate_queries(query: str, client: httpx.AsyncClient) -> List[str]:
    # --- REQUISITO 1: Enriquecimento de Perguntas ---
    logger.info(f"Enriquecendo a pergunta: '{query}'")
    prompt = f"""Gere 3 variações da pergunta a seguir para uma busca em uma base de conhecimento sobre SIAPE. Retorne uma lista JSON com a chave "queries". Pergunta: "{query}" """
    data = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}}
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    try:
        response = await client.post(OPENAI_API_URL, headers=headers, json=data, timeout=20.0)
        response.raise_for_status()
        generated_queries = response.json().get("queries", [])
        if query not in generated_queries: generated_queries.append(query)
        logger.info(f"Perguntas geradas: {generated_queries}")
        return generated_queries
    except Exception as e:
        logger.error(f"Falha no enriquecimento da pergunta: {e}. Usando apenas a pergunta original.")
        return [query]

async def retrieve_documents(query: str) -> List[str]:
    # --- REQUISITO 4 (PARTE 1): Lógica de busca de documentos ---
    logger.info(f"Buscando documentos para a query: '{query}'")
    mock_knowledge_base = [f"Documento simulado número {i} sobre diversas normativas e procedimentos." for i in range(150)]
    return mock_knowledge_base[:50]

def rerank_documents(original_query: str, documents: List[str]) -> List[str]:
    # --- REQUISITO 4 (PARTE 2): Reranking com Cross-Encoder ---
    if not cross_encoder or not documents: return documents
    logger.info(f"Reordenando {len(documents)} documentos com base na pergunta original...")
    model_input = [[original_query, doc] for doc in documents]
    scores = cross_encoder.predict(model_input, show_progress_bar=True)
    scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs]

async def stream_llm_response(query: str, context_docs: List[str], client: httpx.AsyncClient) -> AsyncGenerator[str, None]:
    # --- REQUISITO 3: Geração de Resposta por Streaming ---
    context = "\n\n".join(context_docs)
    user_prompt = f"Contexto Fornecido:\n---\n{context}\n---\n\nPergunta do Usuário: {query}"
    data = {
        "model": "gpt-4-turbo",
        "messages": [
            {"role": "system", "content": VIVI_IA_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    async with client.stream("POST", OPENAI_API_URL, headers=headers, json=data, timeout=180.0) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line.startswith('data: '):
                json_str = line[6:]
                if json_str == '[DONE]': break
                try:
                    chunk = json.loads(json_str)
                    if content := chunk['choices'][0]['delta'].get('content'):
                        yield content
                except json.JSONDecodeError: continue

# --- Endpoint Principal da API ---
@app.post("/query")
async def handle_query_stream(request: QueryRequest):
    original_question = request.question
    logger.info(f"--- INICIANDO NOVO FLUXO DE REQUISIÇÃO: '{original_question}' ---")
    
    unique_docs: Set[str] = set()
    
    async with httpx.AsyncClient() as client:
        # 1. Enriquecimento
        queries = await enrich_and_generate_queries(original_question, client)
        
        # 2. Retrieval
        retrieval_tasks = [retrieve_documents(q) for q in queries]
        list_of_docs = await asyncio.gather(*retrieval_tasks)
        for doc_list in list_of_docs: unique_docs.update(doc_list)
        logger.info(f"Total de {len(unique_docs)} documentos únicos recuperados para rerank.")
        
        # 3. Rerank
        reranked_docs = rerank_documents(original_question, list(unique_docs))
        
        top_k_context = reranked_docs[:5]
        logger.info(f"Top 5 documentos selecionados como contexto final.")
        
        # 4. Geração com Streaming
        return StreamingResponse(
            stream_llm_response(original_question, top_k_context, client),
            media_type="text/event-stream"
        )