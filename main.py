import os
import logging
import time
import asyncio
import json
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
from openai import OpenAI
from sentence_transformers import CrossEncoder
from typing import AsyncGenerator, List, Dict

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- INICIALIZAÇÃO DOS SERVIÇOS ---
try:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "siape-procedimentos")

    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        raise ValueError("Chaves de API PINECONE_API_KEY e OPENAI_API_KEY são necessárias.")

    logger.info("Inicializando clientes...")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    http_client = httpx.Client(proxies=None, timeout=30.0)
    client = OpenAI(api_key=OPENAI_API_KEY, max_retries=3, http_client=http_client)

    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    logger.info("✅ Todos os serviços inicializados com sucesso.")

except Exception as e:
    logger.error(f"❌ ERRO CRÍTICO NA INICIALIZAÇÃO: {e}")
    index = None
    client = None
    reranker_model = None

# --- CONFIGURAÇÃO DA API FASTAPI ---
app = FastAPI(
    title="RAG API com Streaming",
    version="5.1.0",
    description="API RAG com multi-query, re-ranking e resposta via streaming"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    top_k_initial: int = 50
    top_k_final: int = 7

# --- LÓGICA RAG ---

async def get_context_and_sources(original_query: str, top_k_initial: int, top_k_final: int) -> (str, List[str]):
    if not client or not index or not reranker_model:
        raise HTTPException(status_code=503, detail="Serviços essenciais não inicializados.")

    multi_query_prompt = f"""Sua tarefa é gerar 3 versões diferentes da pergunta do usuário para capturar diferentes aspectos semânticos. Pergunta Original: "{original_query}". Gere exatamente 3 variações, uma por linha:"""
    response = await asyncio.to_thread(client.chat.completions.create, model="gpt-3.5-turbo", messages=[{"role": "user", "content": multi_query_prompt}], temperature=0.3, max_tokens=200)
    raw_queries = response.choices[0].message.content.strip().split('\n')
    generated_queries = [q.strip() for q in raw_queries if q.strip()]
    if original_query not in generated_queries: generated_queries.insert(0, original_query)

    embeddings_response = await asyncio.to_thread(client.embeddings.create, input=generated_queries, model="text-embedding-3-small")
    query_embeddings = [data.embedding for data in embeddings_response.data]

    all_candidate_chunks = {}
    for embedding in query_embeddings:
        query_response = await asyncio.to_thread(index.query, vector=embedding, top_k=top_k_initial, include_metadata=True)
        for match in query_response.get('matches', []):
            if match['id'] not in all_candidate_chunks:
                all_candidate_chunks[match['id']] = match
    
    if not all_candidate_chunks:
        return "", []

    candidate_matches = list(all_candidate_chunks.values())
    reranker_input_pairs = [[original_query, match['metadata'].get('text', '')] for match in candidate_matches]
    reranker_scores = await asyncio.to_thread(reranker_model.predict, reranker_input_pairs)
    
    results_with_scores = list(zip(reranker_scores, candidate_matches))
    results_with_scores.sort(key=lambda x: x[0], reverse=True)
    top_results = results_with_scores[:top_k_final]
    
    final_context_chunks = [match['metadata'].get('text', '') for _, match in top_results]
    context = "\n\n---\n\n".join(final_context_chunks)
    sources = list(set([match['metadata'].get('source', 'N/A') for _, match in top_results]))
    
    return context, sources

async def stream_final_answer(prompt: str) -> AsyncGenerator[str, None]:
    try:
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1500,
            stream=True
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except Exception as e:
        logger.error(f"Erro no streaming da OpenAI: {e}")
        yield "Desculpe, um erro ocorreu ao gerar a resposta."

# --- ENDPOINT PRINCIPAL COM STREAMING ---
@app.post("/query")
async def process_query_streaming(request: QueryRequest):

    async def response_generator():
        try:
            context, sources = await get_context_and_sources(request.question, request.top_k_initial, request.top_k_final)

            sources_payload = json.dumps({"sources": sources})
            yield f"{sources_payload}\n---\n"

            if not context:
                yield "Vamos ao que interessa... Não encontrei a resposta para sua pergunta em minha base de conhecimento."
                return

            # <<<<< SEU PROMPT COMPLETO E DETALHADO FOI INSERIDO AQUI >>>>>
            final_prompt = f"""
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

            Contexto Fornecido:
            ---
            {context}
            ---

            Com base estrita no contexto e mantendo sua persona, responda à pergunta do usuário.
            Pergunta do usuário: {request.question}
            """
            
            async for chunk in stream_final_answer(final_prompt):
                yield chunk

        except Exception as e:
            logger.error(f"Erro fatal no processamento do streaming: {e}")
            error_payload = json.dumps({"sources": ["Erro no Servidor"]})
            yield f"{error_payload}\n---\nOcorreu um erro interno: {str(e)}"

    return StreamingResponse(response_generator(), media_type="text/event-stream")