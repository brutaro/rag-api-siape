import os
import traceback
import asyncio
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import AsyncGenerator, List, Dict, Optional

# --- Seção de importações do RAG (existente) ---
from pinecone import Pinecone
from openai import OpenAI
from sentence_transformers import CrossEncoder

# --- Seção de importações do LangChain (NOVO) ---
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.memory import VectorStoreRetrieverMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


# --- CONFIGURAÇÃO E INICIALIZAÇÃO ---
# Chaves de API (sem alterações)
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

print("Iniciando a API e carregando os modelos...")
try:
    # Modelos e clientes existentes
    pc = Pinecone(api_key=PINECONE_API_KEY)
    client = OpenAI(api_key=OPENAI_API_KEY) # Cliente OpenAI padrão para streaming
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    index = pc.Index("siape-procedimentos")

    # Componentes LangChain (NOVO)
    # Modelo de embedding para a memória da conversa
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    # LLM do LangChain para uso na "Chain"
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2, api_key=OPENAI_API_KEY)
    
    print("Modelos e conexão com o índice estabelecidos com sucesso.")
except Exception as e:
        print(f"!!!!!!!!!!!! ERRO INESPERADO DURANTE O PROCESSAMENTO DA QUERY !!!!!!!!!!!!")
        traceback.print_exc()
        # A correção é passar 'e' como um argumento para a função
        async def exception_stream(exception_obj: Exception):
            yield json.dumps({"sources": ["Erro no Servidor"]}) + "\n---\n"
            yield f"Ocorreu um erro interno no servidor: {str(exception_obj)}"
        
        return StreamingResponse(exception_stream(e), media_type="text/plain; charset=utf-8")
    # ... (lógica de erro existente) ...

# --- GERENCIADOR DE MEMÓRIA (NOVO) ---
# ATENÇÃO: Este gerenciador de memória em dicionário é para DEMONSTRAÇÃO.
# Ele será resetado a cada reinicio da API e não é seguro para múltiplos workers.
# Em produção, use um banco de dados externo como Redis.
conversation_memories: Dict[str, VectorStoreRetrieverMemory] = {}

def get_or_create_memory(session_id: str) -> VectorStoreRetrieverMemory:
    """Busca ou cria uma memória de conversa para uma dada sessão."""
    if session_id not in conversation_memories:
        print(f"Criando nova memória para a sessão: {session_id}")
        # Cria um vector store FAISS em memória para guardar o histórico desta sessão
        vectorstore = FAISS(embedding_function=embedding_model, index=None, docstore=None) # Será populado dinamicamente
        retriever = vectorstore.as_retriever(search_kwargs=dict(k=4)) # Busca os 4 turnos mais relevantes da conversa
        # Cria a memória com o retriever
        conversation_memories[session_id] = VectorStoreRetrieverMemory(retriever=retriever, memory_key="chat_history")
    
    return conversation_memories[session_id]

# --- FastAPI App e CORS (sem alterações) ---
app = FastAPI(title="API de RAG com Memória Inteligente")
# ... (código do CORS middleware) ...

# --- Pydantic Model (ALTERADO) ---
class QueryRequest(BaseModel):
    question: str
    session_id: str # O frontend agora envia um ID de sessão
    top_k_initial: int = 50
    top_k_final: int = 7


# --- RAG Pipeline (sem alterações na busca, apenas no uso) ---
def get_document_context(original_query: str, top_k: int, final_k: int):
    # 1. Multi-Query (exatamente como antes)
    multi_query_prompt = f"Sua tarefa é gerar 3 versões diferentes... Pergunta Original: \"{original_query}\""
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": multi_query_prompt}], temperature=0.3, max_tokens=200)
    raw_queries = response.choices[0].message.content.strip().split('\n')
    generated_queries = [q.strip() for q in raw_queries if q.strip()]
    
    # 2. Busca Vetorial Ampla (exatamente como antes)
    embeddings_response = client.embeddings.create(input=generated_queries, model="text-embedding-3-small")
    query_embeddings = [data.embedding for data in embeddings_response.data]
    all_candidate_chunks = {}
    for query_embedding in query_embeddings:
        query_response = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        for match in query_response['matches']:
            if match['id'] not in all_candidate_chunks:
                all_candidate_chunks[match['id']] = match
    
    candidate_matches = list(all_candidate_chunks.values())
    if not candidate_matches:
        return None, []

    # 3. Re-ranking (exatamente como antes)
    reranker_input_pairs = [[original_query, match['metadata']['text']] for match in candidate_matches]
    reranker_scores = reranker_model.predict(reranker_input_pairs)
    results_with_scores = list(zip(reranker_scores, candidate_matches))
    results_with_scores.sort(key=lambda x: x[0], reverse=True)
    
    top_results = results_with_scores[:final_k]
    final_context_chunks = [match['metadata']['text'] for score, match in top_results]
    sources = list(set([match['metadata']['source'] for score, match in top_results]))
    
    return "\n---\n".join(final_context_chunks), sources


@app.post("/query")
async def process_query(request: QueryRequest):
    try:
        # 1. Obter a memória da sessão atual
        memory = get_or_create_memory(request.session_id)
        
        # 2. Carregar histórico relevante da conversa
        # A memória busca no seu vector store interno os turnos relevantes para a nova pergunta
        relevant_history = memory.load_memory_variables({"prompt": request.question})['chat_history']

        # 3. Obter contexto dos documentos (sua lógica RAG existente)
        document_context, sources = get_document_context(request.question, request.top_k_initial, request.top_k_final)
        
        if not document_context:
             async def error_stream():
                yield json.dumps({"sources": []}) + "\n---\n"
                yield "A informação não foi encontrada na base de conhecimento."
             return StreamingResponse(error_stream(), media_type="text/plain; charset=utf-8")

        # 4. Construir o Prompt Final com LangChain
        prompt_template = PromptTemplate(
            input_variables=["chat_history", "document_context", "question"],
            template="""
            Assuma a persona de "Vivi IA", uma especialista em Gestão de Pessoas do serviço público federal brasileiro, focada em SIAPE.
            - Sua comunicação é clara, objetiva, formal e um pouco calorosa.
            - NUNCA especule. Se a resposta não estiver no contexto ou no histórico, diga "A informação sobre este tópico não foi encontrada na minha base de conhecimento."
            - Responda SEMPRE com base EXCLUSIVA no contexto dos documentos e no histórico da conversa.

            Histórico da Conversa Relevante:
            {chat_history}

            Contexto dos Documentos:
            ---
            {document_context}
            ---

            Com base no histórico e no contexto acima, responda à pergunta do usuário.
            Pergunta: {question}
            """
        )

        final_prompt = prompt_template.format(
            chat_history=relevant_history,
            document_context=document_context,
            question=request.question
        )
        
        # 5. Gerar e Streamar a Resposta
        async def stream_and_save():
            # Envia as fontes primeiro
            sources_payload = json.dumps({"sources": sources})
            yield f"{sources_payload}\n---\n"
            
            full_response = ""
            # Usa o cliente OpenAI padrão para manter o streaming de antes
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.2,
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    yield content
                    await asyncio.sleep(0.01)
            
            # 6. Salvar o novo turno na memória DEPOIS de gerar a resposta
            memory.save_context({"prompt": request.question}, {"output": full_response})
            print(f"Contexto salvo para a sessão: {request.session_id}")

        return StreamingResponse(stream_and_save(), media_type="text/plain; charset=utf-8")

    except Exception as e:
        print(f"!!!!!!!!!!!! ERRO INESPERADO DURANTE O PROCESSAMENTO DA QUERY !!!!!!!!!!!!")
        traceback.print_exc()
        async def exception_stream():
            yield json.dumps({"sources": ["Erro no Servidor"]}) + "\n---\n"
            yield f"Ocorreu um erro interno no servidor: {e}"
        return StreamingResponse(exception_stream(), media_type="text/plain; charset=utf-8")