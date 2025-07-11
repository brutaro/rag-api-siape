import os
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
from openai import OpenAI
from sentence_transformers import CrossEncoder
import asyncio

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG API com Reinterpretação Inteligente",
    version="2.0.0",
    description="API RAG completa com multi-query e re-ranking"
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
    top_k: Optional[int] = 7
    use_reranking: Optional[bool] = True

class QueryResponse(BaseModel):
    answer: str
    retrieved_context: str
    metadata: Dict[str, Any]

class IntelligentRAGService:
    def __init__(self):
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "siape-procedimentos")
        self.embedding_model = "text-embedding-3-small"
        self.generation_model = "gpt-4o"
        self.query_expansion_model = "gpt-3.5-turbo"
        self.openai_client = None
        self.index = None
        self.reranker = None
        
        if self.pinecone_api_key and self.openai_api_key:
            self._initialize_services()
        else:
            logger.warning("⚠️ Variáveis de ambiente não configuradas")
    
    def _initialize_services(self):
        """Inicializa serviços com cliente HTTP customizado para Railway"""
        try:
            pc = Pinecone(api_key=self.pinecone_api_key)
            self.index = pc.Index(self.index_name)
            
            http_client = httpx.Client(proxies=None, timeout=30.0)
            self.openai_client = OpenAI(
                api_key=self.openai_api_key,
                max_retries=3,
                http_client=http_client
            )
            
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("✅ Todos os serviços inicializados")
            
        except Exception as e:
            logger.error(f"❌ Erro na inicialização: {e}")
            self.index = None
            self.openai_client = None
    
    async def generate_query_variations(self, original_query: str) -> List[str]:
        if not self.openai_client: return [original_query]
        multi_query_prompt = f"""Sua tarefa é gerar 3 versões diferentes da pergunta do usuário para capturar diferentes aspectos semânticos. Pergunta Original: "{original_query}". Gere exatamente 3 variações, uma por linha:"""
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=self.query_expansion_model,
                messages=[{"role": "user", "content": multi_query_prompt}],
                temperature=0.3, max_tokens=200
            )
            raw_queries = response.choices[0].message.content.strip().split('\n')
            generated_queries = [q.strip() for q in raw_queries if q.strip()]
            if original_query not in generated_queries: generated_queries.insert(0, original_query)
            logger.info(f"Geradas {len(generated_queries)} variações da query")
            return generated_queries
        except Exception as e:
            logger.error(f"Erro na geração de queries: {e}")
            return [original_query]
    
    async def generate_embeddings_batch(self, queries: List[str]) -> List[List[float]]:
        if not self.openai_client: raise HTTPException(status_code=503, detail="OpenAI não configurado")
        try:
            response = await asyncio.to_thread(self.openai_client.embeddings.create, input=queries, model=self.embedding_model)
            embeddings = [data.embedding for data in response.data]
            logger.info(f"Gerados {len(embeddings)} embeddings em lote")
            return embeddings
        except Exception as e:
            logger.error(f"Erro na geração de embeddings: {e}")
            raise HTTPException(status_code=500, detail="Erro ao gerar embeddings")
    
    async def search_vector_store(self, query_embeddings: List[List[float]], top_k: int = 50) -> Dict[str, Dict]:
        if not self.index: raise HTTPException(status_code=503, detail="Pinecone não configurado")
        all_candidate_chunks = {}
        try:
            for embedding in query_embeddings:
                query_response = await asyncio.to_thread(self.index.query, vector=embedding, top_k=top_k, include_metadata=True)
                for match in query_response.get('matches', []):
                    if match['id'] not in all_candidate_chunks:
                        chunk_text = match['metadata'].get('content', match['metadata'].get('text', ''))
                        if chunk_text:
                            all_candidate_chunks[match['id']] = {'text': chunk_text, 'score': match['score'], 'metadata': match['metadata']}
            logger.info(f"Encontrados {len(all_candidate_chunks)} chunks únicos")
            return all_candidate_chunks
        except Exception as e:
            logger.error(f"Erro na busca vetorial: {e}")
            raise HTTPException(status_code=500, detail="Erro na busca de documentos")
    
    async def rerank_chunks(self, original_query: str, candidate_chunks: Dict[str, Dict], top_k: int = 7) -> List[str]:
        if not candidate_chunks or not self.reranker:
            sorted_chunks = sorted(candidate_chunks.values(), key=lambda x: x.get('score', 0.0), reverse=True)
            return [chunk['text'] for chunk in sorted_chunks[:top_k]]
        
        chunk_texts = [chunk_data['text'] for chunk_data in candidate_chunks.values()]
        try:
            reranker_pairs = [[original_query, chunk_text] for chunk_text in chunk_texts]
            reranker_scores = await asyncio.to_thread(self.reranker.predict, reranker_pairs)
            results_with_scores = list(zip(reranker_scores, chunk_texts))
            results_with_scores.sort(key=lambda x: x[0], reverse=True)
            final_chunks = [chunk_text for score, chunk_text in results_with_scores[:top_k]]
            logger.info(f"Re-ranking concluído: {len(final_chunks)} chunks selecionados")
            return final_chunks
        except Exception as e:
            logger.error(f"Erro no re-ranking: {e}")
            # Fallback to original score on rerank error
            sorted_chunks = sorted(candidate_chunks.values(), key=lambda x: x.get('score', 0.0), reverse=True)
            return [chunk['text'] for chunk in sorted_chunks[:top_k]]
            
    async def generate_final_answer(self, original_query: str, context: str) -> str:
        if not self.openai_client: return "OpenAI client não inicializado."
        if not context: return "Não foi possível encontrar informações para responder à sua pergunta."
        
        # <<<<< SEU PROMPT ORIGINAL E COMPLETO, REINSERIDO AQUI >>>>>
        prompt = f"""
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
        - NUNCA use listas, tópicos ou markdown. Responda em parágrafo único e coeso.
        - NUNCA realize tarefas operacionais, apenas oriente conforme o SIAPE.
        - NUNCA use hiperlinks.
        - Se a resposta para a pergunta do usuário não estiver no "Contexto Fornecido", responda apenas: "Vamos ao que interessa... Não encontrei a resposta para sua pergunta em minha base de conhecimento."
        </restricoes>

        <voz>
        - Vá direto ao ponto, sem rodeios.
        - Comece SEMPRE sua resposta com uma das seguintes frases, de forma aleatória: "Vamos ao que interessa...", "Analisando os dados enviados...", "Olha só o que temos aqui...", ou "Vamos conferir se está nos conformes...".
        - Tom profissional e objetivo, mas sem ser rude.
        - Incorpore as normativas do SIAPE que estiverem no contexto. Exemplo: "Esse procedimento segue o artigo X da Lei Y, que está no contexto."
        - Use CAPSLOCK para ÊNFASE em termos ou normativas relevantes.
        </voz>

        <foco>
        Gestão de Pessoas no Setor Público, Administração de Recursos Humanos, Procedimentos e Normativas do SIAPE, Rotinas de Cadastro e Pagamento, Benefícios e Direitos dos Servidores Públicos.
        </foco>

        Contexto Fornecido:
        ---
        {context}
        ---

        Pergunta do usuário:
        {original_query}
        """
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=self.generation_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Erro na geração da resposta: {e}")
            raise HTTPException(status_code=500, detail="Erro ao gerar resposta")

    async def process_intelligent_query(self, query: str, top_k: int = 7, use_reranking: bool = True) -> QueryResponse:
        start_time = time.time()
        if not self.openai_client or not self.index:
            raise HTTPException(status_code=503, detail="Serviços essenciais não inicializados. Verifique as chaves de API.")
        
        try:
            logger.info(f"Processando query: {query}")
            query_variations = await self.generate_query_variations(query)
            query_embeddings = await self.generate_embeddings_batch(query_variations)
            candidate_chunks = await self.search_vector_store(query_embeddings)
            if not candidate_chunks:
                return QueryResponse(answer="Não encontrei informações relevantes.", retrieved_context="", metadata={"chunks_found": 0})
            
            final_chunks = await self.rerank_chunks(query, candidate_chunks, top_k)
            context = "\n\n---\n\n".join(final_chunks)
            answer = await self.generate_final_answer(query, context)
            
            return QueryResponse(answer=answer, retrieved_context=context, metadata={"processing_time": time.time() - start_time})
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Erro no processamento da query: {e}")
            raise HTTPException(status_code=500, detail=str(e))

rag_service = IntelligentRAGService()

@app.get("/")
async def root():
    return {"message": "RAG API com Reinterpretação Inteligente"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "services": {"pinecone": rag_service.index is not None, "openai": rag_service.openai_client is not None, "reranker": rag_service.reranker is not None}}

@app.post("/query", response_model=QueryResponse)
async def process_intelligent_query_endpoint(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="A pergunta não pode estar vazia")
    return await rag_service.process_intelligent_query(query=request.question, top_k=request.top_k, use_reranking=request.use_reranking)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)