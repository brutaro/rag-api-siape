# main.py

import os
import json
import logging
import httpx
import random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from sentence_transformers.cross_encoder import CrossEncoder
from typing import List, AsyncGenerator

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Inicialização da Aplicação FastAPI ---
app = FastAPI()

# --- Modelo de Dados (Pydantic) ---
# O request continua o mesmo
class QueryRequest(BaseModel):
    question: str

# --- Prompt da Persona "Vivi IA" ---
# O seu prompt foi adicionado aqui como uma constante para clareza
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

# --- Variáveis de Ambiente e Chaves de API ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

if not OPENAI_API_KEY:
    logger.error("A variável de ambiente OPENAI_API_KEY não foi definida.")

# --- Carregamento dos Modelos ---
try:
    logger.info("Carregando o modelo de Cross-Encoder...")
    cross_encoder = CrossEncoder('ms-marco-MiniLM-L-6-v2', device='cpu')
    logger.info("Modelo de Cross-Encoder carregado com sucesso.")
except Exception as e:
    logger.error(f"Falha ao carregar o Cross-Encoder: {e}")
    cross_encoder = None

# --- Funções do Pipeline RAG (Retrieval e Rerank) ---
# As funções de retrieval e rerank permanecem as mesmas da versão anterior.
async def retrieve_documents(query: str) -> List[str]:
    logger.info(f"Buscando documentos para a query: '{query}'")
    retrieved_docs = [
        "Para alterar a titularidade, o novo titular deve apresentar RG, CPF e um comprovante de residência.",
        "A troca de titularidade pode ser solicitada online através do portal do cliente ou presencialmente em uma de nossas agências.",
        "Não há custos para a primeira troca de titularidade do ano. Taxas podem ser aplicadas para solicitações subsequentes.",
        "O prazo para a efetivação da troca de titularidade é de até 5 dias úteis após a aprovação da documentação.",
        "Manuais de produtos e especificações técnicas estão disponíveis na seção de downloads do nosso site.",
        "O suporte técnico funciona 24 horas por dia, 7 dias por semana, através do telefone 0800-000-0000."
    ]
    logger.info(f"Documentos recuperados: {len(retrieved_docs)} documentos.")
    return retrieved_docs

def rerank_documents(query: str, documents: List[str]) -> List[str]:
    if not cross_encoder or not documents:
        return documents
    logger.info("Iniciando o processo de rerank...")
    model_input = [[query, doc] for doc in documents]
    scores = cross_encoder.predict(model_input)
    scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    reranked_docs = [doc for score, doc in scored_docs]
    logger.info(f"Documentos após o rerank (Top 3): {reranked_docs[:3]}")
    return reranked_docs

# --- Nova Função de Geração com Streaming ---
async def stream_llm_response(query: str, context_docs: List[str]) -> AsyncGenerator[str, None]:
    """
    Gera a resposta usando a API da OpenAI com streaming e retorna um gerador assíncrono.
    """
    if not context_docs:
        logger.warning("A função stream_llm_response foi chamada sem contexto.")
        yield "Vamos ao que interessa... Não encontrei a resposta para sua pergunta em minha base de conhecimento."
        return

    # Monta o prompt final para o LLM
    context = "\n\n".join(context_docs)
    # A instrução do sistema (persona) é enviada como uma mensagem separada do tipo 'system'
    # O prompt do usuário contém a pergunta e o contexto.
    user_prompt = f"""
    Contexto Fornecido:
    ---
    {context}
    ---
    Pergunta do Usuário: {query}
    """

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    # O parâmetro "stream": True é a chave para habilitar o streaming
    data = {
        "model": "gpt-4-turbo",  # Recomendo um modelo mais robusto para seguir instruções complexas
        "messages": [
            {"role": "system", "content": VIVI_IA_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "stream": True, # HABILITA O STREAMING
    }

    # Usamos um cliente HTTP para manter a conexão aberta durante o stream
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            # A requisição agora é feita com client.stream
            async with client.stream("POST", OPENAI_API_URL, headers=headers, json=data) as response:
                response.raise_for_status()
                
                # Processa a resposta em streaming
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        # Remove o prefixo "data: "
                        json_str = line[6:]
                        # Verifica o sinal de término do stream
                        if json_str == '[DONE]':
                            break
                        try:
                            # Faz o parse do JSON de cada evento
                            chunk = json.loads(json_str)
                            if chunk['choices'][0]['delta'].get('content'):
                                # Pega o pedaço de texto (token) e o envia pelo gerador
                                content_piece = chunk['choices'][0]['delta']['content']
                                yield content_piece
                        except json.JSONDecodeError:
                            logger.warning(f"Não foi possível decodificar o JSON da linha: {json_str}")
                            continue

        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            logger.error(f"Erro na requisição à API OpenAI: {e.response.status_code} - {error_body}")
            # Em caso de erro, podemos também enviar uma mensagem de erro no stream
            yield f"Erro ao comunicar com o serviço de IA: {error_body}"
        except Exception as e:
            logger.error(f"Um erro inesperado ocorreu durante o streaming: {e}")
            yield f"Ocorreu um erro interno no servidor."


# --- Endpoint da API com StreamingResponse ---
@app.post("/query")
async def handle_query_stream(request: QueryRequest):
    """
    Endpoint principal que orquestra o RAG e retorna a resposta por streaming.
    """
    logger.info(f"Recebida nova requisição para /query (stream): '{request.question}'")
    
    # 1. Retrieval
    retrieved_docs = await retrieve_documents(request.question)
    
    # 2. Rerank
    reranked_docs = rerank_documents(request.question, retrieved_docs)
    top_k_docs = reranked_docs[:3]
    
    # 3. Generation (Streaming)
    # A função stream_llm_response retorna um gerador.
    # A StreamingResponse consome esse gerador e envia os dados ao cliente.
    return StreamingResponse(
        stream_llm_response(request.question, top_k_docs),
        media_type="text/event-stream"
    )

@app.on_event("startup")
async def startup_event():
    # Adicionando as frases de início da Vivi para uso, se necessário
    app.state.vivi_starters = [
        "Vamos ao que interessa...", 
        "Analisando os dados enviados...", 
        "Olha só o que temos aqui...", 
        "Vamos conferir se está nos conformes..."
    ]
    logger.info("✅ Todos os serviços inicializados com sucesso.")