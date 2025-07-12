# main.py (BASEADO NO SEU CÓDIGO QUE FUNCIONAVA)

import os
import json
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from sentence_transformers import CrossEncoder # Importação correta
from typing import AsyncGenerator, List
import asyncio

# --- 1. INICIALIZAÇÃO GLOBAL (COMO NO SEU CÓDIGO ORIGINAL) ---
logger.info("Iniciando a API e carregando os modelos...")
try:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("Chave da OpenAI não encontrada.")

    # Cliente OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Carregando o modelo localmente (A ÚNICA ADAPTAÇÃO NECESSÁRIA)
    logger.info("Carregando CrossEncoder do diretório local...")
    reranker_model = CrossEncoder('./cross-encoder-model', device='cpu')
    
    logger.info("✅ Modelos e clientes inicializados com sucesso.")

except Exception as e:
    logger.error(f"ERRO CRÍTICO NA INICIALIZAÇÃO: {e}")
    reranker_model = None

# --- 2. CONFIGURAÇÃO DO FASTAPI (COMO NO SEU CÓDIGO ORIGINAL) ---
app = FastAPI(title="API de RAG - Versão Estável")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

# --- 3. FUNÇÕES DA PIPELINE (ADAPTADAS DO SEU FLUXO) ---

def get_context_mock(original_query: str) -> List[dict]:
    """ Simula a busca de documentos que antes era feita com Pinecone. """
    logger.info(f"Buscando documentos para: '{original_query}'")
    # Simula a recuperação de 50 documentos
    mock_matches = []
    for i in range(50):
        mock_matches.append({
            'id': f'doc_{i}',
            'metadata': {
                'text': f'Este é o texto do documento simulado número {i} sobre o tema {original_query}.',
                'source': f'fonte_{i}.pdf'
            }
        })
    return mock_matches

async def stream_final_answer(final_prompt: str, sources: List[str]) -> AsyncGenerator[str, None]:
    """ Função de streaming usando o cliente OpenAI. """
    try:
        # Envia as fontes primeiro, como no seu código original
        sources_payload = json.dumps({"sources": sources})
        yield f"{sources_payload}\n---\n"
        
        # Faz o streaming da resposta do LLM
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.2,
            stream=True
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
                await asyncio.sleep(0.01) # Pequeno delay para o streaming fluir
    except Exception as e:
        logger.error(f"Erro durante o streaming: {e}")
        yield "Desculpe, ocorreu um erro ao gerar a resposta."

# --- 4. ENDPOINT PRINCIPAL (SEGUINDO A LÓGICA DO SEU CÓDIGO) ---
@app.post("/query")
async def process_query(request: QueryRequest):
    try:
        if not reranker_model:
            raise HTTPException(status_code=503, detail="Serviço indisponível: modelo de rerank não carregado.")

        # 1. Busca de Contexto (usando nossa simulação)
        candidate_matches = get_context_mock(request.question)
        if not candidate_matches:
            # Lógica de erro se nada for encontrado
            ...

        # 2. Rerank (lógica síncrona, como no seu original)
        logger.info(f"Iniciando rerank de {len(candidate_matches)} documentos...")
        reranker_input_pairs = [[request.question, match['metadata']['text']] for match in candidate_matches]
        reranker_scores = reranker_model.predict(reranker_input_pairs)
        logger.info("Rerank concluído.")

        # 3. Processamento dos resultados
        results_with_scores = list(zip(reranker_scores, candidate_matches))
        results_with_scores.sort(key=lambda x: x[0], reverse=True)
        
        top_results = results_with_scores[:7] # Usando top_k_final = 7
        final_context_chunks = [match['metadata']['text'] for score, match in top_results]
        final_context = "\n---\n".join(final_context_chunks)
        sources = list(set([match['metadata']['source'] for score, match in top_results]))
        
        # 4. Geração da Resposta Final
        final_prompt = f"""
        (Você é Vivi IA, uma versão IA da Vivi. Especialista em gestão pública e no SIAPE, responde com precisão e objetividade. Fale na primeira pessoa, sendo direta e eficiente, mas sem tolerar preguiça ou falta de esforço. Suas respostas são EXTREMAMENTE estruturadas, fundamentadas nas normativas do SIAPE e seguem sua <voz>.

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
            </foco>)

        Contexto Fornecido: --- {final_context} ---
        Pergunta do usuário: {request.question}
        """

        logger.info("Contexto final pronto. Iniciando streaming da resposta.")
        return StreamingResponse(stream_final_answer(final_prompt, sources), media_type="text/plain; charset=utf-8")

    except Exception as e:
        logger.error("!!!!!!!!!!!! ERRO INESPERADO DURANTE O PROCESSAMENTO DA QUERY !!!!!!!!!!!!")
        traceback.print_exc()
        # Lógica de streaming de erro
        ...