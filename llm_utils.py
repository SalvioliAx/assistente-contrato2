# llm_utils.py
"""
Este módulo agrupa todas as funções que fazem chamadas diretas ao 
Large Language Model (LLM) para realizar análises complexas.
"""
import streamlit as st
import pandas as pd
import re
import time
from typing import Optional, List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from data_models import InfoContrato, EventoContratual, ListaDeEventos
import numpy as np

@st.cache_data(show_spinner=False)
def extrair_dados_dos_contratos(_vector_store: Optional[FAISS], _nomes_arquivos: list, _t: dict) -> list:
    """Extrai informações estruturadas de múltiplos contratos usando o LLM."""
    if not _vector_store or not _nomes_arquivos: 
        return []
    
    with st.spinner(_t["extracting_data_spinner"]):
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
        resultados_finais = []
        mapa_campos_para_extracao = {
            "nome_banco_emissor": ("Qual o nome principal do banco ou instituição financeira?", "Responda apenas com o nome."),
            "valor_principal_numerico": ("Qual o valor monetário principal ou limite de crédito?", "Forneça apenas o número (ex: 10000.50)."),
            "prazo_total_meses": ("Qual o prazo total em meses?", "Forneça apenas o número de meses."),
            "taxa_juros_anual_numerica": ("Qual a principal taxa de juros anual?", "Forneça apenas o número percentual (ex: 12.5)."),
            "possui_clausula_rescisao_multa": ("Existe multa por rescisão?", "Responda 'Sim', 'Não', ou 'Não claro'."),
            "condicao_limite_credito": ("Qual a política para definir o limite de crédito?", "Resuma em uma frase."),
            "condicao_juros_rotativo": ("Quando os juros do rotativo são aplicados?", "Resuma em uma frase."),
            "condicao_anuidade": ("Qual a política de anuidade?", "Resuma em uma frase."),
            "condicao_cancelamento": ("Quais são as regras para cancelamento?", "Resuma em uma frase.")
        }
        
        barra_progresso = st.progress(0)
        total_operacoes = len(_nomes_arquivos) * len(mapa_campos_para_extracao)
        operacao_atual = 0

        for nome_arquivo in _nomes_arquivos:
            dados_contrato_atual = {"arquivo_fonte": nome_arquivo}
            retriever_arquivo_atual = _vector_store.as_retriever(
                search_type="similarity",
                # Aumentando 'k' para recuperar mais documentos relevantes
                search_kwargs={'filter': {'source': nome_arquivo}, 'k': 8} 
            )
            
            for campo, (pergunta, instrucao) in mapa_campos_para_extracao.items():
                operacao_atual += 1
                barra_progresso.progress(operacao_atual / total_operacoes, text=f"Extracting '{campo}' from {nome_arquivo}")
                
                docs_relevantes = retriever_arquivo_atual.get_relevant_documents(pergunta)
                contexto = "\n\n".join([doc.page_content for doc in docs_relevantes])

                if contexto.strip():
                    prompt_template = PromptTemplate.from_template(
                        "Contexto: {contexto}\n\nPergunta: {pergunta}\n\nInstrução: {instrucao}\nResposta:"
                    )
                    chain = LLMChain(llm=llm, prompt=prompt_template)
                    try:
                        resultado = chain.invoke({"contexto": contexto, "pergunta": pergunta, "instrucao": instrucao})
                        resposta = resultado['text'].strip()
                        
                        if "numerico" in campo or "meses" in campo:
                            numeros = re.findall(r"[\d\.,]+", resposta)
                            if numeros:
                                valor_str = numeros[0].replace('.', '').replace(',', '.')
                                dados_contrato_atual[campo] = float(valor_str) if '.' in valor_str else int(valor_str)
                            else:
                                dados_contrato_atual[campo] = None
                        else:
                            dados_contrato_atual[campo] = resposta if "não encontrado" not in resposta.lower() else "Não encontrado"
                    except Exception as e:
                        dados_contrato_atual[campo] = "Erro na IA"
                else:
                    dados_contrato_atual[campo] = "Contexto não encontrado"
                time.sleep(1)

            try:
                info_validada = InfoContrato(**dados_contrato_atual)
                resultados_finais.append(info_validada.model_dump())
            except Exception as e:
                st.error(_t["pydantic_validation_error"].format(nome_arquivo=nome_arquivo, e=e))

        barra_progresso.empty()
        return resultados_finais

@st.cache_data(show_spinner=False)
def gerar_resumo_executivo(texto_completo: str, nome_arquivo: str, _t: dict) -> str:
    """Gera um resumo executivo do conteúdo de um arquivo."""
    with st.spinner(_t["generating_summary_spinner"]):
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
        prompt = PromptTemplate.from_template(
            "Você é um assistente especialista em análise de contratos. Crie um resumo executivo em português para o documento '{nome_arquivo}'. "
            "Destaque os seguintes pontos em formato de tópicos (bullet points):\n"
            "- Partes Envolvidas (Contratante e Contratado)\n"
            "- Objeto do Contrato\n"
            "- Valores e Condições de Pagamento\n"
            "- Prazos e Vigência\n"
            "- Principais Obrigações de cada parte\n"
            "- Cláusulas sobre Rescisão e Multas\n\n"
            "Seja claro e conciso.\n\n"
            "Texto do Documento para análise:\n\n{texto}"
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.invoke({"nome_arquivo": nome_arquivo, "texto": texto_completo[:15000]})
        return response['text']

@st.cache_data(show_spinner=False)
def analisar_documento_para_riscos(texto_completo: str, nome_arquivo: str, _t: dict) -> str:
    """Analisa o texto completo de um documento e retorna uma análise de riscos."""
    with st.spinner(_t["analyzing_risks_spinner"]):
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
        prompt = PromptTemplate.from_template(
            "Você é um advogado especialista em análise de risco contratual. Analise o texto do contrato '{nome_arquivo}' e identifique "
            "cláusulas que sejam potencialmente arriscadas, ambíguas, abusivas ou desfavoráveis para uma das partes. "
            "Para cada risco identificado, cite um trecho da cláusula e explique o risco associado. "
            "Formate a saída em markdown com títulos para cada risco.\n\n"
            "Texto do Documento para análise:\n\n{texto}"
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.invoke({"nome_arquivo": nome_arquivo, "texto": texto_completo[:15000]})
        return response['text']

@st.cache_data(show_spinner=False)
def extrair_eventos_dos_contratos(textos_completos: List[Dict], _t: dict) -> List[Dict]:
    """
    Extrai datas e eventos importantes de uma lista de documentos usando o LLM
    e o esquema Pydantic para estruturar a saída.
    """
    with st.spinner(_t["extracting_deadlines_spinner"]):
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0) # Temperatura baixa para precisão na extração estruturada
        
        eventos_finais = []
        
        for doc in textos_completos:
            nome_arquivo = doc['nome']
            texto_documento = doc['texto']
            
            # Prompt para instruir o LLM a extrair eventos e datas
            prompt_template = PromptTemplate.from_template(
                "Você é um assistente especializado em análise de contratos. "
                "Sua tarefa é extrair todos os eventos contratuais importantes e suas datas correspondentes do texto fornecido. "
                "Para cada evento, forneça uma descrição concisa, a data exata no formato YYYY-MM-DD (se disponível), e o trecho relevante do contrato. "
                "Se a data não for explicitamente especificada, use 'Não Especificado'.\n\n"
                "Formate sua resposta como uma lista JSON, onde cada item é um objeto com as chaves 'descricao_evento', 'data_evento_str', e 'trecho_relevante'.\n\n"
                "Texto do Contrato '{nome_arquivo}':\n{texto}\n\n"
                "Lista de Eventos (JSON):"
            )
            
            chain = LLMChain(llm=llm, prompt=prompt_template)
            
            try:
                # Limita o texto para evitar exceder o limite de tokens do LLM
                response = chain.invoke({"nome_arquivo": nome_arquivo, "texto": texto_documento[:10000]})
                
                # Tenta parsear a resposta como JSON
                eventos_json = response['text'].strip()
                
                # Valida e adiciona os eventos usando o modelo Pydantic
                lista_eventos_validada = ListaDeEventos(eventos=[EventoContratual(**e) for e in eval(eventos_json)], arquivo_fonte=nome_arquivo)
                
                for evento in lista_eventos_validada.eventos:
                    eventos_finais.append({
                        'Arquivo Fonte': nome_arquivo,
                        'Evento': evento.descricao_evento,
                        'Data Informada': evento.data_evento_str,
                        'Trecho Relevante': evento.trecho_relevante
                    })
            except Exception as e:
                st.warning(f"Não foi possível extrair eventos do arquivo '{nome_arquivo}' devido a um erro: {e}")
                continue # Continua para o próximo arquivo mesmo com erro

        return eventos_finais


@st.cache_data(show_spinner=False)
def verificar_conformidade_documento(texto_ref: str, nome_ref: str, texto_ana: str, nome_ana: str, _t: dict) -> str:
    """Compara dois contratos e retorna um relatório de conformidade."""
    with st.spinner(_t["verifying_compliance_spinner"]):
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1)
        prompt = PromptTemplate.from_template(
            "Compare o 'Documento a Analisar' ('{nome_ana}') com o 'Documento de Referência' ('{nome_ref}'). "
            "Aponte as principais divergências em cláusulas sobre valores, prazos, multas e obrigações. "
            "Seja objetivo e liste as diferenças.\n\n"
            "### Documento de Referência ({nome_ref}):\n{texto_ref}\n\n"
            "### Documento a Analisar ({nome_ana}):\n{texto_ana}"
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.invoke({
            "nome_ref": nome_ref, "texto_ref": texto_ref[:8000],
            "nome_ana": nome_ana, "texto_ana": texto_ana[:8000]
        })
        return response['text']

def detectar_anomalias_no_dataframe(df: pd.DataFrame, t: dict) -> List[str]:
    """Analisa um DataFrame e identifica anomalias estatísticas."""
    if df.empty:
        return [t["empty_dataframe_no_anomalies"]]
    
    anomalias = []
    
    if 'taxa_juros_anual_numerica' in df.columns:
        taxas = df['taxa_juros_anual_numerica'].dropna()
        if len(taxas) > 2:
            limite = taxas.mean() + 2 * taxas.std()
            outliers = df[df['taxa_juros_anual_numerica'] > limite]
            for _, row in outliers.iterrows():
                anomalias.append(t["interest_rate_anomaly"].format(row=row))

    if 'prazo_total_meses' in df.columns:
        prazos = df['prazo_total_meses'].dropna()
        if len(prazos) > 2:
            limite_sup = prazos.mean() + 2 * prazos.std()
            limite_inf = prazos.mean() - 2 * prazos.std()
            outliers = df[(df['prazo_total_meses'] > limite_sup) | (df['prazo_total_meses'] < limite_inf)]
            for _, row in outliers.iterrows():
                anomalias.append(t["term_anomaly"].format(row=row))

    if not anomalias:
        return [t["no_significant_anomalies"]]
        
    return anomalias
