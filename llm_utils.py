# llm_utils.py
import streamlit as st
import pandas as pd
import re
import time
from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from data_models import InfoContrato, ListaDeEventos

# --- AS ASSINATURAS DAS FUNÇÕES FORAM SIMPLIFICADAS ---
# Já não precisam de receber 'api_key' como parâmetro.

@st.cache_data(show_spinner="Extraindo dados detalhados dos contratos...")
def extrair_dados_dos_contratos(_vector_store, _nomes_arquivos) -> list:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    parser = PydanticOutputParser(pydantic_object=InfoContrato)

    prompt = PromptTemplate(
        template="""
        Analise o seguinte texto de contrato e extraia as informações solicitadas.
        Se uma informação não for encontrada, use o valor padrão definido no schema.
        Texto do Contrato: "{texto_documento}"
        Arquivo de Origem: "{nome_arquivo}"
        {format_instructions}
        """,
        input_variables=["texto_documento", "nome_arquivo"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    resultados = []
    for nome in _nomes_arquivos:
        texto_completo = "\n".join([doc.page_content for doc_id, doc in _vector_store.docstore._dict.items() if doc.metadata.get('source') == nome])
        if texto_completo:
            with st.spinner(f"Analisando detalhes de {nome}..."):
                try:
                    output = chain.run(texto_documento=texto_completo, nome_arquivo=nome)
                    parsed_output = parser.parse(output)
                    # Força o nome do arquivo, pois o LLM pode errar
                    parsed_output.arquivo_fonte = nome 
                    resultados.append(parsed_output.dict())
                except Exception as e:
                    st.error(f"Erro ao processar o arquivo {nome}: {e}")
    return resultados


@st.cache_data(show_spinner="Gerando resumo executivo...")
def gerar_resumo_executivo(texto_completo, nome_arquivo) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    prompt = PromptTemplate.from_template(
        """
        Você é um assistente jurídico especializado em simplificar documentos complexos.
        Crie um resumo executivo claro e conciso (máximo de 5 parágrafos) do seguinte contrato.
        O resumo deve destacar:
        1. As partes envolvidas.
        2. O objeto principal do contrato.
        3. Os valores e condições de pagamento mais importantes.
        4. O prazo de vigência e condições de rescisão.
        5. Quaisquer obrigações ou responsabilidades críticas para o contratante.
        Responda em português do Brasil.

        Contrato (originado do arquivo {nome_arquivo}):
        ---
        {texto_contrato}
        ---
        Resumo Executivo:
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    resumo = chain.run({"texto_contrato": texto_completo, "nome_arquivo": nome_arquivo})
    return resumo

# <<< INÍCIO DA CORREÇÃO >>>
# A função abaixo estava faltando e foi adicionada.
@st.cache_data(show_spinner="Analisando cláusulas de risco...")
def analisar_documento_para_riscos(texto_completo: str, nome_arquivo: str) -> str:
    """
    Analisa um documento para identificar cláusulas de risco.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.4)
    prompt = PromptTemplate.from_template(
        """
        Você é um advogado especialista em análise de risco contratual.
        Sua tarefa é ler o contrato abaixo, originado do arquivo '{nome_arquivo}', e identificar potenciais riscos, ambiguidades e cláusulas desfavoráveis para a parte contratante.
        Organize sua análise nos seguintes tópicos em formato Markdown:
        
        - **🚩 Riscos Financeiros:** (Ex: multas, juros altos, taxas escondidas, ausência de limites de responsabilidade)
        - **⚖️ Riscos Operacionais e de Conformidade:** (Ex: obrigações de difícil cumprimento, prazos irreais, cláusulas de rescisão abrupta, leis aplicáveis desfavoráveis)
        - **🤔 Ambiguidade e Omissões:** (Ex: termos mal definidos, falta de especificações, ausência de cláusulas importantes como confidencialidade ou proteção de dados)
        - **⚠️ Pontos de Atenção Críticos:** Um resumo dos 2-3 pontos que exigem maior atenção imediata.

        Se não encontrar riscos em uma categoria, indique "Nenhum risco aparente encontrado.".
        Seja objetivo e cite trechos do contrato quando relevante.

        Contrato para Análise:
        ---
        {texto_contrato}
        ---
        Relatório de Análise de Riscos:
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    analise = chain.run({"texto_contrato": texto_completo, "nome_arquivo": nome_arquivo})
    return analise
# <<< FIM DA CORREÇÃO >>>

@st.cache_data(show_spinner="Extraindo prazos e eventos dos contratos...")
def extrair_eventos_dos_contratos(documentos: List[Dict[str, str]]) -> list:
    """
    Extrai eventos e datas de uma lista de documentos de texto.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    parser = PydanticOutputParser(pydantic_object=ListaDeEventos)

    prompt = PromptTemplate(
        template="""
        Analise o texto do contrato abaixo, originado do arquivo '{nome_arquivo}'.
        Sua tarefa é identificar e listar TODOS os eventos, prazos, vencimentos ou datas importantes mencionados.
        Para cada evento, extraia uma descrição clara, a data (se especificada) e o trecho relevante do texto.
        {format_instructions}

        Texto do Contrato:
        ---
        {texto_contrato}
        ---
        """,
        input_variables=["texto_contrato", "nome_arquivo"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    todos_os_eventos = []
    for doc in documentos:
        nome_arquivo = doc['nome']
        texto = doc['texto']
        with st.spinner(f"Extraindo eventos de {nome_arquivo}..."):
            try:
                output = chain.run(texto_contrato=texto, nome_arquivo=nome_arquivo)
                parsed_output = parser.parse(output)
                for evento in parsed_output.eventos:
                    todos_os_eventos.append({
                        "arquivo_fonte": parsed_output.arquivo_fonte,
                        "descricao_evento": evento.descricao_evento,
                        "data_evento": evento.data_evento_str,
                        "trecho_relevante": evento.trecho_relevante
                    })
            except Exception as e:
                st.warning(f"Não foi possível extrair eventos de '{nome_arquivo}': {e}")
                
    return todos_os_eventos


@st.cache_data(show_spinner="Verificando conformidade entre documentos...")
def verificar_conformidade_documento(texto_referencia, nome_referencia, texto_analisado, nome_analisado) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.2)
    prompt = PromptTemplate.from_template(
        """
        Você é um auditor de conformidade. Sua tarefa é comparar dois documentos e gerar um relatório de conformidade.
        O Documento de Referência é o padrão a ser seguido.
        O Documento em Análise deve ser comparado com o de referência.

        Relatório de Conformidade:
        - Documento de Referência: {nome_referencia}
        - Documento em Análise: {nome_analisado}

        1.  **Resumo da Comparação:** Faça um breve resumo das semelhanças e diferenças gerais.
        2.  **Pontos de Conformidade:** Liste as principais cláusulas ou termos em que o '{nome_analisado}' está em conformidade com o '{nome_referencia}'.
        3.  **Pontos de Divergência (Desvios):** Liste as principais cláusulas ou termos onde o '{nome_analisado}' diverge do '{nome_referencia}'. Seja específico e, se possível, cite os trechos.
        4.  **Recomendações:** Com base nos desvios, sugira ações para ajustar o '{nome_analisado}' para que fique em conformidade com o documento de referência.

        DOCUMENTO DE REFERÊNCIA:
        ---
        {texto_referencia}
        ---

        DOCUMENTO EM ANÁLISE:
        ---
        {texto_analisado}
        ---

        Elabore o Relatório de Conformidade em formato Markdown:
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    resultado = chain.run({
        "nome_referencia": nome_referencia,
        "nome_analisado": nome_analisado,
        "texto_referencia": texto_referencia,
        "texto_analisado": texto_analisado
    })
    return resultado
    
@st.cache_data(show_spinner="Buscando anomalias nos dados...")
def detectar_anomalias_no_dataframe(df: pd.DataFrame) -> List[str]:
    """
    Analisa um DataFrame de dados de contratos para detectar anomalias usando um LLM.
    """
    if df.empty:
        return ["DataFrame está vazio, nenhuma anomalia para detectar."]

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5)
    
    # Converte o dataframe para um formato de string legível pelo LLM
    dados_str = df.to_markdown(index=False)

    prompt = PromptTemplate.from_template(
        """
        Você é um analista de dados financeiros sênior. Sua tarefa é analisar o conjunto de dados de contratos abaixo e identificar anomalias, outliers ou padrões incomuns.
        Procure por:
        - Taxas de juros que são muito mais altas ou baixas que a média.
        - Prazos de contrato que são excessivamente longos ou curtos.
        - Valores principais que se desviam significativamente dos outros.
        - Contratos do mesmo banco com condições muito diferentes.
        - Contratos sem valores numéricos claros onde outros têm.

        Liste cada anomalia encontrada como um item de uma lista, explicando por que você a considera uma anomalia. Se nenhum padrão incomum for encontrado, retorne uma lista com o item "Nenhuma anomalia significativa foi detectada.".

        Dados dos Contratos:
        {dados_contratos}

        Análise de Anomalias (formato de lista):
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    resultado_str = chain.run({"dados_contratos": dados_str})
    
    # Processa o resultado para garantir que é uma lista de strings
    anomalias = [item.strip() for item in resultado_str.split('\n') if item.strip() and item.strip().startswith('-')]
    
    if not anomalias:
        return ["Nenhuma anomalia significativa foi detectada."]
        
    return anomalias
