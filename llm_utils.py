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
from data_models import InfoContrato, ListaDeEventos

@st.cache_data(show_spinner="Extraindo dados detalhados...")
def extrair_dados_dos_contratos(_vector_store, _nomes_arquivos, api_key) -> list:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, google_api_key=api_key)
    parser = PydanticOutputParser(pydantic_object=InfoContrato)
    prompt = PromptTemplate(
        template="Analise o texto do contrato do arquivo '{nome_arquivo}' e extraia as informações.\n{format_instructions}\nTexto: \"{texto_documento}\"",
        input_variables=["texto_documento", "nome_arquivo"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    resultados = []
    for nome in _nomes_arquivos:
        texto_completo = "\n".join([doc.page_content for doc in _vector_store.docstore._dict.values() if doc.metadata.get('source') == nome])
        if texto_completo:
            try:
                output = chain.run(texto_documento=texto_completo, nome_arquivo=nome)
                parsed_output = parser.parse(output)
                parsed_output.arquivo_fonte = nome 
                resultados.append(parsed_output.dict())
            except Exception as e:
                st.error(f"Erro ao processar {nome}: {e}")
    return resultados

@st.cache_data(show_spinner="Gerando resumo...")
def gerar_resumo_executivo(texto_completo, nome_arquivo, api_key) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate.from_template("Crie um resumo executivo do contrato '{nome_arquivo}'.\nTexto:{texto_contrato}\nResumo:")
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"texto_contrato": texto_completo, "nome_arquivo": nome_arquivo})

@st.cache_data(show_spinner="Analisando riscos...")
def analisar_documento_para_riscos(texto_completo, nome_arquivo, api_key) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.4, google_api_key=api_key)
    prompt = PromptTemplate.from_template("Analise o contrato '{nome_arquivo}' e identifique riscos.\nTexto:{texto_contrato}\nRelatório de Riscos:")
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"texto_contrato": texto_completo, "nome_arquivo": nome_arquivo})

@st.cache_data(show_spinner="Extraindo prazos...")
def extrair_eventos_dos_contratos(documentos, api_key) -> list:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, google_api_key=api_key)
    parser = PydanticOutputParser(pydantic_object=ListaDeEventos)
    prompt = PromptTemplate(
        template="Analise o texto do contrato do arquivo '{nome_arquivo}' e liste todos os eventos e prazos.\n{format_instructions}\nTexto:{texto_contrato}",
        input_variables=["texto_contrato", "nome_arquivo"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    todos_os_eventos = []
    for doc in documentos:
        try:
            output = chain.run(texto_contrato=doc['texto'], nome_arquivo=doc['nome'])
            parsed_output = parser.parse(output)
            for evento in parsed_output.eventos:
                todos_os_eventos.append({
                    "arquivo_fonte": parsed_output.arquivo_fonte,
                    "descricao_evento": evento.descricao_evento,
                    "data_evento": evento.data_evento_str,
                    "trecho_relevante": evento.trecho_relevante
                })
        except Exception as e:
            st.warning(f"Não foi possível extrair eventos de '{doc['nome']}': {e}")
    return todos_os_eventos

@st.cache_data(show_spinner="Verificando conformidade...")
def verificar_conformidade_documento(texto_referencia, nome_referencia, texto_analisado, nome_analisado, api_key) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.2, google_api_key=api_key)
    prompt = PromptTemplate.from_template("Compare o documento '{nome_analisado}' com o de referência '{nome_referencia}'.\nRef:{texto_referencia}\nAna:{texto_analisado}\nRelatório:")
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({
        "nome_referencia": nome_referencia, "nome_analisado": nome_analisado,
        "texto_referencia": texto_referencia, "texto_analisado": texto_analisado
    })
    
@st.cache_data(show_spinner="Buscando anomalias...")
def detectar_anomalias_no_dataframe(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return ["DataFrame está vazio."]
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5)
    dados_str = df.to_markdown(index=False)
    prompt = PromptTemplate.from_template("Analise os dados e encontre anomalias.\nDados:{dados_contratos}\nAnomalias:")
    chain = LLMChain(llm=llm, prompt=prompt)
    resultado_str = chain.run({"dados_contratos": dados_str})
    anomalias = [item.strip() for item in resultado_str.split('\n') if item.strip().startswith('-')]
    return anomalias if anomalias else ["Nenhuma anomalia significativa foi detectada."]
