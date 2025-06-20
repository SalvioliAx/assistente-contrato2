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
from data_models import InfoContrato

# --- ASSINATURAS DAS FUNÇÕES ATUALIZADAS PARA ACEITAR 'api_key' ---

@st.cache_data(show_spinner="Extraindo dados detalhados dos contratos...")
def extrair_dados_dos_contratos(_vector_store, _nomes_arquivos, api_key) -> list:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, google_api_key=api_key)
    # ... (resto do código da função)
    return []

@st.cache_data(show_spinner="Gerando resumo executivo...")
def gerar_resumo_executivo(texto_completo, nome_arquivo, api_key) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=api_key)
    # ... (resto do código da função)
    return ""

@st.cache_data(show_spinner="Analisando cláusulas de risco...")
def analisar_documento_para_riscos(texto_completo, nome_arquivo, api_key) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2, google_api_key=api_key)
    # ... (resto do código da função)
    return ""

@st.cache_data(show_spinner="Extraindo prazos e eventos...")
def extrair_eventos_dos_contratos(textos_completos, api_key) -> List[Dict]:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1, google_api_key=api_key)
    # ... (resto do código da função)
    return []

@st.cache_data(show_spinner="Verificando conformidade...")
def verificar_conformidade_documento(texto_ref, nome_ref, texto_ana, nome_ana, api_key) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1, google_api_key=api_key)
    # ... (resto do código da função)
    return ""

def detectar_anomalias_no_dataframe(df: pd.DataFrame) -> List[str]:
    # (esta função não usa o LLM, não precisa de api_key)
    # ... (código inalterado)
    return []
