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

# --- AS ASSINATURAS DAS FUNÇÕES FORAM SIMPLIFICADAS ---
# Já não precisam de receber 'api_key' como parâmetro.

@st.cache_data(show_spinner="Extraindo dados detalhados dos contratos...")
def extrair_dados_dos_contratos(_vector_store, _nomes_arquivos) -> list:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    # ... (resto do código da função)
    return []

@st.cache_data(show_spinner="Gerando resumo executivo...")
def gerar_resumo_executivo(texto_completo, nome_arquivo) -> str:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    # ... (resto do código da função)
    return ""

# ... e assim por diante para todas as outras funções.
# Apenas remova o parâmetro 'api_key' das assinaturas das funções.
