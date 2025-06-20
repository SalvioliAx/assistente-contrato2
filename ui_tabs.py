# ui_tabs.py
import streamlit as st
import pandas as pd
from llm_utils import * # Importa todas as funções
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# --- AS ASSINATURAS DAS FUNÇÕES FORAM SIMPLIFICADAS ---
# Já não precisam de receber 'api_key' como parâmetro.

def render_chat_tab(vector_store, nomes_arquivos):
    st.header("💬 Converse com os seus documentos")
    # ... (código inalterado)
    llm_chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
    # ... (código inalterado)

def render_dashboard_tab(vector_store, nomes_arquivos):
    st.header("📈 Análise Comparativa de Dados Contratuais")
    if st.button("🚀 Gerar Dados para o Dashboard", key="btn_dashboard", use_container_width=True):
        dados_extraidos = extrair_dados_dos_contratos(vector_store, nomes_arquivos)
        # ... (código inalterado)

# ... e assim por diante para todas as outras funções.
# Apenas remova o parâmetro 'api_key' das assinaturas das funções
# e das chamadas às funções do 'llm_utils'.

def render_resumo_tab(vector_store, nomes_arquivos):
    # ...
    resumo = gerar_resumo_executivo(texto_completo, arquivo_selecionado)
    # ...
    
# ... etc.
