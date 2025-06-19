# ui_tabs.py
"""
Este módulo contém funções para renderizar cada uma das abas (tabs)
da interface do usuário do Streamlit. Manter a UI separada da lógica
principal torna o código mais limpo.
"""
import streamlit as st
import pandas as pd
from datetime import datetime
import time

# Importando funções de lógica dos outros módulos
from llm_utils import (
    extrair_dados_dos_contratos, 
    gerar_resumo_executivo, 
    analisar_documento_para_riscos,
    extrair_eventos_dos_contratos,
    verificar_conformidade_documento,
    detectar_anomalias_no_dataframe
)
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import fitz # PyMuPDF

def render_chat_tab(vector_store, nomes_arquivos):
    """Renderiza a aba de Chat Interativo."""
    st.header("💬 Converse com seus documentos")
    # (O restante da sua lógica da `tab_chat` vai aqui, sem alterações)
    # ...
    # Por questões de brevidade, o código interno desta função foi omitido.
    # Copie e cole o conteúdo do `with tab_chat:` do seu script original aqui.

def render_dashboard_tab(vector_store, nomes_arquivos):
    """Renderiza a aba do Dashboard com dados comparativos."""
    st.header("📈 Análise Comparativa de Dados Contratuais")
    # (O restante da sua lógica da `tab_dashboard` vai aqui, sem alterações)
    # ...
    # Copie e cole o conteúdo do `with tab_dashboard:` do seu script original aqui.

# ... E assim por diante para todas as outras abas ...
# Crie uma função para cada `with tab_...:` do seu script original.

def render_resumo_tab(arquivos_originais, nomes_arquivos):
    """Renderiza a aba de Resumo Executivo."""
    st.header("📜 Resumo Executivo de um Contrato")
    # ...

def render_riscos_tab(arquivos_originais):
    """Renderiza a aba de Análise de Riscos."""
    st.header("🚩 Análise de Cláusulas de Risco")
    # ...

def render_prazos_tab(arquivos_originais):
    """Renderiza a aba de Monitoramento de Prazos."""
    st.header("🗓️ Monitoramento de Prazos e Vencimentos")
    # ...

def render_conformidade_tab(arquivos_originais):
    """Renderiza a aba de Verificação de Conformidade."""
    st.header("⚖️ Verificador de Conformidade Contratual")
    # ...

def render_anomalias_tab():
    """Renderiza a aba de Detecção de Anomalias."""
    st.header("📊 Detecção de Anomalias Contratuais")
    # ...

# NOTA: O conteúdo interno das funções acima deve ser copiado
# do seu script `app2.py` original, de dentro de cada bloco `with`.
# Por exemplo, para `render_chat_tab`:
#
# def render_chat_tab(vector_store, nomes_arquivos):
#     st.header("Converse com seus documentos")
#     if not st.session_state.messages : 
#         st.session_state.messages.append({"role": "assistant", "content": f"Olá! ..."})
#     # etc...
