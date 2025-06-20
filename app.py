# app.py
"""
Ponto de entrada principal da aplicação Streamlit "Analisador-IA ProMax".
"""
import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.cloud import secretmanager

from firebase_utils import initialize_services, listar_colecoes_salvas, salvar_colecao_atual, carregar_colecao
from auth_utils import register_user, login_user
from pdf_processing import obter_vector_store_de_uploads
from ui_tabs import (
    render_chat_tab, render_dashboard_tab, render_resumo_tab, 
    render_riscos_tab, render_prazos_tab, render_conformidade_tab, 
    render_anomalias_tab
)

@st.cache_resource
def setup_api_key():
    """Obtém a chave de API da Google do Secret Manager e define-a como uma variável de ambiente."""
    try:
        project_id = "contratiapy"
        secret_id = "google-api-key"
        version_id = "latest"
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        
        client = secretmanager.SecretManagerServiceClient()
        response = client.access_secret_version(name=name)
        api_key = response.payload.data.decode("UTF-8")
        
        # Define a variável de ambiente que todas as bibliotecas irão usar
        os.environ["GOOGLE_API_KEY"] = api_key
        return api_key
    except Exception as e:
        st.error(f"Não foi possível obter a Chave de API do Secret Manager: {e}")
        return None

def render_login_page(db):
    st.title("Bem-vindo ao Analisador-IA ProMax")
    # ... (código inalterado)

def render_main_app(db, BUCKET_NAME, embeddings):
    st.sidebar.title(f"Bem-vindo(a)!")
    st.sidebar.caption(st.session_state.user_email)
    
    with st.sidebar:
        st.header("Gerenciar Documentos")
        user_id = st.session_state.user_id
        modo = st.radio("Carregar documentos:", ("Novo Upload", "Carregar Coleção"), key="modo_carregamento")

        if modo == "Novo Upload":
            arquivos = st.file_uploader("Selecione PDFs", type="pdf", accept_multiple_files=True, key="upload_arquivos")
            if st.button("Processar Documentos", use_container_width=True, disabled=not arquivos):
                # Já não precisamos de passar a chave de API
                vs, nomes = obter_vector_store_de_uploads(arquivos, embeddings)
                if vs and nomes:
                    st.session_state.messages = []
                    st.session_state.vector_store = vs
                    st.session_state.nomes_arquivos = nomes
                    st.session_state.colecao_ativa = None
                    st.rerun()
        else: # Carregar Coleção
            # ... (código inalterado)
            pass

        if st.session_state.get("vector_store") and modo == "Novo Upload":
            st.markdown("---")
            st.subheader("Salvar Coleção Atual")
            nome_colecao = st.text_input("Nome para a nova coleção:", key="nome_nova_colecao")
            if st.button("Salvar", use_container_width=True, disabled=not nome_colecao):
                salvar_colecao_atual(db, user_id, nome_colecao, st.session_state.vector_store, st.session_state.nomes_arquivos)
        
        st.sidebar.markdown("<hr>", unsafe_allow_html=True)
        if st.sidebar.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.title("💡 Analisador-IA ProMax")
    if not st.session_state.get("vector_store"):
        st.info("👈 Por favor, carregue documentos ou uma coleção para começar.")
    else:
        tabs = st.tabs(["💬 Chat", "📈 Dashboard", "📜 Resumo", "🚩 Riscos", "🗓️ Prazos", "⚖️ Conformidade", "📊 Anomalias"])
        vector_store = st.session_state.vector_store
        nomes_arquivos = st.session_state.nomes_arquivos
        
        with tabs[0]: render_chat_tab(vector_store, nomes_arquivos)
        with tabs[1]: render_dashboard_tab(vector_store, nomes_arquivos)
        with tabs[2]: render_resumo_tab(vector_store, nomes_arquivos)
        with tabs[3]: render_riscos_tab(vector_store, nomes_arquivos)
        with tabs[4]: render_prazos_tab(vector_store, nomes_arquivos)
        with tabs[5]: render_conformidade_tab(vector_store, nomes_arquivos)
        with tabs[6]: render_anomalias_tab()

def main():
    st.set_page_config(layout="wide", page_title="Analisador-IA ProMax", page_icon="💡")
    
    # Carrega a chave de API e define a variável de ambiente
    api_key = setup_api_key()
    if not api_key:
        st.error("A aplicação não pode iniciar sem uma Chave de API da Google válida.")
        return

    db, BUCKET_NAME = initialize_services()
    if not db:
        st.error("Falha na conexão com o banco de dados.")
        return
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        render_login_page(db)
    else:
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = None
        render_main_app(db, BUCKET_NAME, embeddings)

if __name__ == "__main__":
    main()
