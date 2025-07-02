# app.py
"""
Ponto de entrada principal da aplicação Streamlit "Analisador-IA ProMax".
"""
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from firebase_utils import (
    initialize_services, 
    listar_colecoes_salvas, 
    salvar_colecao_atual, 
    carregar_colecao
)
from auth_utils import register_user, login_user
from pdf_processing import obter_vector_store_de_uploads
from ui_tabs import (
    render_chat_tab, render_dashboard_tab, render_resumo_tab, 
    render_riscos_tab, render_prazos_tab, render_conformidade_tab, 
    render_anomalias_tab
)

def render_login_page(db):
    """Renderiza a página de login e cadastro."""
    st.title("Bem-vindo ao Analisador-IA ProMax")
    
    login_tab, register_tab = st.tabs(["Login", "Cadastrar"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("E-mail")
            password = st.text_input("Senha", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                user_id = login_user(email, password)
                if user_id:
                    st.session_state.logged_in = True
                    st.session_state.user_id = user_id
                    st.session_state.user_email = email
                    st.rerun()

    with register_tab:
        with st.form("register_form"):
            new_email = st.text_input("Seu E-mail")
            new_password = st.text_input("Crie uma Senha", type="password")
            confirm_password = st.text_input("Confirme a Senha", type="password")
            submitted = st.form_submit_button("Cadastrar")
            if submitted:
                if new_password == confirm_password:
                    register_user(new_email, new_password)
                else:
                    st.error("As senhas não coincidem.")

def render_main_app(db, BUCKET_NAME, embeddings):
    """Renderiza a aplicação principal após o login."""
    st.sidebar.title(f"Bem-vindo(a)!")
    st.sidebar.caption(st.session_state.user_email)
    
    with st.sidebar:
        st.header("Gerenciar Documentos")
        user_id = st.session_state.user_id

        modo = st.radio("Carregar documentos:", ("Novo Upload", "Carregar Coleção"), key="modo_carregamento")

        if modo == "Novo Upload":
            arquivos = st.file_uploader("Selecione PDFs", type="pdf", accept_multiple_files=True, key="upload_arquivos")
            if st.button("Processar Documentos", use_container_width=True, disabled=not arquivos):
                vs, nomes = obter_vector_store_de_uploads(arquivos, embeddings)
                if vs and nomes:
                    st.session_state.messages = []
                    st.session_state.vector_store = vs
                    st.session_state.nomes_arquivos = nomes
                    st.session_state.colecao_ativa = None
                    st.rerun()

        else: # Carregar Coleção
            colecoes = listar_colecoes_salvas(db, user_id)
            if colecoes:
                sel = st.selectbox("Escolha uma coleção:", colecoes, index=None, placeholder="Selecione...", key="select_colecao")
                if st.button("Carregar Coleção", use_container_width=True, disabled=not sel):
                    vs, nomes = carregar_colecao(db, embeddings, user_id, sel)
                    if vs and nomes:
                        st.session_state.messages = []
                        st.session_state.vector_store = vs
                        st.session_state.nomes_arquivos = nomes
                        st.session_state.colecao_ativa = sel
                        st.rerun()
            else:
                st.info("Nenhuma coleção salva.")

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
    """Função principal que gerencia o fluxo da aplicação."""
    st.set_page_config(layout="wide", page_title="Analisador-IA ProMax", page_icon="💡")
    
    # --- CORREÇÃO APLICADA AQUI ---
    # Este CSS oculta o rodapé e o avatar do criador da app.
    st.markdown("""
        <style>
            footer {
                visibility: hidden;
            }
            [data-testid="appCreatorAvatar"] {
                display: none;
            }
            div[data-testid="stDeployButton"] {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)
    
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
