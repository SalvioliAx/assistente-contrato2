# app.py
"""
Ponto de entrada principal da aplicação Streamlit "Analisador-IA ProMax".
"""
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Importando as funções dos módulos refatorados
from firebase_utils import (
    initialize_services, 
    listar_colecoes_salvas, 
    salvar_colecao_atual, 
    carregar_colecao
)
from auth_utils import register_user, login_user # <-- Nova importação
from pdf_processing import obter_vector_store_de_uploads
from ui_tabs import (
    render_chat_tab, 
    render_dashboard_tab, 
    render_resumo_tab, 
    render_riscos_tab, 
    render_prazos_tab, 
    render_conformidade_tab, 
    render_anomalias_tab
)

def render_login_page(db):
    """Renderiza a página de login e cadastro."""
    st.title("Bem-vindo ao Analisador-IA ProMax")
    
    login_tab, register_tab = st.tabs(["Login", "Cadastrar"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Usuário")
            password = st.text_input("Senha", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if login_user(db, username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()

    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("Novo Usuário")
            new_password = st.text_input("Nova Senha", type="password")
            confirm_password = st.text_input("Confirme a Senha", type="password")
            submitted = st.form_submit_button("Cadastrar")
            if submitted:
                if new_password == confirm_password:
                    register_user(db, new_username, new_password)
                else:
                    st.error("As senhas não coincidem.")

def render_main_app(db, BUCKET_NAME, embeddings):
    """Renderiza a aplicação principal após o login."""
    st.sidebar.title(f"Bem-vindo, {st.session_state.username}")
    
    with st.sidebar:
        st.image("https://i.imgur.com/aozL2jD.png", width=100)
        st.header("Gerenciar Documentos")

        modo = st.radio("Carregar documentos:", ("Novo Upload", "Carregar Coleção"), key="modo_carregamento")

        if modo == "Novo Upload":
            arquivos = st.file_uploader("Selecione PDFs", type="pdf", accept_multiple_files=True, key="upload_arquivos")
            if st.button("Processar Documentos", use_container_width=True, disabled=not arquivos):
                vs, nomes = obter_vector_store_de_uploads(arquivos, embeddings)
                if vs and nomes:
                    st.session_state.messages = []
                    st.session_state.vector_store = vs
                    st.session_state.nomes_arquivos = nomes
                    st.session_state.arquivos_pdf_originais = arquivos
                    st.session_state.colecao_ativa = None
                    st.success(f"{len(nomes)} documento(s) processado(s)!")
                    st.rerun()

        else: # Carregar Coleção
            colecoes = listar_colecoes_salvas(db)
            if colecoes:
                sel = st.selectbox("Escolha uma coleção:", colecoes, index=None, placeholder="Selecione...", key="select_colecao")
                if st.button("Carregar Coleção", use_container_width=True, disabled=not sel):
                    vs, nomes = carregar_colecao(db, embeddings, sel)
                    if vs and nomes:
                        st.session_state.messages = []
                        st.session_state.vector_store = vs
                        st.session_state.nomes_arquivos = nomes
                        st.session_state.colecao_ativa = sel
                        st.session_state.arquivos_pdf_originais = None
                        st.rerun()
            else:
                st.info("Nenhuma coleção salva no Firebase.")

        if st.session_state.get("vector_store") and st.session_state.get("arquivos_pdf_originais"):
            st.markdown("---")
            st.subheader("Salvar Coleção")
            nome_colecao = st.text_input("Nome para a nova coleção:", key="nome_nova_colecao")
            if st.button("Salvar", use_container_width=True, disabled=not nome_colecao):
                salvar_colecao_atual(db, BUCKET_NAME, nome_colecao, st.session_state.vector_store, st.session_state.nomes_arquivos)
        
        st.markdown("---")
        if st.session_state.get("colecao_ativa"):
            st.markdown(f"**🔥 Coleção Ativa:** `{st.session_state.colecao_ativa}`")
        elif st.session_state.get("nomes_arquivos"):
            st.markdown(f"**📄 Arquivos em Memória:** {len(st.session_state.nomes_arquivos)}")

        st.markdown("---")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()

    # Conteúdo principal da aplicação
    st.title("💡 Analisador-IA ProMax")
    if not st.session_state.get("vector_store"):
        st.info("👈 Por favor, carregue e processe documentos na barra lateral para começar.")
    else:
        tab_chat, tab_dash, tab_res, tab_risk, tab_prazo, tab_conf, tab_anom = st.tabs([
            "💬 Chat", "📈 Dashboard", "📜 Resumo", "🚩 Riscos", "🗓️ Prazos", "⚖️ Conformidade", "📊 Anomalias"
        ])
        
        vector_store = st.session_state.vector_store
        nomes_arquivos = st.session_state.nomes_arquivos
        
        with tab_chat:
            render_chat_tab(vector_store, nomes_arquivos)
        with tab_dash:
            render_dashboard_tab(vector_store, nomes_arquivos)
        with tab_res:
            render_resumo_tab(vector_store, nomes_arquivos)
        with tab_risk:
            render_riscos_tab(vector_store, nomes_arquivos)
        with tab_prazo:
            render_prazos_tab(vector_store, nomes_arquivos)
        with tab_conf:
            render_conformidade_tab(vector_store, nomes_arquivos)
        with tab_anom:
            render_anomalias_tab()

def main():
    """Função principal que executa a aplicação Streamlit."""
    st.set_page_config(layout="wide", page_title="Analisador-IA ProMax", page_icon="💡")
    hide_streamlit_style = "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>"
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    db, BUCKET_NAME = initialize_services()
    if not db:
        st.error("Não foi possível conectar ao Firebase. A aplicação não pode continuar.")
        return

    embeddings = None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e:
        st.sidebar.error(f"Erro ao inicializar embeddings: {e}")
        return

    # Gerenciamento do estado de login
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        render_login_page(db)
    else:
        # Inicializa o estado da sessão da aplicação principal apenas se não existir
        if "vector_store" not in st.session_state:
            st.session_state.messages = []
            st.session_state.vector_store = None
            st.session_state.arquivos_pdf_originais = None
            st.session_state.colecao_ativa = None
            st.session_state.nomes_arquivos = []
        
        render_main_app(db, BUCKET_NAME, embeddings)

if __name__ == "__main__":
    main()
