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

def load_css():
    """Carrega o CSS customizado para a refatoração visual."""
    css = """
    <style>
        /* Importa uma fonte mais profissional do Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        /* Aplica a fonte a toda a aplicação */
        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif;
        }

        /* Cor de fundo principal */
        .stApp {
            background-color: #F0F2F6;
        }

        /* Estilo para os botões */
        .stButton>button {
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            background-color: #1C64F2;
            color: white;
            font-weight: 600;
            transition: all 0.2s ease-in-out;
        }

        .stButton>button:hover {
            background-color: #0F4AB2;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .stButton>button:focus {
            outline: none;
            box-shadow: 0 0 0 3px rgba(28, 100, 242, 0.4);
        }
        
        /* Estilo para a sidebar */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #E0E0E0;
        }

        /* Estilo para as abas (tabs) */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 48px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 8px;
            padding: 10px 16px;
        }

        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF;
            font-weight: 600;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        /* Estilo para o formulário de login */
        .login-container {
            background-color: #FFFFFF;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            margin-top: -50px; /* Eleva um pouco o container */
        }

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def render_login_page(db):
    """Renderiza a página de login e registo com um layout aprimorado."""
    
    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.image("https://i.imgur.com/aozL2jD.png", width=100)
        st.title("Bem-vindo(a)")
        st.caption("Faça login ou registe-se para continuar")
        
        login_tab, register_tab = st.tabs(["Login", "Registar"])

        with login_tab:
            with st.form("login_form"):
                email = st.text_input("E-mail", placeholder="seu.email@exemplo.com")
                password = st.text_input("Senha", type="password", placeholder="********")
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
                new_email = st.text_input("O seu E-mail", placeholder="seu.email@exemplo.com")
                new_password = st.text_input("Crie uma Senha", type="password", placeholder="Pelo menos 6 caracteres")
                confirm_password = st.text_input("Confirme a Senha", type="password", placeholder="Repita a senha")
                submitted = st.form_submit_button("Registar")
                if submitted:
                    if new_password == confirm_password:
                        register_user(new_email, new_password)
                    else:
                        st.error("As senhas não coincidem.")
        st.markdown('</div>', unsafe_allow_html=True)


def render_main_app(db, BUCKET_NAME, embeddings):
    """Renderiza a aplicação principal após o login."""
    with st.sidebar:
        st.title(f"Bem-vindo(a)!")
        st.caption(st.session_state.user_email)
        st.markdown("---")
        
        st.header("Gerir Documentos")
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
                    st.session_state.arquivos_pdf_originais = arquivos
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

        if st.session_state.get("vector_store"):
            if modo == "Novo Upload":
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
    """Função principal que gere o fluxo da aplicação."""
    st.set_page_config(layout="wide", page_title="Analisador-IA ProMax", page_icon="💡")
    load_css()
    
    db, BUCKET_NAME = initialize_services()
    if not db:
        st.error("Falha na ligação à base de dados.")
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
