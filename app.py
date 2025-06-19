# app.py
"""
Ponto de entrada principal da aplicação Streamlit "Analisador-IA ProMax".
Versão com refatoração visual completa usando Tailwind CSS.
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

def load_visual_assets():
    """Carrega o CSS customizado e o script do Tailwind CSS."""
    st.markdown("""
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
            #MainMenu, footer, header { visibility: hidden; }
            .stButton>button {
                transition: all 0.2s ease-in-out;
            }
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
            div[role="radiogroup"] {
                display: flex;
                flex-direction: row;
                gap: 10px;
                border-bottom: 2px solid #E5E7EB;
                padding-bottom: 10px;
            }
            div[role="radiogroup"] label > div:first-child { display: none; }
            div[role="radiogroup"] label {
                cursor: pointer;
                padding: 8px 16px;
                border-radius: 8px;
                transition: all 0.2s;
            }
            div[role="radiogroup"] label:hover { background-color: #F3F4F6; }
        </style>
    """, unsafe_allow_html=True)

def render_login_page(db):
    """Renderiza a página de login com um layout aprimorado usando Tailwind CSS."""
    st.markdown("""
        <div class="fixed top-0 left-0 w-full h-full bg-gradient-to-br from-gray-50 to-gray-200 flex items-center justify-center p-4">
            <div class="bg-white w-full max-w-md p-8 rounded-2xl shadow-lg">
                <div class="flex flex-col items-center">
                    <img src="https://i.imgur.com/aozL2jD.png" alt="Logo" class="w-20 h-20 mb-4"/>
                    <h1 class="text-3xl font-bold text-gray-800">Bem-vindo(a)</h1>
                    <p class="text-gray-500 mt-2">Faça login ou registe-se para analisar os seus contratos</p>
                </div>
    """, unsafe_allow_html=True)
    
    login_tab, register_tab = st.tabs(["Login", "Registar"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("E-mail", placeholder="seu.email@exemplo.com")
            password = st.text_input("Senha", type="password", placeholder="********")
            submitted = st.form_submit_button("Entrar")
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
    
    st.markdown("</div></div>", unsafe_allow_html=True)

def render_main_app(db, BUCKET_NAME, embeddings):
    """Renderiza a aplicação principal após o login com o novo design."""
    st.markdown('<div class="bg-gray-100 min-h-screen">', unsafe_allow_html=True)
    
    with st.sidebar:
        st.title(f"Bem-vindo(a)!")
        st.caption(st.session_state.user_email)
        st.markdown("---")
        
        st.header("Gerir Documentos")
        user_id = st.session_state.user_id

        modo = st.radio("Carregar documentos:", ("Novo Upload", "Carregar Coleção"), key="modo_carregamento", horizontal=True)

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
            sel = st.selectbox("Escolha uma coleção:", colecoes, index=None, placeholder="Selecione...", key="select_colecao")
            if st.button("Carregar Coleção", use_container_width=True, disabled=not sel):
                vs, nomes = carregar_colecao(db, embeddings, user_id, sel)
                if vs and nomes:
                    st.session_state.messages = []
                    st.session_state.vector_store = vs
                    st.session_state.nomes_arquivos = nomes
                    st.session_state.colecao_ativa = sel
                    st.rerun()

        if st.session_state.get("vector_store") and modo == "Novo Upload":
            st.markdown("---")
            st.subheader("Salvar Coleção Atual")
            nome_colecao = st.text_input("Nome para a nova coleção:", key="nome_nova_colecao")
            if st.button("Salvar", use_container_width=True, disabled=not nome_colecao):
                salvar_colecao_atual(db, user_id, nome_colecao, st.session_state.vector_store, st.session_state.nomes_arquivos)
        
        st.sidebar.markdown("<div class='mt-auto'></div><hr>", unsafe_allow_html=True)
        if st.sidebar.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.markdown("""
        <div class="p-4 sm:p-6 lg:p-8">
            <div class="flex justify-between items-center">
                <div class="flex items-center gap-4">
                    <img src="https://i.imgur.com/aozL2jD.png" alt="Logo" class="w-12 h-12"/>
                    <h1 class="text-3xl font-bold text-gray-800">Analisador-IA ProMax</h1>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="px-4 sm:px-6 lg:px-8">', unsafe_allow_html=True)

    if not st.session_state.get("vector_store"):
        st.info("👈 Por favor, carregue documentos ou uma coleção para começar.")
    else:
        tab_options = ["💬 Chat", "📈 Dashboard", "📜 Resumo", "🚩 Riscos", "🗓️ Prazos", "⚖️ Conformidade", "📊 Anomalias"]
        
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = tab_options[0]

        selected_tab = st.radio("Navegação", tab_options, key="tab_navigation", horizontal=True, label_visibility="collapsed")
        st.session_state.active_tab = selected_tab
        
        st.markdown('<div class="bg-white p-6 rounded-2xl shadow-sm mt-4">', unsafe_allow_html=True)

        vector_store = st.session_state.vector_store
        nomes_arquivos = st.session_state.nomes_arquivos
        
        # Dicionário de funções para renderizar cada aba
        tab_functions = {
            "💬 Chat": render_chat_tab, "📈 Dashboard": render_dashboard_tab,
            "📜 Resumo": render_resumo_tab, "🚩 Riscos": render_riscos_tab,
            "🗓️ Prazos": render_prazos_tab, "⚖️ Conformidade": render_conformidade_tab,
            "📊 Anomalias": render_anomalias_tab
        }

        # --- CORREÇÃO APLICADA AQUI ---
        # Chamamos a função da aba ativa, passando os argumentos corretos para cada uma.
        active_function = tab_functions[st.session_state.active_tab]
        if st.session_state.active_tab == "📊 Anomalias":
            active_function() # Esta função não precisa de argumentos
        else:
            active_function(vector_store, nomes_arquivos)

        st.markdown('</div>')

    st.markdown('</div>')
    st.markdown('</div>')
    
def main():
    """Função principal que gere o fluxo da aplicação."""
    st.set_page_config(layout="wide", page_title="Analisador-IA ProMax", page_icon="💡")
    load_visual_assets()
    
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
