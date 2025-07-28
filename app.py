# app.py
"""
Ponto de entrada principal da aplicação Streamlit "Analisador-IA ProMax".
"""
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Importa o dicionário de traduções
from translations import TRANSLATIONS
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

def render_login_page(db, t):
    """Renderiza a página de login e cadastro."""
    st.title(t["welcome_title"])
    
    login_tab, register_tab = st.tabs([t["login_tab"], t["register_tab"]])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input(t["email_label"])
            password = st.text_input(t["password_label"], type="password")
            submitted = st.form_submit_button(t["login_button"])
            if submitted:
                # Passa o dicionário 't' para as funções de autenticação
                user_id = login_user(email, password, t)
                if user_id:
                    st.session_state.logged_in = True
                    st.session_state.user_id = user_id
                    st.session_state.user_email = email
                    st.rerun()

    with register_tab:
        with st.form("register_form"):
            new_email = st.text_input(t["new_email_label"])
            new_password = st.text_input(t["new_password_label"], type="password")
            confirm_password = st.text_input(t["confirm_password_label"], type="password")
            submitted = st.form_submit_button(t["register_button"])
            if submitted:
                if new_password == confirm_password:
                    # Passa o dicionário 't' para as funções de autenticação
                    register_user(new_email, new_password, t)
                else:
                    st.error(t["password_mismatch_error"])

def render_main_app(db, BUCKET_NAME, embeddings, t):
    """Renderiza a aplicação principal após o login."""
    st.sidebar.title(t["sidebar_welcome"])
    st.sidebar.caption(st.session_state.user_email)
    
    with st.sidebar:
        st.header(t["manage_documents_header"])
        user_id = st.session_state.user_id

        modo = st.radio(
            t["load_documents_radio"], 
            (t["new_upload_option"], t["load_collection_option"]), 
            key="modo_carregamento"
        )

        if modo == t["new_upload_option"]:
            arquivos = st.file_uploader(t["select_pdfs_uploader"], type="pdf", accept_multiple_files=True, key="upload_arquivos")
            if st.button(t["process_documents_button"], use_container_width=True, disabled=not arquivos):
                vs, nomes = obter_vector_store_de_uploads(arquivos, embeddings, t)
                if vs and nomes:
                    st.session_state.messages = []
                    st.session_state.vector_store = vs
                    st.session_state.nomes_arquivos = nomes
                    st.session_state.colecao_ativa = None
                    st.rerun()

        else: # Carregar Coleção
            colecoes = listar_colecoes_salvas(db, user_id, t)
            if colecoes:
                sel = st.selectbox(
                    t["choose_collection_selectbox"], 
                    colecoes, 
                    index=None, 
                    placeholder=t["select_placeholder"], 
                    key="select_colecao"
                )
                if st.button(t["load_collection_button"], use_container_width=True, disabled=not sel):
                    vs, nomes = carregar_colecao(db, embeddings, user_id, sel, t)
                    if vs and nomes:
                        st.session_state.messages = []
                        st.session_state.vector_store = vs
                        st.session_state.nomes_arquivos = nomes
                        st.session_state.colecao_ativa = sel
                        st.rerun()
            else:
                st.info(t["no_saved_collections_info"])

        if st.session_state.get("vector_store") and modo == t["new_upload_option"]:
            st.markdown("---")
            st.subheader(t["save_current_collection_header"])
            nome_colecao = st.text_input(t["new_collection_name_input"], key="nome_nova_colecao")
            if st.button(t["save_button"], use_container_width=True, disabled=not nome_colecao):
                salvar_colecao_atual(db, user_id, nome_colecao, st.session_state.vector_store, st.session_state.nomes_arquivos, t)
        
        st.sidebar.markdown("<hr>", unsafe_allow_html=True)
        if st.sidebar.button(t["logout_button"]):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.title(t["main_app_title"])
    if not st.session_state.get("vector_store"):
        st.info(t["load_documents_prompt"])
    else:
        tab_labels = [
            t["chat_tab_label"], t["dashboard_tab_label"], t["summary_tab_label"], 
            t["risks_tab_label"], t["deadlines_tab_label"], t["compliance_tab_label"], 
            t["anomalies_tab_label"]
        ]
        tabs = st.tabs(tab_labels)
        vector_store = st.session_state.vector_store
        nomes_arquivos = st.session_state.nomes_arquivos
        
        with tabs[0]: render_chat_tab(vector_store, nomes_arquivos, t)
        with tabs[1]: render_dashboard_tab(vector_store, nomes_arquivos, t)
        with tabs[2]: render_resumo_tab(vector_store, nomes_arquivos, t)
        with tabs[3]: render_riscos_tab(vector_store, nomes_arquivos, t)
        with tabs[4]: render_prazos_tab(vector_store, nomes_arquivos, t)
        with tabs[5]: render_conformidade_tab(vector_store, nomes_arquivos, t)
        with tabs[6]: render_anomalias_tab(t)

def main():
    """Função principal que gerencia o fluxo da aplicação."""
    st.set_page_config(layout="wide", page_title="Analisador-IA ProMax", page_icon="💡")
    
    st.markdown("""
        <style>
            footer { visibility: hidden; }
            [data-testid="appCreatorAvatar"] { display: none; }
            div[data-testid="stDeployButton"] { display: none; }
        </style>
    """, unsafe_allow_html=True)

    # --- IMPLEMENTAÇÃO DO SELETOR DE IDIOMA ---
    if "lang" not in st.session_state:
        st.session_state.lang = "pt" # Define o português como padrão

    lang_options = {"Português": "pt", "English": "en", "Español": "es"}
    
    # Invertendo o dicionário para encontrar a chave (display name) pelo valor (código)
    # Isso é necessário para definir o `index` do selectbox corretamente
    lang_codes = list(lang_options.values())
    lang_names = list(lang_options.keys())
    current_lang_index = lang_codes.index(st.session_state.lang)

    selected_lang_name = st.sidebar.selectbox(
        label="Idioma / Language / Idioma",
        options=lang_names,
        index=current_lang_index
    )
    st.session_state.lang = lang_options[selected_lang_name]
    
    # 't' é o dicionário de tradução para o idioma selecionado
    t = TRANSLATIONS[st.session_state.lang]
    # --- FIM DA IMPLEMENTAÇÃO ---

    db, BUCKET_NAME = initialize_services(t)
    if not db:
        st.error(t["db_connection_error"])
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        render_login_page(db, t)
    else:
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = None
        
        render_main_app(db, BUCKET_NAME, embeddings, t)

if __name__ == "__main__":
    main()
