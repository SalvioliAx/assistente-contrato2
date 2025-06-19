# app.py
"""
Ponto de entrada principal da aplicação Streamlit "Analisador-IA ProMax".
Este arquivo é responsável por:
- Configurar a página.
- Inicializar serviços (Firebase, Google AI).
- Gerenciar o estado da sessão.
- Construir a interface da barra lateral (sidebar).
- Orquestrar a renderização das abas (tabs) a partir do módulo `ui_tabs`.
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

def main():
    """Função principal que executa a aplicação Streamlit."""
    st.set_page_config(layout="wide", page_title="Analisador-IA ProMax", page_icon="💡")
    hide_streamlit_style = "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>"
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.title("💡 Analisador-IA ProMax")

    db, BUCKET_NAME = initialize_services()

    embeddings = None
    if db:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        except Exception as e:
            st.sidebar.error(f"Erro ao inicializar embeddings: {e}")

    # Inicializa o estado da sessão se não existir
    if "vector_store" not in st.session_state:
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.session_state.arquivos_pdf_originais = None
        st.session_state.colecao_ativa = None
        st.session_state.nomes_arquivos = []


    with st.sidebar:
        st.image("https://i.imgur.com/aozL2jD.png", width=100)
        st.header("Gerenciar Documentos")

        if not db or not embeddings:
            st.error("Aplicação desabilitada. Verifique as conexões.")
            return

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

    if not st.session_state.get("vector_store"):
        st.info("👈 Por favor, carregue e processe documentos na barra lateral para começar.")
    else:
        tab_chat, tab_dash, tab_res, tab_risk, tab_prazo, tab_conf, tab_anom = st.tabs([
            "💬 Chat", "📈 Dashboard", "📜 Resumo", "🚩 Riscos", "🗓️ Prazos", "⚖️ Conformidade", "📊 Anomalias"
        ])
        
        # --- CORREÇÃO APLICADA AQUI ---
        # Garantindo que os argumentos corretos sejam passados para todas as abas
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

if __name__ == "__main__":
    main()
