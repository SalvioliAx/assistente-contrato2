import streamlit as st
import os
import pandas as pd
from typing import Optional
import re

# Importações do LangChain e Pydantic
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate

# --- SCHEMA DE DADOS PARA O DASHBOARD ---
class InfoContrato(BaseModel):
    """Modelo de dados para extrair informações de um contrato de cartão de crédito."""
    arquivo_fonte: str = Field(description="O nome do arquivo de origem do contrato.")
    nome_banco: Optional[str] = Field(default=None, description="O nome do banco ou instituição financeira emissora do cartão.")
    nome_titular: Optional[str] = Field(default=None, description="O nome completo do titular do contrato.")
    limite_credito: Optional[float] = Field(default=None, description="O valor do limite de crédito total. Extrair apenas o número.")
    taxa_juros_rotativo: Optional[float] = Field(default=None, description="A taxa de juros mensal do crédito rotativo em percentual. Extrair apenas o número.")
    valor_anuidade: Optional[float] = Field(default=None, description="O valor da anuidade do cartão. Se for parcelado, some o valor total. Se não houver, coloque 0.")

# --- CONFIGURAÇÃO DA PÁGINA E DA CHAVE DE API ---
st.set_page_config(layout="wide", page_title="Analisador-IA", page_icon="💳")
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = google_api_key
except (KeyError, FileNotFoundError):
    st.sidebar.error("Chave de API do Google não encontrada! Por favor, configure-a nos secrets.")
    google_api_key = None

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- FUNÇÕES DE PROCESSAMENTO (CACHE) ---

@st.cache_resource(show_spinner="Analisando documentos...")
def obter_vector_store(lista_arquivos_pdf):
    if not lista_arquivos_pdf: return None
    documentos_totais = []
    for arquivo_pdf in lista_arquivos_pdf:
        with open(arquivo_pdf.name, "wb") as f: f.write(arquivo_pdf.getbuffer())
        loader = PyPDFLoader(arquivo_pdf.name)
        pages = loader.load()
        for page in pages: page.metadata["source"] = arquivo_pdf.name
        documentos_totais.extend(pages)
        os.remove(arquivo_pdf.name)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs_fragmentados = text_splitter.split_documents(documentos_totais)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs_fragmentados, embeddings)
    return vector_store

@st.cache_data(show_spinner="Extraindo dados para o dashboard... Este processo pode ser lento.")
def extrair_dados_dos_contratos(_vector_store: FAISS, _nomes_arquivos: list) -> list:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    prompt_template = PromptTemplate.from_template(
        "Do texto abaixo, extraia a seguinte informação: '{info_desejada}'.\n"
        "Se a informação não for encontrada, responda com 'Não encontrado'.\n"
        "Responda apenas com a informação solicitada, sem frases extras.\n\n"
        "TEXTO:\n{contexto}\n\nINFORMAÇÃO A EXTRAIR:"
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    resultados_finais = []
    barra_progresso = st.progress(0, text="Iniciando análise de contratos...")

    for i, nome_arquivo in enumerate(_nomes_arquivos):
        dados_contrato_atual = {"arquivo_fonte": nome_arquivo}
        retriever_arquivo_atual = _vector_store.as_retriever(search_kwargs={'filter': {'source': nome_arquivo}})
        mapa_campos_perguntas = {
            "nome_banco": "Qual é o nome do banco ou instituição financeira?",
            "nome_titular": "Qual é o nome do titular do contrato?",
            "limite_credito": "Qual é o valor do limite de crédito?",
            "taxa_juros_rotativo": "Qual a taxa de juros do crédito rotativo em porcentagem?",
            "valor_anuidade": "Qual o valor da anuidade?"
        }
        for campo, pergunta in mapa_campos_perguntas.items():
            barra_progresso.progress((i + (list(mapa_campos_perguntas.keys()).index(campo) / len(mapa_campos_perguntas))) / len(_nomes_arquivos), 
                                     text=f"Analisando '{campo}' em {nome_arquivo}")
            docs_relevantes = retriever_arquivo_atual.get_relevant_documents(pergunta)
            contexto = "\n\n".join([doc.page_content for doc in docs_relevantes])
            if contexto:
                resultado = chain.invoke({"info_desejada": pergunta, "contexto": contexto})
                resposta = resultado['text'].strip()
                if campo in ["limite_credito", "taxa_juros_rotativo", "valor_anuidade"]:
                    numeros = re.findall(r"[\d\.,]+", resposta)
                    if numeros:
                        try:
                            valor_limpo = float(numeros[0].replace('.', '').replace(',', '.'))
                            dados_contrato_atual[campo] = valor_limpo
                        except ValueError: dados_contrato_atual[campo] = None
                    else: dados_contrato_atual[campo] = None
                else:
                    dados_contrato_atual[campo] = resposta if "não encontrado" not in resposta.lower() else None
            else: dados_contrato_atual[campo] = None
        resultados_finais.append(InfoContrato(**dados_contrato_atual).dict())
    barra_progresso.empty()
    st.success("Análise de todos os documentos concluída!")
    return resultados_finais

# --- LAYOUT PRINCIPAL E SIDEBAR ---
st.title("💳 Analisador de Contratos IA")
st.sidebar.header("1. Upload dos Contratos")
arquivos_pdf = st.sidebar.file_uploader(
    "Selecione um ou mais contratos em PDF", type="pdf", accept_multiple_files=True
)
st.sidebar.header("2. Configurações de Idioma")
idioma_selecionado = st.sidebar.selectbox(
    "Selecione o idioma para as respostas do CHAT:",
    ("Português", "Inglês", "Espanhol")
)

# --- INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- LÓGICA DAS ABAS ---
tab_chat, tab_dashboard = st.tabs(["💬 Chat com Contratos", "📈 Dashboard Analítico"])

# --- ABA DE CHAT (COMPLETA E RESTAURADA) ---
with tab_chat:
    st.header("Converse com seus documentos")
    template_prompt_chat = PromptTemplate.from_template(
        """Use os seguintes trechos de contexto para responder à pergunta no final.
        INSTRUÇÕES DE FORMATAÇÃO DA RESPOSTA:
        Sua resposta final deve ter duas partes, separadas por '|||'.
        1. Parte 1: A resposta completa e detalhada para a pergunta do usuário, no idioma {language}.
        2. Parte 2: A citação exata e literal da sentença do contexto que foi mais importante para formular a resposta.
        CONTEXTO: {context}
        PERGUNTA: {question}
        RESPOSTA (seguindo o formato acima):"""
    )
    
    if arquivos_pdf:
        vector_store = obter_vector_store(arquivos_pdf)
        if not st.session_state.messages:
            st.session_state.messages.append({"role": "assistant", "content": "Olá! Os documentos estão prontos para consulta. Qual sua pergunta?"})

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message:
                    with st.expander("Ver Fontes Utilizadas"):
                        for doc in message["sources"]:
                            texto_fonte = doc.page_content
                            sentenca_chave = message.get("sentenca_chave")
                            if sentenca_chave and sentenca_chave in texto_fonte:
                                texto_formatado = texto_fonte.replace(sentenca_chave, f"<span style='background-color: #FFFACD; padding: 2px; border-radius: 3px;'>{sentenca_chave}</span>")
                            else: texto_formatado = texto_fonte
                            st.markdown(f"**Fonte: `{doc.metadata.get('source', 'N/A')}` (Página {doc.metadata.get('page', 'N/A')})**")
                            st.markdown(texto_formatado, unsafe_allow_html=True)

        if prompt := st.chat_input("Faça sua pergunta sobre os contratos..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Pesquisando e formulando a resposta..."):
                    llm_chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm_chat, chain_type="stuff",
                        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": template_prompt_chat.partial(language=idioma_selecionado)}
                    )
                    resultado = qa_chain({"query": prompt})
                    resposta_bruta = resultado["result"]
                    fontes = resultado["source_documents"]
                    try:
                        resposta_principal, sentenca_chave = resposta_bruta.split('|||')
                        sentenca_chave = sentenca_chave.strip()
                    except ValueError:
                        resposta_principal = resposta_bruta
                        sentenca_chave = None
                    st.markdown(resposta_principal)
                    st.session_state.messages.append({"role": "assistant", "content": resposta_principal, "sources": fontes, "sentenca_chave": sentenca_chave})
                    st.rerun()

    else:
        st.info("Por favor, faça o upload de um ou mais documentos em PDF para começar.")

# --- ABA DE DASHBOARD (COMPLETA) ---
with tab_dashboard:
    st.header("Análise Comparativa dos Contratos")
    st.markdown("Clique no botão abaixo para extrair e comparar os dados chave de todos os documentos carregados.")
    if arquivos_pdf:
        if st.button("🚀 Gerar Análise Comparativa"):
            if not google_api_key:
                st.error("Por favor, configure sua chave de API do Google para continuar.")
            else:
                vector_store = obter_vector_store(arquivos_pdf)
                nomes_arquivos = [f.name for f in arquivos_pdf]
                dados_extraidos = extrair_dados_dos_contratos(vector_store, nomes_arquivos)
                if dados_extraidos:
                    df = pd.DataFrame(dados_extraidos)
                    st.session_state.df_dashboard = df # Salva o dataframe no estado da sessão
                    st.rerun() # Recarrega para exibir os dados abaixo
        
        # Exibe o dataframe se ele existir no estado da sessão
        if 'df_dashboard' in st.session_state:
            df = st.session_state.df_dashboard
            st.info("Dica: Clique no cabeçalho de uma coluna para ordenar os dados.")
            st.dataframe(df)
            
            st.subheader("Estatísticas Rápidas")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Taxa de Juros do Rotativo (%)")
                if 'taxa_juros_rotativo' in df.columns and not df['taxa_juros_rotativo'].dropna().empty:
                    st.write(df['taxa_juros_rotativo'].dropna().describe())
                else: st.write("Nenhum dado encontrado.")
            with col2:
                st.write("Limite de Crédito (R$)")
                if 'limite_credito' in df.columns and not df['limite_credito'].dropna().empty:
                    st.write(df['limite_credito'].dropna().describe())
                else: st.write("Nenhum dado encontrado.")

    else:
        st.info("Por favor, faça o upload dos documentos na barra lateral para ativar o dashboard.")
