import streamlit as st
import os
import pandas as pd
from typing import Optional

# Importações do LangChain e Pydantic
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field # Pydantic para definir o schema
from langchain.output_parsers import PydanticOutputParser
from langchain.chains.base import Chain

# --- SCHEMA DE DADOS COM PYDANTIC (O "FORMULÁRIO" DA IA) ---

class InfoContrato(BaseModel):
    """Modelo de dados para extrair informações de um contrato de publicação."""
    nome_autor: Optional[str] = Field(description="O nome completo do autor ou da autora principal.")
    titulo_obra: Optional[str] = Field(description="O título da obra ou livro objeto do contrato.")
    percentual_royalties: Optional[float] = Field(description="O percentual de royalties sobre as vendas. Extrair apenas o número. Ex: 10.5")
    valor_adiantamento: Optional[float] = Field(description="O valor monetário do adiantamento (se houver). Extrair apenas o número.")
    data_assinatura: Optional[str] = Field(description="A data em que o contrato foi assinado, no formato DD/MM/AAAA.")
    clausula_audiolivro: Optional[str] = Field(description="Resumo de uma ou duas frases sobre os direitos para audiolivro, se mencionados. Se não houver, preencha com 'Não mencionado'.")

# --- CONFIGURAÇÃO DA PÁGINA E DA CHAVE DE API ---
st.set_page_config(layout="wide", page_title="Contrat-IA", page_icon="📊")

try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    google_api_key = st.sidebar.text_input("Cole sua Chave de API do Google aqui:", type="password")

os.environ["GOOGLE_API_KEY"] = google_api_key

# --- FUNÇÕES DE PROCESSAMENTO (CACHE) ---

@st.cache_resource(show_spinner="Analisando documentos para o chat...")
def obter_vector_store(lista_arquivos_pdf):
    # (Esta função permanece a mesma da versão anterior)
    if not lista_arquivos_pdf: return None
    documentos_totais = []
    for arquivo_pdf in lista_arquivos_pdf:
        with open(arquivo_pdf.name, "wb") as f: f.write(arquivo_pdf.getbuffer())
        loader = PyPDFLoader(arquivo_pdf.name)
        pages = loader.load()
        for page in pages: page.metadata["source"] = arquivo_pdf.name
        documentos_totais.extend(pages)
        os.remove(arquivo_pdf.name)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200) # Aumentado para dar mais contexto
    docs_fragmentados = text_splitter.split_documents(documentos_totais)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs_fragmentados, embeddings)
    return vector_store

@st.cache_data(show_spinner="Extraindo dados para o dashboard...")
def extrair_dados_dos_contratos(_docs_por_arquivo: dict, llm: ChatGoogleGenerativeAI) -> list:
    """
    Função para iterar sobre cada documento e extrair os dados estruturados.
    """
    parser = PydanticOutputParser(pydantic_object=InfoContrato)
    prompt = PromptTemplate(
        template="""
        Você é um assistente especialista em análise de contratos. Extraia as informações solicitadas do texto abaixo.
        {format_instructions}
        TEXTO DO CONTRATO:
        {contract_text}
        """,
        input_variables=["contract_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    
    resultados = []
    for nome_arquivo, texto in _docs_por_arquivo.items():
        st.info(f"Analisando: {nome_arquivo}...")
        try:
            output = chain.invoke({"contract_text": texto})
            # Adiciona o nome do arquivo ao resultado
            output_dict = output.dict()
            output_dict['arquivo_fonte'] = nome_arquivo
            resultados.append(output_dict)
        except Exception as e:
            st.error(f"Não foi possível analisar o arquivo {nome_arquivo}. Erro: {e}")
            # Adiciona uma linha de erro ao resultado para sabermos qual falhou
            resultados.append({"arquivo_fonte": nome_arquivo, "nome_autor": "ERRO NA ANÁLISE"})

    st.success("Análise de todos os documentos concluída!")
    return resultados


# --- LAYOUT PRINCIPAL E SIDEBAR ---

st.title("📊 Contrat-IA: Seu Analista Editorial Inteligente")
st.sidebar.header("1. Upload dos Contratos")
arquivos_pdf = st.sidebar.file_uploader(
    "Selecione um ou mais contratos em PDF", type="pdf", accept_multiple_files=True
)
st.sidebar.header("2. Configurações de Idioma")
idioma_selecionado = st.sidebar.selectbox(
    "Selecione o idioma para o chat:",
    ("Português", "Inglês", "Espanhol")
)

# --- ABAS DE FUNCIONALIDADES ---
tab_chat, tab_dashboard = st.tabs(["💬 Chat com Contratos", "📈 Dashboard Analítico"])

# --- LÓGICA DA ABA DE CHAT ---
with tab_chat:
    st.header("Faça perguntas sobre qualquer um dos contratos carregados")
    if arquivos_pdf:
        vector_store = obter_vector_store(arquivos_pdf)
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Olá! O que você gostaria de saber sobre os contratos?"}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # (A lógica de exibir fontes com highlight permanece aqui)

        if prompt := st.chat_input("Sua pergunta..."):
            # (Toda a lógica do chat que já tínhamos vai aqui, sem alterações)
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            # ... (resto da lógica do chat)

    else:
        st.info("Por favor, faça o upload de um ou mais documentos em PDF para começar.")


# --- LÓGICA DA ABA DE DASHBOARD ---
with tab_dashboard:
    st.header("Análise Comparativa de todos os Contratos")
    st.markdown("Clique no botão abaixo para extrair e comparar os dados chave de todos os documentos carregados.")

    if arquivos_pdf:
        if st.button("🚀 Gerar Análise Comparativa"):
            # Prepara os dados para a extração
            docs_por_arquivo = {}
            for arquivo in arquivos_pdf:
                with open(arquivo.name, "wb") as f: f.write(arquivo.getbuffer())
                loader = PyPDFLoader(arquivo.name)
                # Pega as 10 primeiras páginas para otimizar, ou o documento inteiro se for menor
                docs_por_arquivo[arquivo.name] = "\n".join([p.page_content for p in loader.load()[:10]])
                os.remove(arquivo.name)

            # Executa a extração
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0) # Usando um modelo mais potente para extração
            dados_extraidos = extrair_dados_dos_contratos(docs_por_arquivo, llm)
            
            if dados_extraidos:
                # Converte para DataFrame do Pandas e exibe
                df = pd.DataFrame(dados_extraidos)
                st.info("Dica: Clique no cabeçalho de uma coluna para ordenar os dados.")
                st.dataframe(df)

                # Mostra estatísticas básicas
                st.subheader("Estatísticas Rápidas dos Royalties")
                st.write(df['percentual_royalties'].describe())

                # Cria um gráfico simples
                st.subheader("Distribuição de Royalties por Autor")
                df_chart = df.dropna(subset=['percentual_royalties', 'nome_autor'])
                if not df_chart.empty:
                    st.bar_chart(df_chart, x='nome_autor', y='percentual_royalties')
    else:
        st.info("Por favor, faça o upload dos documentos na barra lateral para ativar o dashboard.")
