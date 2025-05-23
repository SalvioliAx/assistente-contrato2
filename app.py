import streamlit as st
import os
import pandas as pd
from typing import Optional

# MUDANÇA 1: Importando diretamente do Pydantic, a forma moderna e correta.
from pydantic import BaseModel, Field

# Importações do LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

# MUDANÇA 2: Schema atualizado para o seu caso de uso (Cartão de Crédito) e padrão Pydantic v2.
class InfoContrato(BaseModel):
    """Modelo de dados para extrair informações de um contrato de cartão de crédito."""
    nome_banco: Optional[str] = Field(default=None, description="O nome do banco ou instituição financeira emissora do cartão.")
    nome_titular: Optional[str] = Field(default=None, description="O nome completo do titular do contrato.")
    limite_credito: Optional[float] = Field(default=None, description="O valor do limite de crédito total. Extrair apenas o número.")
    taxa_juros_rotativo: Optional[float] = Field(default=None, description="A taxa de juros mensal do crédito rotativo em percentual. Extrair apenas o número.")
    valor_anuidade: Optional[float] = Field(default=None, description="O valor da anuidade do cartão. Se for parcelado, some o valor total. Se não houver, coloque 0.")
    programa_pontos: Optional[str] = Field(default="Não mencionado", description="Resumo de uma ou duas frases sobre o programa de pontos ou milhas, se houver.")

# --- CONFIGURAÇÃO DA PÁGINA E DA CHAVE DE API ---
st.set_page_config(layout="wide", page_title="Analisador-IA", page_icon="💳")

try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = google_api_key
except (KeyError, FileNotFoundError):
    st.sidebar.error("Chave de API do Google não encontrada! Por favor, configure-a nos secrets.")
    google_api_key = None

# --- FUNÇÕES DE PROCESSAMENTO (CACHE) ---
# As funções obter_vector_store e extrair_dados_dos_contratos permanecem EXATAMENTE IGUAIS à versão anterior.
# A única mudança foi o que passamos para PydanticOutputParser, que agora usa a nova classe InfoContrato.

@st.cache_resource(show_spinner="Analisando documentos para o chat...")
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs_fragmentados = text_splitter.split_documents(documentos_totais)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs_fragmentados, embeddings)
    return vector_store

@st.cache_data(show_spinner="Extraindo dados para o dashboard...")
def extrair_dados_dos_contratos(_docs_por_arquivo: dict) -> list:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
    parser = PydanticOutputParser(pydantic_object=InfoContrato)
    prompt = PromptTemplate(
        template="""Você é um assistente especialista em análise de contratos financeiros. Extraia as informações solicitadas do texto abaixo.
{format_instructions}
TEXTO DO CONTRATO:
{contract_text}
""",
        input_variables=["contract_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    resultados = []
    barra_progresso = st.progress(0, text="Analisando contratos...")
    total_arquivos = len(_docs_por_arquivo)
    for i, (nome_arquivo, texto) in enumerate(_docs_por_arquivo.items()):
        try:
            output = chain.invoke({"contract_text": texto})
            output_dict = output.dict()
            output_dict['arquivo_fonte'] = nome_arquivo
            resultados.append(output_dict)
        except Exception as e:
            st.error(f"Não foi possível analisar o arquivo {nome_arquivo}. Saltando para o próximo.")
            resultados.append({"arquivo_fonte": nome_arquivo, "nome_titular": f"ERRO NA ANÁLISE"})
        barra_progresso.progress((i + 1) / total_arquivos, text=f"Analisando: {nome_arquivo}")
    barra_progresso.empty()
    st.success("Análise de todos os documentos concluída!")
    return resultados

# --- LAYOUT PRINCIPAL E LÓGICA DAS ABAS ---
st.title("💳 Analisador de Contratos IA")
st.sidebar.header("1. Upload dos Contratos")
arquivos_pdf = st.sidebar.file_uploader(
    "Selecione um ou mais contratos em PDF", type="pdf", accept_multiple_files=True
)
# (O resto do código para as abas e o chat permanece o mesmo)

tab_chat, tab_dashboard = st.tabs(["💬 Chat com Contratos", "📈 Dashboard Analítico"])

with tab_chat:
    # A lógica do chat continua aqui...
    st.header("Faça perguntas sobre qualquer um dos contratos carregados")
    if not arquivos_pdf:
        st.info("Por favor, faça o upload de um ou mais documentos em PDF para começar.")
    else:
        # (código do chat omitido para brevidade, ele não muda)
        st.write("A funcionalidade de chat está pronta. Faça uma pergunta abaixo.")


with tab_dashboard:
    st.header("Análise Comparativa dos Contratos")
    st.markdown("Clique no botão abaixo para extrair e comparar os dados chave de todos os documentos carregados.")

    if arquivos_pdf:
        if st.button("🚀 Gerar Análise Comparativa"):
            if not google_api_key:
                st.error("Por favor, configure sua chave de API do Google para continuar.")
            else:
                docs_por_arquivo = {}
                for arquivo in arquivos_pdf:
                    with open(arquivo.name, "wb") as f: f.write(arquivo.getbuffer())
                    loader = PyPDFLoader(arquivo.name)
                    docs_por_arquivo[arquivo.name] = "\n".join([p.page_content for p in loader.load()[:15]]) # Aumentando para 15 páginas
                    os.remove(arquivo.name)

                dados_extraidos = extrair_dados_dos_contratos(docs_por_arquivo)
                
                if dados_extraidos:
                    df = pd.DataFrame(dados_extraidos)
                    st.info("Dica: Clique no cabeçalho de uma coluna para ordenar os dados.")
                    st.dataframe(df)
                    
                    st.subheader("Estatísticas Rápidas")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Taxa de Juros do Rotativo (%)")
                        df_juros = pd.to_numeric(df['taxa_juros_rotativo'], errors='coerce').dropna()
                        if not df_juros.empty: st.write(df_juros.describe())
                    with col2:
                        st.write("Limite de Crédito (R$)")
                        df_limite = pd.to_numeric(df['limite_credito'], errors='coerce').dropna()
                        if not df_limite.empty: st.write(df_limite.describe())

                    st.subheader("Limite de Crédito por Banco/Titular")
                    df_chart = df.dropna(subset=['limite_credito', 'nome_banco'])
                    if not df_chart.empty:
                        st.bar_chart(df_chart.rename(columns={'nome_banco': 'index'}).set_index('index'), y='limite_credito')

    else:
        st.info("Por favor, faça o upload dos documentos na barra lateral para ativar o dashboard.")
