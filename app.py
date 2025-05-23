import streamlit as st
import os
import pandas as pd
from typing import Optional
import re

# Importações (sem alterações)
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate

# Schema de Dados (Adicionamos um campo para o nome do arquivo para referência)
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

# --- FUNÇÕES DE PROCESSAMENTO (CACHE) ---

@st.cache_resource(show_spinner="Analisando documentos...")
def obter_vector_store(lista_arquivos_pdf):
    if not lista_arquivos_pdf: return None
    all_docs = []
    for arquivo_pdf in lista_arquivos_pdf:
        with open(arquivo_pdf.name, "wb") as f: f.write(arquivo_pdf.getbuffer())
        loader = PyPDFLoader(arquivo_pdf.name)
        pages = loader.load()
        for page in pages: page.metadata["source"] = arquivo_pdf.name
        all_docs.extend(pages)
        os.remove(arquivo_pdf.name)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs_fragmentados = text_splitter.split_documents(all_docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs_fragmentados, embeddings)
    return vector_store

# --- NOVA FUNÇÃO DE EXTRAÇÃO INTELIGENTE ---
@st.cache_data(show_spinner="Extraindo dados para o dashboard... Este processo pode ser lento.")
def extrair_dados_dos_contratos(_vector_store: FAISS, _nomes_arquivos: list) -> list:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    
    # Template de prompt para extrair UMA informação por vez
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
        
        # Filtra o vector store para buscar apenas no documento atual
        retriever_arquivo_atual = _vector_store.as_retriever(
            search_kwargs={'filter': {'source': nome_arquivo}}
        )

        # Mapeia os campos do nosso "formulário" para perguntas em linguagem natural
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
            
            # Etapa 1: Encontrar com o "imã" (RAG)
            docs_relevantes = retriever_arquivo_atual.get_relevant_documents(pergunta)
            contexto = "\n\n".join([doc.page_content for doc in docs_relevantes])
            
            # Etapa 2: Extrair com a "pinça" (LLM focado)
            if contexto:
                resultado = chain.invoke({"info_desejada": pergunta, "contexto": contexto})
                resposta = resultado['text'].strip()
                
                # Tenta limpar e converter os valores numéricos
                if campo in ["limite_credito", "taxa_juros_rotativo", "valor_anuidade"]:
                    numeros = re.findall(r"[\d\.,]+", resposta)
                    if numeros:
                        try:
                            valor_limpo = float(numeros[0].replace('.', '').replace(',', '.'))
                            dados_contrato_atual[campo] = valor_limpo
                        except ValueError:
                            dados_contrato_atual[campo] = None
                    else:
                        dados_contrato_atual[campo] = None
                else:
                    dados_contrato_atual[campo] = resposta if "não encontrado" not in resposta.lower() else None
            else:
                dados_contrato_atual[campo] = None

        resultados_finais.append(InfoContrato(**dados_contrato_atual).dict())

    barra_progresso.empty()
    st.success("Análise de todos os documentos concluída!")
    return resultados_finais

# --- LAYOUT PRINCIPAL E LÓGICA DAS ABAS ---
st.title("💳 Analisador de Contratos IA")
st.sidebar.header("1. Upload dos Contratos")
arquivos_pdf = st.sidebar.file_uploader(
    "Selecione um ou mais contratos em PDF", type="pdf", accept_multiple_files=True
)

tab_chat, tab_dashboard = st.tabs(["💬 Chat com Contratos", "📈 Dashboard Analítico"])

with tab_chat:
    st.header("Faça perguntas sobre qualquer um dos contratos carregados")
    if not arquivos_pdf:
        st.info("Por favor, faça o upload de um ou mais documentos para começar.")
    else:
        st.write("Funcionalidade de chat pronta.")
        # Lógica do chat (omitida para brevidade)

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
                    st.info("Dica: Clique no cabeçalho de uma coluna para ordenar os dados.")
                    st.dataframe(df)
                    
                    st.subheader("Estatísticas Rápidas")
                    # (Lógica de exibição das estatísticas e gráficos permanece a mesma da versão anterior)
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
