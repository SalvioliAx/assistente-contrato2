import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- CONFIGURAÇÃO DA PÁGINA E DA CHAVE DE API ---

# Define o layout da página para ser mais largo, o título da página e o ícone.
st.set_page_config(layout="wide", page_title="Contrat-IA", page_icon="📄")

# Tenta carregar a chave de API dos secrets do Streamlit (para quando for implantado)
# Se não encontrar, pede para o usuário inserir no sidebar.
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    google_api_key = st.sidebar.text_input("Cole sua Chave de API do Google aqui:", type="password")

# Esconde o menu de "hamburger" e o rodapé do Streamlit
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# --- FUNÇÕES DO MOTOR DE IA (LÓGICA DO BACKEND) ---

@st.cache_resource(show_spinner="Processando documento... Por favor, aguarde.")
def processar_documento(arquivo_pdf_bytes, google_api_key):
    """
    Função para carregar, fragmentar e vetorizar o documento PDF.
    Usa o cache do Streamlit para não reprocessar o mesmo arquivo várias vezes.
    """
    if not google_api_key:
        st.error("Chave de API do Google não fornecida!")
        return None

    # Salva os bytes do arquivo em um arquivo temporário
    with open("temp.pdf", "wb") as f:
        f.write(arquivo_pdf_bytes)

    # Carrega o PDF
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load_and_split()

    # Fragmenta os documentos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    docs = text_splitter.split_documents(pages)

    # Cria embeddings e o banco de dados vetorial
    os.environ["GOOGLE_API_KEY"] = google_api_key
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs, embeddings)
    
    # Monta a cadeia de QA
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )
    return qa_chain


# --- LAYOUT DA INTERFACE (O QUE O USUÁRIO VÊ) ---

# Título principal da aplicação
st.title("📄 Contrat-IA: Converse com seus Contratos")
st.markdown("Faça o upload de um contrato em PDF e faça perguntas sobre ele em linguagem natural.")

# Sidebar para o upload de arquivos
st.sidebar.header("1. Upload do Contrato")
arquivo_pdf = st.sidebar.file_uploader("Selecione o arquivo PDF", type="pdf")

if arquivo_pdf:
    # Se um arquivo foi carregado, processa e armazena na "memória" da sessão
    if "qa_chain" not in st.session_state or st.session_state.get("nome_arquivo") != arquivo_pdf.name:
        st.session_state.nome_arquivo = arquivo_pdf.name
        arquivo_pdf_bytes = arquivo_pdf.getvalue()
        st.session_state.qa_chain = processar_documento(arquivo_pdf_bytes, google_api_key)

    # Inicializa o histórico de chat se ainda não existir
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": f"Olá! Estou pronto para responder perguntas sobre o documento **{arquivo_pdf.name}**."}]

    # Exibe o histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("Ver Fontes Utilizadas"):
                    for i, doc in enumerate(message["sources"]):
                        st.markdown(f"**Fonte {i+1} (Página {doc.metadata.get('page', 'N/A')})**")
                        st.info(f"{doc.page_content[:250]}...")

    # Input para a pergunta do usuário
    if prompt := st.chat_input("Qual sua pergunta sobre o contrato?"):
        # Adiciona a pergunta do usuário ao histórico e exibe
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Gera e exibe a resposta da IA
        with st.chat_message("assistant"):
            with st.spinner("Analisando o documento..."):
                if st.session_state.qa_chain:
                    resultado = st.session_state.qa_chain({"query": prompt})
                    resposta = resultado["result"]
                    fontes = resultado["source_documents"]
                    
                    st.markdown(resposta)

                    # Expander com as fontes
                    with st.expander("Ver Fontes Utilizadas"):
                        for i, doc in enumerate(fontes):
                             st.markdown(f"**Fonte {i+1} (Página {doc.metadata.get('page', 'N/A')})**")
                             st.info(f"{doc.page_content[:250]}...")
                    
                    # Adiciona a resposta completa (com fontes) ao histórico
                    st.session_state.messages.append({"role": "assistant", "content": resposta, "sources": fontes})
                else:
                    st.error("Ocorreu um erro ao processar o documento. Verifique a chave de API.")

else:
    st.info("Por favor, faça o upload de um documento em PDF para começar.")
