import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- CONFIGURAÇÃO DA PÁGINA E DA CHAVE DE API ---

st.set_page_config(layout="wide", page_title="Contrat-IA", page_icon="📄")

try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    google_api_key = st.sidebar.text_input("Cole sua Chave de API do Google aqui:", type="password")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# --- FUNÇÕES DO MOTOR DE IA (LÓGICA DO BACKEND) ---

@st.cache_resource(show_spinner="Analisando documento... Por favor, aguarde.")
def obter_vector_store(arquivo_pdf_bytes, google_api_key):
    """
    Função que agora SÓ processa o PDF e retorna o banco de dados vetorial.
    A cadeia de QA será criada depois, com o idioma selecionado.
    """
    if not google_api_key:
        st.error("Chave de API do Google não fornecida!")
        return None

    os.environ["GOOGLE_API_KEY"] = google_api_key
    
    with open("temp.pdf", "wb") as f:
        f.write(arquivo_pdf_bytes)

    loader = PyPDFLoader("temp.pdf")
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    docs = text_splitter.split_documents(pages)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs, embeddings)
    
    return vector_store

# --- NOSSO NOVO TEMPLATE DE PROMPT MULTILÍNGUE ---
template_prompt = """
Use os seguintes trechos de contexto para responder à pergunta no final.
Se você não sabe a resposta, apenas diga que não sabe, não tente inventar uma resposta.
Responda de forma completa e detalhada.

CONTEXTO: {context}

PERGUNTA: {question}

INSTRUÇÃO FINAL: Responda a pergunta acima estritamente no seguinte idioma: {language}.
RESPOSTA PRESTATIVA:
"""

# --- LAYOUT DA INTERFACE (O QUE O USUÁRIO VÊ) ---

st.title("📄 Contrat-IA: Converse com seus Contratos")
st.markdown("Faça o upload de um contrato em PDF e faça perguntas sobre ele em linguagem natural.")

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.header("1. Upload do Contrato")
arquivo_pdf = st.sidebar.file_uploader("Selecione o arquivo PDF", type="pdf")

st.sidebar.header("2. Configurações")
idioma_selecionado = st.sidebar.selectbox(
    "Selecione o idioma da resposta:",
    ("Português", "Inglês", "Espanhol", "Francês", "Alemão")
)

# --- LÓGICA PRINCIPAL DA APLICAÇÃO ---
if arquivo_pdf:
    if "vector_store" not in st.session_state or st.session_state.get("nome_arquivo") != arquivo_pdf.name:
        st.session_state.nome_arquivo = arquivo_pdf.name
        arquivo_pdf_bytes = arquivo_pdf.getvalue()
        st.session_state.vector_store = obter_vector_store(arquivo_pdf_bytes, google_api_key)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": f"Olá! Estou pronto para responder perguntas sobre o documento **{arquivo_pdf.name}**."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("Ver Fontes Utilizadas"):
                    for i, doc in enumerate(message["sources"]):
                        st.markdown(f"**Fonte {i+1} (Página {doc.metadata.get('page', 'N/A')})**")
                        st.info(f"{doc.page_content[:250]}...")

    if prompt := st.chat_input("Qual sua pergunta sobre o contrato?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analisando e formulando a resposta..."):
                vector_store = st.session_state.vector_store
                if vector_store:
                    # Criação da cadeia de QA "na hora", usando o novo prompt
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
                    
                    # Preenche o template do prompt com as variáveis necessárias
                    prompt_template = PromptTemplate(
                        template=template_prompt, input_variables=["context", "question", "language"]
                    )
                    
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                        return_source_documents=True,
                        # A mágica acontece aqui, ao passar o prompt personalizado
                        chain_type_kwargs={"prompt": prompt_template.partial(language=idioma_selecionado)}
                    )
                    
                    resultado = qa_chain({"query": prompt})
                    resposta = resultado["result"]
                    fontes = resultado["source_documents"]
                    
                    st.markdown(resposta)

                    with st.expander("Ver Fontes Utilizadas"):
                        for i, doc in enumerate(fontes):
                             st.markdown(f"**Fonte {i+1} (Página {doc.metadata.get('page', 'N/A')})**")
                             st.info(f"{doc.page_content[:250]}...")
                    
                    st.session_state.messages.append({"role": "assistant", "content": resposta, "sources": fontes})
                else:
                    st.error("Ocorreu um erro ao processar o documento. Verifique a chave de API.")

else:
    st.info("Por favor, faça o upload de um documento em PDF para começar.")
