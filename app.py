import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- CONFIGURAÇÃO DA PÁGINA E DA CHAVE DE API ---

st.set_page_config(layout="wide", page_title="Contrat-IA", page_icon="📚")

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


# --- FUNÇÃO ATUALIZADA PARA MÚLTIPLOS DOCUMENTOS ---

@st.cache_resource(show_spinner="Analisando documentos... Isso pode levar um tempo.")
def obter_vector_store(lista_arquivos_pdf, google_api_key):
    """
    Função que agora processa uma LISTA de arquivos PDF e os unifica.
    """
    if not google_api_key:
        st.error("Chave de API do Google não fornecida!")
        return None
    if not lista_arquivos_pdf:
        return None

    os.environ["GOOGLE_API_KEY"] = google_api_key
    
    documentos_totais = []
    # Itera sobre cada arquivo PDF carregado
    for arquivo_pdf in lista_arquivos_pdf:
        # Salva o arquivo temporariamente para ser lido pelo PyPDFLoader
        with open(arquivo_pdf.name, "wb") as f:
            f.write(arquivo_pdf.getbuffer())
        
        loader = PyPDFLoader(arquivo_pdf.name)
        pages = loader.load()

        # Adiciona a informação de origem (nome do arquivo) a cada página
        for page in pages:
            page.metadata["source"] = arquivo_pdf.name
        
        documentos_totais.extend(pages)
        os.remove(arquivo_pdf.name) # Limpa o arquivo temporário

    # Fragmenta todos os documentos de uma vez
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    docs_fragmentados = text_splitter.split_documents(documentos_totais)

    # Cria os embeddings e o banco de dados vetorial unificado
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs_fragmentados, embeddings)
    
    return vector_store

# --- TEMPLATE DE PROMPT (sem alterações) ---
template_prompt = """
Use os seguintes trechos de contexto para responder à pergunta no final.
Se você não sabe a resposta, apenas diga que não sabe, não tente inventar uma resposta.
Responda de forma completa e detalhada, citando de qual fonte (documento) a informação foi retirada, se possível.

CONTEXTO: {context}

PERGUNTA: {question}

INSTRUÇÃO FINAL: Responda a pergunta acima estritamente no seguinte idioma: {language}.
RESPOSTA PRESTATIVA:
"""

# --- LAYOUT DA INTERFACE ---

st.title("📚 Contrat-IA: Sua Base de Conhecimento Editorial")
st.markdown("Faça o upload de múltiplos contratos e faça perguntas que cruzam informações entre eles.")

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.header("1. Upload dos Contratos")
arquivos_pdf = st.sidebar.file_uploader(
    "Selecione um ou mais contratos em PDF",
    type="pdf",
    accept_multiple_files=True  # <-- A GRANDE MUDANÇA AQUI
)

st.sidebar.header("2. Configurações")
idioma_selecionado = st.sidebar.selectbox(
    "Selecione o idioma da resposta:",
    ("Português", "Inglês", "Espanhol", "Francês")
)

# --- LÓGICA PRINCIPAL DA APLICAÇÃO ---
if arquivos_pdf:
    nomes_arquivos = sorted([f.name for f in arquivos_pdf])
    # Verifica se os arquivos mudaram para decidir se reprocessa
    if "vector_store" not in st.session_state or st.session_state.get("nomes_arquivos") != nomes_arquivos:
        st.session_state.nomes_arquivos = nomes_arquivos
        st.session_state.vector_store = obter_vector_store(arquivos_pdf, google_api_key)
        # Limpa o histórico de chat ao carregar novos documentos
        st.session_state.messages = [{"role": "assistant", "content": f"Olá! Analisei {len(arquivos_pdf)} contratos. O que você gostaria de saber?"}]

    if "messages" not in st.session_state:
         st.session_state.messages = [{"role": "assistant", "content": f"Olá! Analisei {len(arquivos_pdf)} contratos. O que você gostaria de saber?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("Ver Fontes Utilizadas"):
                    for doc in message["sources"]:
                        # Exibe o nome do arquivo de origem junto com a página
                        st.markdown(f"**Fonte: `{doc.metadata.get('source', 'N/A')}` (Página {doc.metadata.get('page', 'N/A')})**")
                        st.info(f"{doc.page_content[:250]}...")

    if prompt := st.chat_input("Ex: 'Quais autores têm cláusula de direitos para audiolivro?'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pesquisando em todos os contratos..."):
                vector_store = st.session_state.vector_store
                if vector_store:
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
                    prompt_template = PromptTemplate(template=template_prompt, input_variables=["context", "question", "language"])
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm, chain_type="stuff",
                        retriever=vector_store.as_retriever(search_kwargs={"k": 5}), # Aumentamos para 5 fontes para ter mais contexto
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": prompt_template.partial(language=idioma_selecionado)}
                    )
                    resultado = qa_chain({"query": prompt})
                    resposta = resultado["result"]
                    fontes = resultado["source_documents"]
                    st.markdown(resposta)
                    with st.expander("Ver Fontes Utilizadas"):
                        for doc in fontes:
                             st.markdown(f"**Fonte: `{doc.metadata.get('source', 'N/A')}` (Página {doc.metadata.get('page', 'N/A')})**")
                             st.info(f"{doc.page_content[:250]}...")
                    st.session_state.messages.append({"role": "assistant", "content": resposta, "sources": fontes})
                else:
                    st.error("Ocorreu um erro. Por favor, verifique se os documentos foram carregados corretamente.")

else:
    st.info("Por favor, faça o upload de um ou mais documentos em PDF para começar.")
