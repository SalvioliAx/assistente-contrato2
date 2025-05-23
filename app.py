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


# --- FUNÇÃO DO MOTOR (sem alterações) ---

@st.cache_resource(show_spinner="Analisando documentos... Isso pode levar um tempo.")
def obter_vector_store(lista_arquivos_pdf, google_api_key):
    if not google_api_key:
        st.error("Chave de API do Google não fornecida!")
        return None
    if not lista_arquivos_pdf:
        return None

    os.environ["GOOGLE_API_KEY"] = google_api_key
    
    documentos_totais = []
    for arquivo_pdf in lista_arquivos_pdf:
        with open(arquivo_pdf.name, "wb") as f:
            f.write(arquivo_pdf.getbuffer())
        
        loader = PyPDFLoader(arquivo_pdf.name)
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = arquivo_pdf.name
        
        documentos_totais.extend(pages)
        os.remove(arquivo_pdf.name)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    docs_fragmentados = text_splitter.split_documents(documentos_totais)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs_fragmentados, embeddings)
    
    return vector_store

# --- TEMPLATE DE PROMPT ATUALIZADO ---
template_prompt = """
Use os seguintes trechos de contexto para responder à pergunta no final.
Se você não sabe a resposta, apenas diga que não sabe, não tente inventar uma resposta.

INSTRUÇÕES DE FORMATAÇÃO DA RESPOSTA:
Sua resposta final deve ter duas partes, separadas por '|||'.
1. Parte 1: A resposta completa e detalhada para a pergunta do usuário, no idioma {language}.
2. Parte 2: A citação exata e literal da sentença do contexto que foi mais importante para formular a resposta.

EXEMPLO DE FORMATO:
[Aqui vai a resposta completa para a pergunta]|||[Aqui vai a citação exata da sentença do documento.]

CONTEXTO:
{context}

PERGUNTA:
{question}

RESPOSTA (seguindo o formato acima):
"""

# --- LAYOUT DA INTERFACE ---

st.title("📚 Contrat-IA: Sua Base de Conhecimento Editorial")
st.markdown("Faça o upload de múltiplos contratos e faça perguntas que cruzam informações entre eles.")

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.header("1. Upload dos Contratos")
arquivos_pdf = st.sidebar.file_uploader(
    "Selecione um ou mais contratos em PDF",
    type="pdf",
    accept_multiple_files=True
)

st.sidebar.header("2. Configurações")
idioma_selecionado = st.sidebar.selectbox(
    "Selecione o idioma da resposta:",
    ("Português", "Inglês", "Espanhol", "Francês")
)

# --- LÓGICA PRINCIPAL DA APLICAÇÃO ---
if arquivos_pdf:
    nomes_arquivos = sorted([f.name for f in arquivos_pdf])
    if "vector_store" not in st.session_state or st.session_state.get("nomes_arquivos") != nomes_arquivos:
        st.session_state.nomes_arquivos = nomes_arquivos
        st.session_state.vector_store = obter_vector_store(arquivos_pdf, google_api_key)
        st.session_state.messages = [{"role": "assistant", "content": f"Olá! Analisei {len(arquivos_pdf)} contratos. O que você gostaria de saber?"}]

    if "messages" not in st.session_state:
         st.session_state.messages = [{"role": "assistant", "content": f"Olá! Analisei {len(arquivos_pdf)} contratos. O que você gostaria de saber?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("Ver Fontes Utilizadas"):
                    for doc in message["sources"]:
                        # Lógica para exibir o texto com ou sem destaque
                        texto_fonte = doc.page_content
                        sentenca_chave = message.get("sentenca_chave")
                        if sentenca_chave and sentenca_chave in texto_fonte:
                            # Substitui a sentença chave por uma versão com fundo destacado
                            texto_formatado = texto_fonte.replace(
                                sentenca_chave,
                                f"<span style='background-color: #FFFACD; padding: 2px; border-radius: 3px;'>{sentenca_chave}</span>"
                            )
                        else:
                            texto_formatado = texto_fonte

                        st.markdown(f"**Fonte: `{doc.metadata.get('source', 'N/A')}` (Página {doc.metadata.get('page', 'N/A')})**")
                        # Usa unsafe_allow_html para renderizar o destaque
                        st.markdown(texto_formatado, unsafe_allow_html=True)


    if prompt := st.chat_input("Ex: 'Quais autores têm cláusula de direitos para audiolivro?'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pesquisando e formulando a resposta..."):
                vector_store = st.session_state.vector_store
                if vector_store:
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
                    prompt_template = PromptTemplate(template=template_prompt, input_variables=["context", "question", "language"])
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm, chain_type="stuff",
                        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": prompt_template.partial(language=idioma_selecionado)}
                    )
                    resultado = qa_chain({"query": prompt})
                    resposta_bruta = resultado["result"]
                    fontes = resultado["source_documents"]

                    # --- NOVA LÓGICA PARA PARSEAR A RESPOSTA ---
                    try:
                        resposta_principal, sentenca_chave = resposta_bruta.split('|||')
                        sentenca_chave = sentenca_chave.strip()
                    except ValueError:
                        # Fallback caso o modelo não siga o formato
                        resposta_principal = resposta_bruta
                        sentenca_chave = None
                    
                    st.markdown(resposta_principal)

                    # Adiciona a resposta completa, incluindo a sentença chave, ao histórico
                    st.session_state.messages.append({"role": "assistant", "content": resposta_principal, "sources": fontes, "sentenca_chave": sentenca_chave})
                    # Força o rerender da página para exibir a nova mensagem com o highlight
                    st.rerun() 
                else:
                    st.error("Ocorreu um erro. Por favor, verifique se os documentos foram carregados corretamente.")

else:
    st.info("Por favor, faça o upload de um ou mais documentos em PDF para começar.")
