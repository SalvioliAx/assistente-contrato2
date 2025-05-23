import streamlit as st
import os
import pandas as pd
from typing import Optional
import re
from datetime import datetime # Import para o nome do arquivo de exportação

# Importações do LangChain e Pydantic
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate

# --- SCHEMA DE DADOS PARA O DASHBOARD (sem alterações) ---
class InfoContrato(BaseModel):
    """Modelo de dados para extrair políticas e condições de um contrato de cartão de crédito."""
    arquivo_fonte: str = Field(description="O nome do arquivo de origem do contrato.")
    nome_banco: Optional[str] = Field(default="Não encontrado", description="O nome do banco ou instituição financeira emissora do cartão.")
    condicao_limite_credito: Optional[str] = Field(default="Não encontrado", description="Resumo da política de como o limite de crédito é definido, analisado e alterado.")
    condicao_juros_rotativo: Optional[str] = Field(default="Não encontrado", description="Resumo da regra de como e quando os juros do crédito rotativo são aplicados.")
    condicao_anuidade: Optional[str] = Field(default="Não encontrado", description="Resumo da política de cobrança da anuidade, se é diferenciada ou básica e como é cobrada.")
    condicao_cancelamento: Optional[str] = Field(default="Não encontrado", description="Resumo das condições sob as quais o contrato pode ser rescindido ou cancelado pelo banco ou pelo cliente.")

# --- CONFIGURAÇÃO DA PÁGINA E DA CHAVE DE API ---
st.set_page_config(layout="wide", page_title="Analisador-IA", page_icon="⚖️")
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = google_api_key
except (KeyError, FileNotFoundError):
    st.sidebar.error("Chave de API do Google não encontrada! Por favor, configure-a nos secrets.")
    google_api_key = None

hide_streamlit_style = """
<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- FUNÇÕES DE PROCESSAMENTO (CACHE) ---

@st.cache_resource(show_spinner="Analisando documentos...")
def obter_vector_store(lista_arquivos_pdf):
    if not lista_arquivos_pdf or not google_api_key: return None
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

@st.cache_data(show_spinner="Extraindo políticas dos contratos...")
def extrair_dados_dos_contratos(_vector_store: FAISS, _nomes_arquivos: list) -> list:
    # (sem alterações nesta função)
    if not _vector_store or not google_api_key: return []
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    prompt_template = PromptTemplate.from_template(
        "Do texto abaixo, resuma em uma ou duas frases a resposta para a seguinte pergunta: '{info_desejada}'.\n"
        "Se a informação não estiver no texto, responda com 'Não encontrado'.\n"
        "Seja conciso e direto.\n\n"
        "TEXTO:\n{contexto}\n\nRESUMO DA RESPOSTA:"
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    resultados_finais = []
    barra_progresso = st.progress(0, text="Iniciando análise de contratos...")
    for i, nome_arquivo in enumerate(_nomes_arquivos):
        dados_contrato_atual = {"arquivo_fonte": nome_arquivo}
        retriever_arquivo_atual = _vector_store.as_retriever(search_kwargs={'filter': {'source': nome_arquivo}, 'k': 4})
        mapa_campos_perguntas = {
            "nome_banco": "Qual o nome do banco ou emissor principal deste contrato?",
            "condicao_limite_credito": "Qual é a política ou condição para definir o limite de crédito?",
            "condicao_juros_rotativo": "Sob quais condições os juros do crédito rotativo são aplicados?",
            "condicao_anuidade": "Qual é a política de cobrança da anuidade descrita no contrato?",
            "condicao_cancelamento": "Quais são as regras para o cancelamento ou rescisão do contrato?"
        }
        for campo, pergunta in mapa_campos_perguntas.items():
            barra_progresso.progress((i + (list(mapa_campos_perguntas.keys()).index(campo) / len(mapa_campos_perguntas))) / len(_nomes_arquivos), 
                                     text=f"Analisando '{campo}' em {nome_arquivo}")
            docs_relevantes = retriever_arquivo_atual.get_relevant_documents(pergunta)
            contexto = "\n\n".join([doc.page_content for doc in docs_relevantes])
            if contexto:
                resultado = chain.invoke({"info_desejada": pergunta, "contexto": contexto})
                resposta = resultado['text'].strip()
                dados_contrato_atual[campo] = resposta
            else: dados_contrato_atual[campo] = "Contexto não encontrado."
        resultados_finais.append(InfoContrato(**dados_contrato_atual).dict())
    barra_progresso.empty()
    st.success("Análise de todos os documentos concluída!")
    return resultados_finais

@st.cache_data(show_spinner="Gerando resumo executivo...")
def gerar_resumo_executivo(arquivo_pdf_selecionado_bytes, nome_arquivo, google_api_key_func):
    # (sem alterações nesta função)
    if not arquivo_pdf_selecionado_bytes or not google_api_key_func:
        return "Erro: Arquivo ou chave de API não fornecidos."
    with open(nome_arquivo, "wb") as f: f.write(arquivo_pdf_selecionado_bytes)
    loader = PyPDFLoader(nome_arquivo)
    documento_completo_paginas = loader.load()
    os.remove(nome_arquivo)
    texto_completo = "\n\n".join([page.page_content for page in documento_completo_paginas])
    llm_resumo = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    template_prompt_resumo = PromptTemplate.from_template(
        "Você é um assistente especializado em analisar e resumir documentos jurídicos, como contratos.\n"
        "Com base no texto do contrato fornecido abaixo, crie um resumo executivo em 5 a 7 tópicos concisos (bullet points).\n"
        "Destaque os seguintes aspectos, se presentes: as partes principais envolvidas, o objeto principal do contrato, "
        "prazo de vigência (se houver), principais obrigações financeiras ou condições de pagamento, e as "
        "principais condições ou motivos para rescisão ou cancelamento do contrato.\n"
        "Seja claro e direto.\n\n"
        "TEXTO DO CONTRATO:\n{texto_contrato}\n\n"
        "RESUMO EXECUTIVO:"
    )
    chain_resumo = LLMChain(llm=llm_resumo, prompt=template_prompt_resumo)
    try:
        resultado = chain_resumo.invoke({"texto_contrato": texto_completo})
        return resultado['text']
    except Exception as e:
        return f"Erro ao gerar resumo: {e}"

# --- NOVA FUNÇÃO PARA FORMATAR O CHAT PARA EXPORTAÇÃO ---
def formatar_chat_para_markdown(mensagens_chat):
    texto_formatado = "# Histórico da Conversa com Analisador-IA\n\n"
    for mensagem in mensagens_chat:
        if mensagem["role"] == "user":
            texto_formatado += f"## Você:\n{mensagem['content']}\n\n"
        elif mensagem["role"] == "assistant":
            texto_formatado += f"## IA:\n{mensagem['content']}\n"
            if "sources" in mensagem and mensagem["sources"]:
                texto_formatado += "### Fontes Utilizadas:\n"
                for i, doc in enumerate(mensagem["sources"]):
                    texto_fonte_original = doc.page_content
                    sentenca_chave = mensagem.get("sentenca_chave")
                    # Prepara o texto da fonte para Markdown (escapa caracteres especiais se necessário)
                    texto_fonte_md = texto_fonte_original.replace('\n', '  \n') # Para quebras de linha em MD
                    
                    if sentenca_chave and sentenca_chave in texto_fonte_original:
                        # No Markdown, o destaque pode ser feito com ** (negrito) ou * (itálico)
                        # Usar negrito para a sentença chave
                        texto_formatado_fonte = texto_fonte_md.replace(sentenca_chave, f"**{sentenca_chave}**")
                    else:
                        texto_formatado_fonte = texto_fonte_md
                    
                    texto_formatado += f"- **Fonte {i+1} (Documento: `{doc.metadata.get('source', 'N/A')}`, Página: {doc.metadata.get('page', 'N/A')})**:\n"
                    texto_formatado += f"  > {texto_formatado_fonte[:300]}...\n\n" # Usando blockquote para o trecho
            texto_formatado += "---\n\n" # Separador entre mensagens da IA
    return texto_formatado

# --- LAYOUT PRINCIPAL ---
st.title("⚖️ Analisador de Contratos IA")
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
if "resumo_gerado" not in st.session_state:
    st.session_state.resumo_gerado = ""
if "arquivo_resumido" not in st.session_state:
    st.session_state.arquivo_resumido = None

# --- LÓGICA DAS ABAS ---
tab_chat, tab_dashboard, tab_resumo = st.tabs(["💬 Chat com Contratos", "📈 Dashboard Analítico", "📜 Resumo Executivo"])

# --- ABA DE CHAT COM BOTÃO DE EXPORTAÇÃO ---
with tab_chat:
    st.header("Converse com seus documentos")
    
    # Template de prompt específico para o chat com highlight
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
    
    if arquivos_pdf and google_api_key:
        vector_store_chat = obter_vector_store(arquivos_pdf)

        # NOVA SEÇÃO: Botão de Exportação
        if st.session_state.messages: # Mostra o botão apenas se houver mensagens
            chat_exportado_md = formatar_chat_para_markdown(st.session_state.messages)
            agora = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="📥 Exportar Conversa (Markdown)",
                data=chat_exportado_md,
                file_name=f"conversa_contratos_{agora}.md",
                mime="text/markdown"
            )
        st.markdown("---") # Linha separadora

        if vector_store_chat:
            if not st.session_state.messages:
                st.session_state.messages.append({"role": "assistant", "content": "Olá! Seus documentos foram analisados. Qual sua pergunta?"})

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
                            retriever=vector_store_chat.as_retriever(search_kwargs={"k": 5}),
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
                            resposta_principal, sentenca_chave = resposta_bruta, None
                        
                        st.markdown(resposta_principal)
                        st.session_state.messages.append({"role": "assistant", "content": resposta_principal, "sources": fontes, "sentenca_chave": sentenca_chave})
                        st.rerun()
        else:
             st.warning("O motor de análise de documentos não pôde ser iniciado. Verifique o upload e a chave de API.")
    else:
        st.info("Por favor, faça o upload de um ou mais documentos e configure a chave de API na barra lateral para começar.")


# --- ABA DE DASHBOARD (sem alterações) ---
with tab_dashboard:
    # (Lógica do Dashboard permanece a mesma da versão anterior)
    st.header("Análise Comparativa de Políticas Contratuais")
    st.markdown("Clique no botão abaixo para extrair e comparar as **políticas e condições chave** de todos os documentos carregados.")
    if arquivos_pdf and google_api_key:
        if st.button("🚀 Gerar Análise Comparativa de Políticas"):
            vector_store_dash = obter_vector_store(arquivos_pdf)
            if vector_store_dash:
                nomes_arquivos = [f.name for f in arquivos_pdf]
                dados_extraidos = extrair_dados_dos_contratos(vector_store_dash, nomes_arquivos)
                if dados_extraidos: st.session_state.df_dashboard = pd.DataFrame(dados_extraidos)
            else: st.error("Não foi possível analisar os documentos para o dashboard.")
        if 'df_dashboard' in st.session_state:
            st.info("Tabela de políticas contratuais. Use a barra de rolagem horizontal para ver todas as colunas.")
            st.dataframe(st.session_state.df_dashboard)
    else:
        st.info("Por favor, faça o upload dos documentos e configure a chave de API na barra lateral para ativar o dashboard.")

# --- ABA DE RESUMO EXECUTIVO (sem alterações) ---
with tab_resumo:
    # (Lógica do Resumo permanece a mesma da versão anterior)
    st.header("📜 Resumo Executivo de um Contrato")
    if arquivos_pdf and google_api_key:
        lista_nomes_arquivos = [f.name for f in arquivos_pdf]
        arquivo_selecionado_nome = st.selectbox(
            "Escolha um contrato para resumir:",
            options=lista_nomes_arquivos,
            key="select_arquivo_resumo"
        )
        if st.button("✍️ Gerar Resumo Executivo"):
            arquivo_obj_selecionado = next((arq for arq in arquivos_pdf if arq.name == arquivo_selecionado_nome), None)
            if arquivo_obj_selecionado:
                resumo = gerar_resumo_executivo(arquivo_obj_selecionado.getvalue(), arquivo_obj_selecionado.name, google_api_key)
                st.session_state.resumo_gerado = resumo
                st.session_state.arquivo_resumido = arquivo_selecionado_nome
            else: st.error("Arquivo selecionado não encontrado.")
        if st.session_state.get("arquivo_resumido") == arquivo_selecionado_nome and st.session_state.resumo_gerado:
            st.subheader(f"Resumo do Contrato: {st.session_state.arquivo_resumido}")
            st.markdown(st.session_state.resumo_gerado)
        elif not st.session_state.resumo_gerado and st.session_state.get("arquivo_resumido") == arquivo_selecionado_nome :
             st.warning("Clique em 'Gerar Resumo Executivo' para ver o resumo deste documento.")
    else:
        st.info("Por favor, faça o upload de um ou mais documentos e configure a chave de API na barra lateral para usar esta funcionalidade.")
