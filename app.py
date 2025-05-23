import streamlit as st
import os
import pandas as pd
from typing import Optional, List
import re
from datetime import datetime
import json
from pathlib import Path

# Importações do LangChain e Pydantic
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate

# --- DEFINIÇÕES GLOBAIS E DIRETÓRIO DE COLEÇÕES ---
COLECOES_DIR = Path("colecoes_ia")
COLECOES_DIR.mkdir(exist_ok=True)

# --- SCHEMAS DE DADOS ---
class InfoContrato(BaseModel): # Para o Dashboard
    arquivo_fonte: str = Field(description="O nome do arquivo de origem do contrato.")
    nome_banco: Optional[str] = Field(default="Não encontrado", description="O nome do banco ou instituição financeira emissora do cartão.")
    condicao_limite_credito: Optional[str] = Field(default="Não encontrado", description="Resumo da política de como o limite de crédito é definido, analisado e alterado.")
    condicao_juros_rotativo: Optional[str] = Field(default="Não encontrado", description="Resumo da regra de como e quando os juros do crédito rotativo são aplicados.")
    condicao_anuidade: Optional[str] = Field(default="Não encontrado", description="Resumo da política de cobrança da anuidade, se é diferenciada ou básica e como é cobrada.")
    condicao_cancelamento: Optional[str] = Field(default="Não encontrado", description="Resumo das condições sob as quais o contrato pode ser rescindido ou cancelado pelo banco ou pelo cliente.")

# --- CONFIGURAÇÃO DA PÁGINA E DA CHAVE DE API ---
st.set_page_config(layout="wide", page_title="Analisador-IA Pro", page_icon="🛡️")
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = google_api_key
except (KeyError, FileNotFoundError):
    st.sidebar.warning("Chave de API do Google não configurada nos Secrets.")
    google_api_key = st.sidebar.text_input("(OU) Cole sua Chave de API do Google aqui:", type="password", key="api_key_input_main")
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
    else:
        google_api_key = None

hide_streamlit_style = "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>"
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- FUNÇÕES DE GERENCIAMENTO DE COLEÇÕES ---
# (Funções listar_colecoes_salvas, salvar_colecao_atual, carregar_colecao permanecem as mesmas)
def listar_colecoes_salvas():
    if not COLECOES_DIR.exists(): return []
    return [d.name for d in COLECOES_DIR.iterdir() if d.is_dir()]

def salvar_colecao_atual(nome_colecao, vector_store_atual, nomes_arquivos_atuais):
    if not nome_colecao.strip():
        st.error("Por favor, forneça um nome para a coleção."); return False
    caminho_colecao = COLECOES_DIR / nome_colecao
    try:
        caminho_colecao.mkdir(parents=True, exist_ok=True)
        vector_store_atual.save_local(str(caminho_colecao / "faiss_index"))
        with open(caminho_colecao / "manifest.json", "w") as f: json.dump(nomes_arquivos_atuais, f)
        st.success(f"Coleção '{nome_colecao}' salva com sucesso!"); return True
    except Exception as e: st.error(f"Erro ao salvar coleção: {e}"); return False

@st.cache_resource(show_spinner="Carregando coleção...")
def carregar_colecao(nome_colecao, _embeddings_obj):
    # ... (código da função inalterado)
    caminho_colecao = COLECOES_DIR / nome_colecao
    caminho_indice = caminho_colecao / "faiss_index"
    caminho_manifesto = caminho_colecao / "manifest.json"
    if not caminho_indice.exists() or not caminho_manifesto.exists():
        st.error(f"Coleção '{nome_colecao}' está incompleta ou corrompida."); return None, None
    try:
        vector_store = FAISS.load_local(str(caminho_indice), embeddings=_embeddings_obj, allow_dangerous_deserialization=True)
        with open(caminho_manifesto, "r") as f: nomes_arquivos = json.load(f)
        st.success(f"Coleção '{nome_colecao}' carregada!"); return vector_store, nomes_arquivos
    except Exception as e: st.error(f"Erro ao carregar coleção '{nome_colecao}': {e}"); return None, None


# --- FUNÇÕES DE PROCESSAMENTO DE DOCUMENTOS ---
@st.cache_resource(show_spinner="Analisando documentos para busca e chat...")
def obter_vector_store_de_uploads(lista_arquivos_pdf_upload, _embeddings_obj):
    # (sem alterações)
    if not lista_arquivos_pdf_upload or not google_api_key: return None
    # ... (código da função inalterado) ...
    documentos_totais = []
    for arquivo_pdf in lista_arquivos_pdf_upload:
        temp_file_path = Path(arquivo_pdf.name) # Usar Path para consistência
        with open(temp_file_path, "wb") as f: f.write(arquivo_pdf.getbuffer())
        loader = PyPDFLoader(str(temp_file_path))
        pages = loader.load()
        for page in pages: page.metadata["source"] = arquivo_pdf.name
        documentos_totais.extend(pages)
        os.remove(temp_file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs_fragmentados = text_splitter.split_documents(documentos_totais)
    vector_store = FAISS.from_documents(docs_fragmentados, _embeddings_obj)
    return vector_store, [f.name for f in lista_arquivos_pdf_upload]


@st.cache_data(show_spinner="Extraindo políticas para o dashboard...")
def extrair_dados_dos_contratos(_vector_store: FAISS, _nomes_arquivos: list) -> list:
    # (sem alterações)
    if not _vector_store or not google_api_key: return []
    # ... (código da função inalterado) ...
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
    st.success("Análise de políticas para dashboard concluída!")
    return resultados_finais

@st.cache_data(show_spinner="Gerando resumo executivo...")
def gerar_resumo_executivo(arquivo_pdf_bytes, nome_arquivo_original):
    # (sem alterações, mas chave de API é global)
    if not arquivo_pdf_bytes or not google_api_key: return "Erro: Arquivo ou chave de API não fornecidos."
    # ... (código da função inalterado) ...
    with open(nome_arquivo_original, "wb") as f: f.write(arquivo_pdf_bytes)
    loader = PyPDFLoader(nome_arquivo_original)
    documento_completo_paginas = loader.load()
    os.remove(nome_arquivo_original)
    texto_completo = "\n\n".join([page.page_content for page in documento_completo_paginas])
    llm_resumo = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    template_prompt_resumo = PromptTemplate.from_template(
        # ... (prompt do resumo inalterado) ...
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
    except Exception as e: return f"Erro ao gerar resumo: {e}"

# --- NOVA FUNÇÃO PARA ANÁLISE DE RISCOS ---
@st.cache_data(show_spinner="Analisando riscos no documento...")
def analisar_documento_para_riscos(texto_completo_doc, nome_arquivo_doc):
    if not texto_completo_doc or not google_api_key:
        return f"Não foi possível analisar riscos para '{nome_arquivo_doc}': Texto ou Chave API ausente."

    llm_riscos = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
    # Prompt mais elaborado para análise de riscos
    prompt_riscos_template = PromptTemplate.from_template(
        "Você é um advogado especialista em análise de riscos contratuais. "
        "Analise o texto do contrato fornecido abaixo e identifique cláusulas ou omissões que possam representar riscos significativos. "
        "Para cada risco identificado, por favor:\n"
        "1. Descreva o risco de forma clara e concisa.\n"
        "2. Cite o trecho exato da cláusula relevante (ou mencione a ausência de uma cláusula esperada).\n"
        "3. Classifique o risco (ex: Financeiro, Operacional, Legal, Rescisão, Propriedade Intelectual, Confidencialidade, etc.).\n"
        "Concentre-se nos riscos mais impactantes. Se nenhum risco significativo for encontrado, declare isso explicitamente.\n"
        "Use formatação Markdown para sua resposta, com um título para cada risco.\n\n"
        "TEXTO DO CONTRATO ({nome_arquivo}):\n{texto_contrato}\n\n"
        "ANÁLISE DE RISCOS:"
    )
    chain_riscos = LLMChain(llm=llm_riscos, prompt=prompt_riscos_template)
    try:
        resultado = chain_riscos.invoke({"nome_arquivo": nome_arquivo_doc, "texto_contrato": texto_completo_doc})
        return resultado['text']
    except Exception as e:
        return f"Erro ao analisar riscos para '{nome_arquivo_doc}': {e}"

def formatar_chat_para_markdown(mensagens_chat):
    # (sem alterações)
    texto_formatado = "# Histórico da Conversa com Analisador-IA\n\n"
    # ... (código da função inalterado) ...
    for mensagem in mensagens_chat:
        if mensagem["role"] == "user":
            texto_formatado += f"## Você:\n{mensagem['content']}\n\n"
        elif mensagem["role"] == "assistant":
            texto_formatado += f"## IA:\n{mensagem['content']}\n"
            if "sources" in mensagem and mensagem["sources"]:
                texto_formatado += "### Fontes Utilizadas:\n"
                for i, doc_fonte in enumerate(mensagem["sources"]):
                    texto_fonte_original = doc_fonte.page_content
                    sentenca_chave = mensagem.get("sentenca_chave")
                    texto_fonte_md = texto_fonte_original.replace('\n', '  \n')
                    if sentenca_chave and sentenca_chave in texto_fonte_original:
                        texto_formatado_fonte = texto_fonte_md.replace(sentenca_chave, f"**{sentenca_chave}**")
                    else: texto_formatado_fonte = texto_fonte_md
                    texto_formatado += f"- **Fonte {i+1} (Documento: `{doc_fonte.metadata.get('source', 'N/A')}`, Página: {doc_fonte.metadata.get('page', 'N/A')})**:\n"
                    texto_formatado += f"  > {texto_formatado_fonte[:300]}...\n\n" # Usando blockquote para o trecho
            texto_formatado += "---\n\n"
    return texto_formatado

# --- INICIALIZAÇÃO DO OBJETO DE EMBEDDINGS ---
if google_api_key:
    embeddings_global = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
else:
    embeddings_global = None
    st.error("Chave de API do Google não configurada. Algumas funcionalidades podem não operar.")


# --- LAYOUT PRINCIPAL ---
st.title("🛡️ Analisador-IA Pro")

# --- SIDEBAR COM GERENCIAMENTO DE COLEÇÕES ---
# (Sidebar permanece o mesmo da versão anterior)
st.sidebar.header("Gerenciar Documentos")
modo_documento = st.sidebar.radio(
    "Como você quer carregar os documentos?",
    ("Fazer novo upload de PDFs", "Carregar coleção existente"),
    key="modo_doc_radio"
)
arquivos_pdf_upload_sidebar = None
if modo_documento == "Fazer novo upload de PDFs":
    arquivos_pdf_upload_sidebar = st.sidebar.file_uploader(
        "Selecione um ou mais contratos em PDF", type="pdf", accept_multiple_files=True, key="uploader_sidebar"
    )
    if arquivos_pdf_upload_sidebar:
        if st.sidebar.button("Processar Documentos Carregados", key="btn_proc_upload_sidebar"):
            st.session_state.vector_store, st.session_state.nomes_arquivos = obter_vector_store_de_uploads(arquivos_pdf_upload_sidebar, embeddings_global)
            st.session_state.arquivos_pdf_originais = arquivos_pdf_upload_sidebar # Salva os objetos de arquivo para a análise de riscos
            st.session_state.colecao_ativa = None
            st.session_state.messages = []
            st.session_state.pop('df_dashboard', None); st.session_state.pop('resumo_gerado', None); st.session_state.pop('analise_riscos_resultados', None)
            st.rerun()
elif modo_documento == "Carregar coleção existente":
    # ... (lógica de carregar coleção inalterada) ...
    colecoes_disponiveis = listar_colecoes_salvas()
    if colecoes_disponiveis:
        colecao_selecionada = st.sidebar.selectbox("Escolha uma coleção:", colecoes_disponiveis, key="select_colecao_sidebar")
        if st.sidebar.button("Carregar Coleção Selecionada", key="btn_load_colecao_sidebar"):
            vs, nomes_arqs = carregar_colecao(colecao_selecionada, embeddings_global)
            if vs and nomes_arqs:
                st.session_state.vector_store = vs
                st.session_state.nomes_arquivos = nomes_arqs
                st.session_state.colecao_ativa = colecao_selecionada
                st.session_state.arquivos_pdf_originais = None # Limpa, pois não temos os objetos de arquivo aqui
                st.session_state.messages = []
                st.session_state.pop('df_dashboard', None); st.session_state.pop('resumo_gerado', None); st.session_state.pop('analise_riscos_resultados', None)
                st.rerun()
    else: st.sidebar.info("Nenhuma coleção salva ainda.")

if "vector_store" in st.session_state and st.session_state.vector_store is not None and arquivos_pdf_upload_sidebar: # Só permite salvar se veio de um upload novo
    st.sidebar.markdown("---")
    st.sidebar.subheader("Salvar Coleção Atual")
    nome_nova_colecao = st.sidebar.text_input("Nome para a nova coleção:", key="input_nome_colecao_sidebar")
    if st.sidebar.button("Salvar Coleção", key="btn_save_colecao_sidebar"):
        if nome_nova_colecao and st.session_state.nomes_arquivos:
            salvar_colecao_atual(nome_nova_colecao, st.session_state.vector_store, st.session_state.nomes_arquivos)
        else: st.sidebar.warning("Dê um nome e certifique-se de que há docs carregados.")

if "colecao_ativa" in st.session_state and st.session_state.colecao_ativa:
    st.sidebar.markdown(f"**Coleção Ativa:** `{st.session_state.colecao_ativa}`")
elif "nomes_arquivos" in st.session_state and st.session_state.nomes_arquivos:
    st.sidebar.markdown(f"**Arquivos Carregados:** {len(st.session_state.nomes_arquivos)}")

st.sidebar.header("Configurações de Idioma")
idioma_selecionado = st.sidebar.selectbox(
    "Selecione o idioma para as respostas do CHAT:",
    ("Português", "Inglês", "Espanhol"),
    key="idioma_chat_key_sidebar"
)

# --- INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
if "messages" not in st.session_state: st.session_state.messages = []
if "resumo_gerado" not in st.session_state: st.session_state.resumo_gerado = ""
if "arquivo_resumido" not in st.session_state: st.session_state.arquivo_resumido = None
if "df_dashboard" not in st.session_state: st.session_state.df_dashboard = None
if "analise_riscos_resultados" not in st.session_state: st.session_state.analise_riscos_resultados = {}

# --- LÓGICA DAS ABAS ---
# MUDANÇA: Adicionada a nova aba
tab_chat, tab_dashboard, tab_resumo, tab_riscos = st.tabs([
    "💬 Chat com Contratos", 
    "📈 Dashboard Analítico", 
    "📜 Resumo Executivo",
    "🚩 Análise de Riscos"
])

# Condição principal para habilitar as abas
if google_api_key and (("vector_store" in st.session_state and st.session_state.vector_store is not None) or ("arquivos_pdf_originais" in st.session_state and st.session_state.arquivos_pdf_originais is not None)):
    vector_store_global = st.session_state.get("vector_store")
    nomes_arquivos_global = st.session_state.get("nomes_arquivos", [])
    arquivos_pdf_originais_global = st.session_state.get("arquivos_pdf_originais") # Usado para Resumo e Análise de Riscos

    with tab_chat:
        # (Lógica do Chat permanece a mesma)
        st.header("Converse com seus documentos")
        if not vector_store_global:
            st.warning("O motor de busca de documentos não está pronto. Verifique o upload/coleção.")
        else:
            # ... (código completo do chat como na versão anterior) ...
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
            if not st.session_state.messages :
                st.session_state.messages.append({"role": "assistant", "content": f"Olá! Documentos da coleção '{st.session_state.get('colecao_ativa', 'atual')}' prontos. Qual sua pergunta?"})
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "sources" in message:
                        with st.expander("Ver Fontes Utilizadas"):
                            for doc_fonte in message["sources"]:
                                texto_fonte = doc_fonte.page_content
                                sentenca_chave = message.get("sentenca_chave")
                                if sentenca_chave and sentenca_chave in texto_fonte:
                                    texto_formatado = texto_fonte.replace(sentenca_chave, f"<span style='background-color: #FFFACD; padding: 2px; border-radius: 3px;'>{sentenca_chave}</span>")
                                else: texto_formatado = texto_fonte
                                st.markdown(f"**Fonte: `{doc_fonte.metadata.get('source', 'N/A')}` (Página {doc_fonte.metadata.get('page', 'N/A')})**")
                                st.markdown(texto_formatado, unsafe_allow_html=True)
            if st.session_state.messages :
                chat_exportado_md = formatar_chat_para_markdown(st.session_state.messages)
                agora = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(label="📥 Exportar Conversa",data=chat_exportado_md,
                                   file_name=f"conversa_contratos_{agora}.md", mime="text/markdown",
                                   key="export_chat_btn_tab")
                st.markdown("---")
            if prompt := st.chat_input("Faça sua pergunta sobre os contratos..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Pesquisando..."):
                        llm_chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm_chat, chain_type="stuff",
                            retriever=vector_store_global.as_retriever(search_kwargs={"k": 5}),
                            return_source_documents=True,
                            chain_type_kwargs={"prompt": template_prompt_chat.partial(language=idioma_selecionado)}
                        )
                        resultado = qa_chain({"query": prompt})
                        resposta_bruta = resultado["result"]
                        fontes = resultado["source_documents"]
                        try:
                            resposta_principal, sentenca_chave = resposta_bruta.split('|||')
                            sentenca_chave = sentenca_chave.strip()
                        except ValueError: resposta_principal, sentenca_chave = resposta_bruta, None
                        st.markdown(resposta_principal)
                        st.session_state.messages.append({"role": "assistant", "content": resposta_principal, "sources": fontes, "sentenca_chave": sentenca_chave})
                        st.rerun()

    with tab_dashboard:
        # (Lógica do Dashboard permanece a mesma)
        st.header("Análise Comparativa de Políticas Contratuais")
        st.markdown("Clique no botão para extrair e comparar as políticas chave dos documentos carregados.")
        if vector_store_global and nomes_arquivos_global:
            if st.button("🚀 Gerar Análise Comparativa de Políticas", key="btn_dashboard_tab"):
                dados_extraidos = extrair_dados_dos_contratos(vector_store_global, nomes_arquivos_global)
                if dados_extraidos: st.session_state.df_dashboard = pd.DataFrame(dados_extraidos)
            if st.session_state.df_dashboard is not None:
                st.info("Tabela de políticas contratuais. Use a barra de rolagem horizontal.")
                st.dataframe(st.session_state.df_dashboard)
        else:
             st.warning("Carregue documentos ou uma coleção para usar o dashboard.")


    with tab_resumo:
        # (Lógica do Resumo permanece a mesma)
        st.header("📜 Resumo Executivo de um Contrato")
        if arquivos_pdf_originais_global: # Funciona melhor se tivermos os arquivos originais da sessão
            lista_nomes_arquivos_resumo = [f.name for f in arquivos_pdf_originais_global]
            arquivo_selecionado_nome_resumo = st.selectbox(
                "Escolha um contrato para resumir:", options=lista_nomes_arquivos_resumo, key="select_resumo_tab"
            )
            if st.button("✍️ Gerar Resumo Executivo", key="btn_resumo_tab"):
                arquivo_obj_selecionado = next((arq for arq in arquivos_pdf_originais_global if arq.name == arquivo_selecionado_nome_resumo), None)
                if arquivo_obj_selecionado:
                    resumo = gerar_resumo_executivo(arquivo_obj_selecionado.getvalue(), arquivo_obj_selecionado.name)
                    st.session_state.resumo_gerado = resumo
                    st.session_state.arquivo_resumido = arquivo_selecionado_nome_resumo
                else: st.error("Arquivo selecionado não encontrado.")
            if st.session_state.get("arquivo_resumido") == arquivo_selecionado_nome_resumo and st.session_state.resumo_gerado:
                st.subheader(f"Resumo do Contrato: {st.session_state.arquivo_resumido}")
                st.markdown(st.session_state.resumo_gerado)
        elif nomes_arquivos_global:
             st.info("A função de resumo funciona melhor com arquivos recém-carregados. Para coleções salvas, esta função pode ser limitada (texto completo não armazenado na coleção).")
        else:
            st.warning("Carregue documentos para usar a função de resumo.")

    # --- NOVA ABA DE ANÁLISE DE RISCOS ---
    with tab_riscos:
        st.header("🚩 Análise de Cláusulas de Risco")
        st.markdown("Esta funcionalidade analisa os documentos carregados na sessão atual em busca de cláusulas potencialmente arriscadas.")

        if arquivos_pdf_originais_global: # Verifica se temos os objetos de arquivo da sessão atual
            if st.button("🔎 Analisar Riscos em Todos os Documentos Carregados", key="btn_analise_riscos"):
                st.session_state.analise_riscos_resultados = {} # Limpa resultados anteriores
                
                textos_completos_docs = []
                for arquivo_pdf_obj in arquivos_pdf_originais_global:
                    with open(arquivo_pdf_obj.name, "wb") as f: f.write(arquivo_pdf_obj.getbuffer())
                    loader = PyPDFLoader(arquivo_pdf_obj.name)
                    texto_doc = "\n\n".join([page.page_content for page in loader.load()])
                    textos_completos_docs.append({"nome": arquivo_pdf_obj.name, "texto": texto_doc})
                    os.remove(arquivo_pdf_obj.name)

                resultados_analise = {}
                for doc_info in textos_completos_docs:
                    st.info(f"Analisando riscos em: {doc_info['nome']}...")
                    resultado_risco = analisar_documento_para_riscos(doc_info["texto"], doc_info["nome"])
                    resultados_analise[doc_info["nome"]] = resultado_risco
                st.session_state.analise_riscos_resultados = resultados_analise
            
            if st.session_state.analise_riscos_resultados:
                st.markdown("---")
                for nome_arquivo, analise in st.session_state.analise_riscos_resultados.items():
                    with st.expander(f"Riscos Identificados em: {nome_arquivo}", expanded=True):
                        st.markdown(analise) # A IA deve retornar Markdown formatado
        
        elif "colecao_ativa" in st.session_state and st.session_state.colecao_ativa:
            st.warning("A Análise de Riscos detalhada funciona melhor com arquivos recém-carregados (sessão atual). Para coleções salvas, o texto completo original não é armazenado, limitando esta análise. Considere fazer um novo upload dos arquivos da coleção se precisar desta funcionalidade.")
        else:
            st.info("Faça o upload de documentos para ativar a análise de riscos.")

else:
    st.info("Por favor, faça o upload de documentos ou carregue uma coleção, e configure a chave de API na barra lateral para começar.")
