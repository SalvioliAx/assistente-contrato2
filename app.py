import streamlit as st
import os
import pandas as pd
from typing import Optional, List
import re
from datetime import datetime
import json # Para salvar a lista de nomes de arquivos da coleção
from pathlib import Path # Para gerenciamento de pastas

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
COLECOES_DIR.mkdir(exist_ok=True) # Cria a pasta se não existir

# --- SCHEMA DE DADOS PARA O DASHBOARD (sem alterações) ---
class InfoContrato(BaseModel):
    arquivo_fonte: str = Field(description="O nome do arquivo de origem do contrato.")
    nome_banco: Optional[str] = Field(default="Não encontrado", description="O nome do banco ou instituição financeira emissora do cartão.")
    # ... (outros campos como antes) ...
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
    st.sidebar.warning("Chave de API do Google não configurada nos Secrets.")
    google_api_key = st.sidebar.text_input("(OU) Cole sua Chave de API do Google aqui:", type="password")
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
    else:
        google_api_key = None # Garante que é None se não for fornecido

hide_streamlit_style = "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>"
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- FUNÇÕES DE GERENCIAMENTO DE COLEÇÕES ---

def listar_colecoes_salvas():
    if not COLECOES_DIR.exists():
        return []
    return [d.name for d in COLECOES_DIR.iterdir() if d.is_dir()]

def salvar_colecao_atual(nome_colecao, vector_store_atual, nomes_arquivos_atuais):
    if not nome_colecao.strip():
        st.error("Por favor, forneça um nome para a coleção.")
        return False
    
    caminho_colecao = COLECOES_DIR / nome_colecao
    try:
        caminho_colecao.mkdir(parents=True, exist_ok=True) # Cria a pasta, permitindo sobrescrever
        vector_store_atual.save_local(str(caminho_colecao / "faiss_index"))
        with open(caminho_colecao / "manifest.json", "w") as f:
            json.dump(nomes_arquivos_atuais, f)
        st.success(f"Coleção '{nome_colecao}' salva com sucesso!")
        return True
    except Exception as e:
        st.error(f"Erro ao salvar a coleção: {e}")
        return False

@st.cache_resource(show_spinner="Carregando coleção...")
def carregar_colecao(nome_colecao, _embeddings_obj): # Adicionado _embeddings_obj
    caminho_colecao = COLECOES_DIR / nome_colecao
    caminho_indice = caminho_colecao / "faiss_index"
    caminho_manifesto = caminho_colecao / "manifest.json"

    if not caminho_indice.exists() or not caminho_manifesto.exists():
        st.error(f"Coleção '{nome_colecao}' está incompleta ou corrompida.")
        return None, None
    try:
        vector_store = FAISS.load_local(str(caminho_indice), embeddings=_embeddings_obj, allow_dangerous_deserialization=True)
        with open(caminho_manifesto, "r") as f:
            nomes_arquivos = json.load(f)
        st.success(f"Coleção '{nome_colecao}' carregada!")
        return vector_store, nomes_arquivos
    except Exception as e:
        st.error(f"Erro ao carregar a coleção '{nome_colecao}': {e}")
        return None, None

# --- FUNÇÕES DE PROCESSAMENTO DE DOCUMENTOS ---

@st.cache_resource(show_spinner="Analisando documentos...")
def obter_vector_store_de_uploads(lista_arquivos_pdf_upload, _embeddings_obj):
    if not lista_arquivos_pdf_upload or not google_api_key: return None
    documentos_totais = []
    for arquivo_pdf in lista_arquivos_pdf_upload:
        # Salva temporariamente para o loader ler
        temp_file_path = Path(arquivo_pdf.name)
        with open(temp_file_path, "wb") as f: f.write(arquivo_pdf.getbuffer())
        
        loader = PyPDFLoader(str(temp_file_path))
        pages = loader.load()
        for page in pages: page.metadata["source"] = arquivo_pdf.name # Usar o nome original
        documentos_totais.extend(pages)
        
        os.remove(temp_file_path) # Limpa o arquivo temporário

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs_fragmentados = text_splitter.split_documents(documentos_totais)
    vector_store = FAISS.from_documents(docs_fragmentados, _embeddings_obj)
    return vector_store, [f.name for f in lista_arquivos_pdf_upload]


@st.cache_data(show_spinner="Extraindo políticas dos contratos...")
def extrair_dados_dos_contratos(_vector_store: FAISS, _nomes_arquivos: list) -> list:
    # (sem alterações nesta função, mas agora pode receber _vector_store de uma coleção)
    if not _vector_store or not google_api_key: return []
    # ... (resto da função como antes) ...
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
def gerar_resumo_executivo(arquivo_pdf_bytes, nome_arquivo_original, google_api_key_func):
    # (sem alterações nesta função)
    if not arquivo_pdf_bytes or not google_api_key_func: return "Erro: Arquivo ou chave de API não fornecidos."
    # ... (resto da função como antes) ...
    with open(nome_arquivo_original, "wb") as f: f.write(arquivo_pdf_bytes)
    loader = PyPDFLoader(nome_arquivo_original)
    documento_completo_paginas = loader.load()
    os.remove(nome_arquivo_original)
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
    except Exception as e: return f"Erro ao gerar resumo: {e}"


def formatar_chat_para_markdown(mensagens_chat):
    # (sem alterações nesta função)
    texto_formatado = "# Histórico da Conversa com Analisador-IA\n\n"
    # ... (resto da função como antes) ...
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
                    texto_fonte_md = texto_fonte_original.replace('\n', '  \n')
                    if sentenca_chave and sentenca_chave in texto_fonte_original:
                        texto_formatado_fonte = texto_fonte_md.replace(sentenca_chave, f"**{sentenca_chave}**")
                    else: texto_formatado_fonte = texto_fonte_md
                    texto_formatado += f"- **Fonte {i+1} (Documento: `{doc.metadata.get('source', 'N/A')}`, Página: {doc.metadata.get('page', 'N/A')})**:\n"
                    texto_formatado += f"  > {texto_formatado_fonte[:300]}...\n\n"
            texto_formatado += "---\n\n"
    return texto_formatado


# --- INICIALIZAÇÃO DO OBJETO DE EMBEDDINGS (GLOBALMENTE OU CACHEADO) ---
# Necessário para carregar o índice FAISS
if google_api_key:
    embeddings_global = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
else:
    embeddings_global = None


# --- LAYOUT PRINCIPAL ---
st.title("⚖️ Analisador de Contratos IA")

# --- SIDEBAR COM GERENCIAMENTO DE COLEÇÕES ---
st.sidebar.header("Gerenciar Documentos")
modo_documento = st.sidebar.radio(
    "Como você quer carregar os documentos?",
    ("Fazer novo upload de PDFs", "Carregar coleção existente")
)

arquivos_pdf_upload = None # Para o upload
if modo_documento == "Fazer novo upload de PDFs":
    arquivos_pdf_upload = st.sidebar.file_uploader(
        "Selecione um ou mais contratos em PDF", type="pdf", accept_multiple_files=True
    )
    if arquivos_pdf_upload:
        if st.sidebar.button("Processar Documentos Carregados"):
            st.session_state.vector_store, st.session_state.nomes_arquivos = obter_vector_store_de_uploads(arquivos_pdf_upload, embeddings_global)
            st.session_state.colecao_ativa = None # Limpa coleção ativa se fizer novo upload
            st.session_state.messages = [] # Limpa chat
            st.session_state.pop('df_dashboard', None) # Limpa dashboard
            st.session_state.pop('resumo_gerado', None) # Limpa resumo
            st.rerun()

elif modo_documento == "Carregar coleção existente":
    colecoes_disponiveis = listar_colecoes_salvas()
    if colecoes_disponiveis:
        colecao_selecionada = st.sidebar.selectbox("Escolha uma coleção:", colecoes_disponiveis)
        if st.sidebar.button("Carregar Coleção Selecionada"):
            vs, nomes_arqs = carregar_colecao(colecao_selecionada, embeddings_global)
            if vs and nomes_arqs:
                st.session_state.vector_store = vs
                st.session_state.nomes_arquivos = nomes_arqs
                st.session_state.colecao_ativa = colecao_selecionada
                st.session_state.messages = [] # Limpa chat
                st.session_state.pop('df_dashboard', None) # Limpa dashboard
                st.session_state.pop('resumo_gerado', None) # Limpa resumo
                st.rerun()
    else:
        st.sidebar.info("Nenhuma coleção salva ainda.")

# Se um vector_store está ativo (de upload ou coleção), mostra opção de salvar
if "vector_store" in st.session_state and st.session_state.vector_store is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Salvar Coleção Atual")
    nome_nova_colecao = st.sidebar.text_input("Nome para a nova coleção:")
    if st.sidebar.button("Salvar Coleção"):
        if nome_nova_colecao and st.session_state.nomes_arquivos:
            salvar_colecao_atual(nome_nova_colecao, st.session_state.vector_store, st.session_state.nomes_arquivos)
            # Atualizar a lista de coleções disponíveis sem rerun, se possível, ou indicar que foi salva.
        else:
            st.sidebar.warning("Dê um nome para a coleção e certifique-se de que há documentos carregados.")

if "colecao_ativa" in st.session_state and st.session_state.colecao_ativa:
    st.sidebar.markdown(f"**Coleção Ativa:** `{st.session_state.colecao_ativa}`")
elif "nomes_arquivos" in st.session_state and st.session_state.nomes_arquivos:
    st.sidebar.markdown(f"**Arquivos Carregados:** {len(st.session_state.nomes_arquivos)}")


st.sidebar.header("Configurações de Idioma")
idioma_selecionado = st.sidebar.selectbox(
    "Selecione o idioma para as respostas do CHAT:",
    ("Português", "Inglês", "Espanhol"),
    key="idioma_chat_key"
)

# --- INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
if "messages" not in st.session_state: st.session_state.messages = []
if "resumo_gerado" not in st.session_state: st.session_state.resumo_gerado = ""
if "arquivo_resumido" not in st.session_state: st.session_state.arquivo_resumido = None
if "df_dashboard" not in st.session_state: st.session_state.df_dashboard = None

# --- LÓGICA DAS ABAS ---
tab_chat, tab_dashboard, tab_resumo = st.tabs(["💬 Chat com Contratos", "📈 Dashboard Analítico", "📜 Resumo Executivo"])

# Condição principal para habilitar as abas: Chave de API e Vector Store devem existir
if google_api_key and "vector_store" in st.session_state and st.session_state.vector_store is not None:
    vector_store_global = st.session_state.vector_store
    nomes_arquivos_global = st.session_state.nomes_arquivos

    # --- ABA DE CHAT ---
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
        if not st.session_state.messages :
            st.session_state.messages.append({"role": "assistant", "content": f"Olá! Documentos da coleção '{st.session_state.get('colecao_ativa', 'atual')}' prontos. Qual sua pergunta?"})

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
        
        if st.session_state.messages : # Botão de exportação
            chat_exportado_md = formatar_chat_para_markdown(st.session_state.messages)
            agora = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(label="📥 Exportar Conversa",data=chat_exportado_md,
                               file_name=f"conversa_contratos_{agora}.md", mime="text/markdown",
                               key="export_chat_btn")
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
                    except ValueError:
                        resposta_principal, sentenca_chave = resposta_bruta, None
                    st.markdown(resposta_principal)
                    st.session_state.messages.append({"role": "assistant", "content": resposta_principal, "sources": fontes, "sentenca_chave": sentenca_chave})
                    st.rerun()
    
    # --- ABA DE DASHBOARD ---
    with tab_dashboard:
        st.header("Análise Comparativa de Políticas Contratuais")
        st.markdown("Clique no botão para extrair e comparar as políticas chave dos documentos carregados.")
        if st.button("🚀 Gerar Análise Comparativa de Políticas", key="btn_dashboard"):
            dados_extraidos = extrair_dados_dos_contratos(vector_store_global, nomes_arquivos_global)
            if dados_extraidos: st.session_state.df_dashboard = pd.DataFrame(dados_extraidos)
        
        if st.session_state.df_dashboard is not None:
            st.info("Tabela de políticas contratuais. Use a barra de rolagem horizontal.")
            st.dataframe(st.session_state.df_dashboard)
            # (Lógica de estatísticas omitida para brevidade, mas pode ser adicionada como antes)

    # --- ABA DE RESUMO EXECUTIVO ---
    with tab_resumo:
        st.header("📜 Resumo Executivo de um Contrato")
        if nomes_arquivos_global:
            arquivo_selecionado_nome_resumo = st.selectbox(
                "Escolha um contrato para resumir:", options=nomes_arquivos_global, key="select_resumo"
            )
            if st.button("✍️ Gerar Resumo Executivo", key="btn_resumo"):
                # Precisamos encontrar o objeto UploadedFile original para obter os bytes
                # Esta parte é um pouco mais complexa se só tivermos nomes de arquivos de uma coleção salva
                # Para simplificar, vamos assumir que `arquivos_pdf_upload` ainda está disponível se for de um upload novo
                # Ou precisaríamos de uma forma de reler o arquivo da coleção (não implementado aqui para simplicidade)
                arquivo_obj_selecionado = None
                if arquivos_pdf_upload: # Se os arquivos vieram de um upload novo
                    arquivo_obj_selecionado = next((arq for arq in arquivos_pdf_upload if arq.name == arquivo_selecionado_nome_resumo), None)

                if arquivo_obj_selecionado:
                    resumo = gerar_resumo_executivo(arquivo_obj_selecionado.getvalue(), arquivo_obj_selecionado.name, google_api_key)
                    st.session_state.resumo_gerado = resumo
                    st.session_state.arquivo_resumido = arquivo_selecionado_nome_resumo
                else:
                    st.warning("Para gerar resumo de coleções salvas, essa funcionalidade precisaria ser expandida para recarregar o conteúdo original do arquivo.")
            
            if st.session_state.get("arquivo_resumido") == arquivo_selecionado_nome_resumo and st.session_state.resumo_gerado:
                st.subheader(f"Resumo do Contrato: {st.session_state.arquivo_resumido}")
                st.markdown(st.session_state.resumo_gerado)

else:
    st.info("Por favor, faça o upload de documentos ou carregue uma coleção, e configure a chave de API na barra lateral para começar.")
