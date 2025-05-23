import streamlit as st
import os
import pandas as pd
from typing import Optional, List
import re
from datetime import datetime, date
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
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.messages import AIMessage # Para referência do tipo de objeto

# --- DEFINIÇÕES GLOBAIS ---
COLECOES_DIR = Path("colecoes_ia")
COLECOES_DIR.mkdir(exist_ok=True)

# --- SCHEMAS DE DADOS ---
class InfoContrato(BaseModel):
    arquivo_fonte: str = Field(description="O nome do arquivo de origem do contrato.")
    nome_banco: Optional[str] = Field(default="Não encontrado", description="O nome do banco ou instituição financeira emissora do cartão.")
    condicao_limite_credito: Optional[str] = Field(default="Não encontrado", description="Resumo da política de como o limite de crédito é definido, analisado e alterado.")
    condicao_juros_rotativo: Optional[str] = Field(default="Não encontrado", description="Resumo da regra de como e quando os juros do crédito rotativo são aplicados.")
    condicao_anuidade: Optional[str] = Field(default="Não encontrado", description="Resumo da política de cobrança da anuidade, se é diferenciada ou básica e como é cobrada.")
    condicao_cancelamento: Optional[str] = Field(default="Não encontrado", description="Resumo das condições sob as quais o contrato pode ser rescindido ou cancelado pelo banco ou pelo cliente.")

class EventoContratual(BaseModel):
    descricao_evento: str = Field(description="Uma descrição clara e concisa do evento ou prazo. Ex: 'Vencimento do contrato', 'Data de assinatura', 'Prazo para pagamento da fatura'.")
    data_evento_str: Optional[str] = Field(default="Não Especificado", description="A data do evento no formato YYYY-MM-DD. Se uma data EXATA não puder ser determinada ou não se aplicar, use a string 'Não Especificado'. NUNCA use null ou deixe vazio.")
    trecho_relevante: Optional[str] = Field(default=None, description="O trecho exato do contrato que menciona este evento/data.")

class ListaDeEventos(BaseModel):
    eventos: List[EventoContratual] = Field(description="Lista de eventos contratuais com suas datas.")
    arquivo_fonte: str = Field(description="O nome do arquivo de origem do contrato de onde estes eventos foram extraídos.")

# --- CONFIGURAÇÃO DA PÁGINA E DA CHAVE DE API ---
st.set_page_config(layout="wide", page_title="Analisador-IA Pro", page_icon="🛡️")
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = google_api_key
except (KeyError, FileNotFoundError):
    st.sidebar.warning("Chave de API do Google não configurada nos Secrets.")
    google_api_key = st.sidebar.text_input("(OU) Cole sua Chave de API do Google aqui:", type="password", key="api_key_input_main")
    if google_api_key: os.environ["GOOGLE_API_KEY"] = google_api_key
    else: google_api_key = None
hide_streamlit_style = "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>"
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- FUNÇÕES DE GERENCIAMENTO DE COLEÇÕES ---
def listar_colecoes_salvas():
    if not COLECOES_DIR.exists(): return []
    return [d.name for d in COLECOES_DIR.iterdir() if d.is_dir()]

def salvar_colecao_atual(nome_colecao, vector_store_atual, nomes_arquivos_atuais):
    if not nome_colecao.strip(): st.error("Por favor, forneça um nome para a coleção."); return False
    caminho_colecao = COLECOES_DIR / nome_colecao
    try:
        caminho_colecao.mkdir(parents=True, exist_ok=True)
        vector_store_atual.save_local(str(caminho_colecao / "faiss_index"))
        with open(caminho_colecao / "manifest.json", "w") as f: json.dump(nomes_arquivos_atuais, f)
        st.success(f"Coleção '{nome_colecao}' salva com sucesso!"); return True
    except Exception as e: st.error(f"Erro ao salvar coleção: {e}"); return False

@st.cache_resource(show_spinner="Carregando coleção...")
def carregar_colecao(nome_colecao, _embeddings_obj):
    caminho_colecao = COLECOES_DIR / nome_colecao; caminho_indice = caminho_colecao / "faiss_index"; caminho_manifesto = caminho_colecao / "manifest.json"
    if not caminho_indice.exists() or not caminho_manifesto.exists(): st.error(f"Coleção '{nome_colecao}' incompleta."); return None, None
    try:
        vector_store = FAISS.load_local(str(caminho_indice), embeddings=_embeddings_obj, allow_dangerous_deserialization=True)
        with open(caminho_manifesto, "r") as f: nomes_arquivos = json.load(f)
        st.success(f"Coleção '{nome_colecao}' carregada!"); return vector_store, nomes_arquivos
    except Exception as e: st.error(f"Erro ao carregar coleção '{nome_colecao}': {e}"); return None, None

# --- FUNÇÕES DE PROCESSAMENTO DE DOCUMENTOS ---
@st.cache_resource(show_spinner="Analisando documentos para busca e chat...")
def obter_vector_store_de_uploads(lista_arquivos_pdf_upload, _embeddings_obj):
    if not lista_arquivos_pdf_upload or not google_api_key or not _embeddings_obj : return None, None
    documentos_totais = [];
    for arquivo_pdf in lista_arquivos_pdf_upload:
        temp_file_path = Path(arquivo_pdf.name)
        with open(temp_file_path, "wb") as f: f.write(arquivo_pdf.getbuffer())
        loader = PyPDFLoader(str(temp_file_path)); pages = loader.load()
        for page in pages: page.metadata["source"] = arquivo_pdf.name
        documentos_totais.extend(pages); os.remove(temp_file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs_fragmentados = text_splitter.split_documents(documentos_totais)
    vector_store = FAISS.from_documents(docs_fragmentados, _embeddings_obj)
    return vector_store, [f.name for f in lista_arquivos_pdf_upload]

@st.cache_data(show_spinner="Extraindo políticas para o dashboard...")
def extrair_dados_dos_contratos(_vector_store: FAISS, _nomes_arquivos: list) -> list:
    if not _vector_store or not google_api_key or not _nomes_arquivos: return []
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    prompt_template = PromptTemplate.from_template(
        "Do texto abaixo, resuma em uma ou duas frases a resposta para a seguinte pergunta: '{info_desejada}'.\n"
        "Se a informação não estiver no texto, responda com 'Não encontrado'.\n"
        "Seja conciso e direto.\n\n"
        "TEXTO:\n{contexto}\n\nRESUMO DA RESPOSTA:")
    chain = LLMChain(llm=llm, prompt=prompt_template) # LLMChain retorna dict com 'text'
    resultados_finais = []
    barra_progresso = st.progress(0, text="Iniciando análise de políticas...")
    for i, nome_arquivo in enumerate(_nomes_arquivos):
        dados_contrato_atual = {"arquivo_fonte": nome_arquivo}
        retriever_arquivo_atual = _vector_store.as_retriever(search_kwargs={'filter': {'source': nome_arquivo}, 'k': 4})
        mapa_campos_perguntas = {
            "nome_banco": "Qual o nome do banco ou emissor principal deste contrato?",
            "condicao_limite_credito": "Qual é a política ou condição para definir o limite de crédito?",
            "condicao_juros_rotativo": "Sob quais condições os juros do crédito rotativo são aplicados?",
            "condicao_anuidade": "Qual é a política de cobrança da anuidade descrita no contrato?",
            "condicao_cancelamento": "Quais são as regras para o cancelamento ou rescisão do contrato?"}
        for campo, pergunta in mapa_campos_perguntas.items():
            barra_progresso.progress((i + (list(mapa_campos_perguntas.keys()).index(campo) / len(mapa_campos_perguntas))) / len(_nomes_arquivos),
                                     text=f"Analisando '{campo}' em {nome_arquivo}")
            docs_relevantes = retriever_arquivo_atual.get_relevant_documents(pergunta)
            contexto = "\n\n".join([doc.page_content for doc in docs_relevantes])
            if contexto:
                try:
                    resultado = chain.invoke({"info_desejada": pergunta, "contexto": contexto})
                    resposta = resultado['text'].strip() # Correto para LLMChain
                    dados_contrato_atual[campo] = resposta
                except Exception as e_invoke:
                    st.warning(f"Erro ao invocar LLM para {campo} em {nome_arquivo}: {e_invoke}")
                    dados_contrato_atual[campo] = "Erro na IA"
            else: dados_contrato_atual[campo] = "Contexto não encontrado."
        resultados_finais.append(InfoContrato(**dados_contrato_atual).dict())
    barra_progresso.empty(); st.success("Análise de políticas para dashboard concluída!")
    return resultados_finais

@st.cache_data(show_spinner="Gerando resumo executivo...")
def gerar_resumo_executivo(arquivo_pdf_bytes, nome_arquivo_original):
    if not arquivo_pdf_bytes or not google_api_key: return "Erro: Arquivo ou chave de API não fornecidos."
    with open(nome_arquivo_original, "wb") as f: f.write(arquivo_pdf_bytes)
    loader = PyPDFLoader(nome_arquivo_original); documento_completo_paginas = loader.load(); os.remove(nome_arquivo_original)
    texto_completo = "\n\n".join([page.page_content for page in documento_completo_paginas])
    llm_resumo = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    template_prompt_resumo = PromptTemplate.from_template(
        "Você é um assistente especializado em analisar e resumir documentos jurídicos, como contratos.\n"
        "Com base no texto do contrato fornecido abaixo, crie um resumo executivo em 5 a 7 tópicos concisos (bullet points).\n"
        "Destaque os seguintes aspectos, se presentes: as partes principais envolvidas, o objeto principal do contrato, "
        "prazo de vigência (se houver), principais obrigações financeiras ou condições de pagamento, e as "
        "principais condições ou motivos para rescisão ou cancelamento do contrato.\n"
        "Seja claro e direto.\n\nTEXTO DO CONTRATO:\n{texto_contrato}\n\nRESUMO EXECUTIVO:")
    chain_resumo = LLMChain(llm=llm_resumo, prompt=template_prompt_resumo) # LLMChain retorna dict com 'text'
    try: resultado = chain_resumo.invoke({"texto_contrato": texto_completo}); return resultado['text'] # Correto para LLMChain
    except Exception as e: return f"Erro ao gerar resumo: {e}"

@st.cache_data(show_spinner="Analisando riscos no documento...")
def analisar_documento_para_riscos(texto_completo_doc, nome_arquivo_doc):
    if not texto_completo_doc or not google_api_key: return f"Não foi possível analisar riscos para '{nome_arquivo_doc}': Texto ou Chave API ausente."
    llm_riscos = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
    prompt_riscos_template = PromptTemplate.from_template(
        "Você é um advogado especialista em análise de riscos contratuais. "
        "Analise o texto do contrato fornecido abaixo e identifique cláusulas ou omissões que possam representar riscos significativos. "
        "Para cada risco identificado, por favor:\n1. Descreva o risco de forma clara e concisa.\n"
        "2. Cite o trecho exato da cláusula relevante (ou mencione a ausência de uma cláusula esperada).\n"
        "3. Classifique o risco (ex: Financeiro, Operacional, Legal, Rescisão, Propriedade Intelectual, Confidencialidade, etc.).\n"
        "Concentre-se nos riscos mais impactantes. Se nenhum risco significativo for encontrado, declare isso explicitamente.\n"
        "Use formatação Markdown para sua resposta, com um título para cada risco.\n\n"
        "TEXTO DO CONTRATO ({nome_arquivo}):\n{texto_contrato}\n\nANÁLISE DE RISCOS:")
    chain_riscos = LLMChain(llm=llm_riscos, prompt=prompt_riscos_template) # LLMChain retorna dict com 'text'
    try: resultado = chain_riscos.invoke({"nome_arquivo": nome_arquivo_doc, "texto_contrato": texto_completo_doc}); return resultado['text'] # Correto para LLMChain
    except Exception as e: return f"Erro ao analisar riscos para '{nome_arquivo_doc}': {e}"

# --- FUNÇÃO DE EXTRAÇÃO DE EVENTOS (COM CORREÇÃO) ---
@st.cache_data(show_spinner="Extraindo datas e prazos dos contratos...")
def extrair_eventos_dos_contratos(textos_completos_docs: List[dict]) -> List[dict]:
    if not textos_completos_docs or not google_api_key: return []

    llm_eventos = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    parser = PydanticOutputParser(pydantic_object=ListaDeEventos)

    prompt_eventos_template_str = """Analise o texto do contrato abaixo. Sua tarefa é identificar TODOS os eventos, datas, prazos e períodos importantes mencionados.
Para cada evento encontrado, extraia as seguintes informações:
1.  'descricao_evento': Uma descrição clara e concisa do evento (ex: 'Data de assinatura do contrato', 'Vencimento da primeira parcela', 'Prazo final para entrega do produto', 'Início da vigência', 'Período de carência para alteração de vencimento').
2.  'data_evento_str': A data específica do evento no formato YYYY-MM-DD. Se uma data EXATA não puder ser determinada ou não se aplicar (ex: '10 dias antes do vencimento', 'prazo indeterminado', 'na fatura mensal'), preencha este campo OBRIGATORIAMENTE com a string 'Não Especificado'. NUNCA use null, None ou deixe o campo vazio.
3.  'trecho_relevante': O trecho curto e exato do contrato que menciona este evento/data.

{format_instructions}

TEXTO DO CONTRATO ({arquivo_fonte}):
{texto_contrato}

ATENÇÃO: O campo 'data_evento_str' DEVE SEMPRE ser uma string. Se não houver data específica, use 'Não Especificado'.
LISTA DE EVENTOS ENCONTRADOS:
"""
    prompt_eventos = PromptTemplate(
        template=prompt_eventos_template_str,
        input_variables=["texto_contrato", "arquivo_fonte"],
        partial_variables={"format_instructions": parser.get_format_instructions().replace("```json", "").replace("```", "").strip()}
    )

    output_fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0)) # LLM para o fixing parser
    
    # Cadeia que só roda o LLM, para obter a string bruta primeiro
    chain_eventos_llm_only = prompt_eventos | llm_eventos 

    todos_os_eventos_formatados = []
    barra_progresso = st.progress(0, text="Iniciando extração de datas...")

    for i, doc_info in enumerate(textos_completos_docs):
        nome_arquivo = doc_info["nome"]
        texto_contrato = doc_info["texto"]
        barra_progresso.progress((i + 1) / len(textos_completos_docs), text=f"Analisando datas em: {nome_arquivo}")
        
        try:
            # Etapa 1: Obter a resposta da IA como objeto AIMessage
            resposta_ia_obj = chain_eventos_llm_only.invoke({
                "texto_contrato": texto_contrato,
                "arquivo_fonte": nome_arquivo
            })
            # CORREÇÃO: Acessar o conteúdo da AIMessage
            resposta_ia_str = resposta_ia_obj.content 

            # Etapa 2: Tentar parsear com o PydanticOutputParser
            try:
                resultado_parseado = parser.parse(resposta_ia_str)
            except Exception as e_parse: 
                st.write(f"Parser Pydantic inicial falhou para {nome_arquivo}. Tentando com OutputFixingParser. Erro: {e_parse}")
                st.write(f"Resposta da IA que causou o erro (primeiros 500 chars): {resposta_ia_str[:500]}...")
                resultado_parseado = output_fixing_parser.parse(resposta_ia_str) # OutputFixingParser espera a string
            
            if resultado_parseado and isinstance(resultado_parseado, ListaDeEventos):
                for evento in resultado_parseado.eventos:
                    data_obj = None
                    if evento.data_evento_str and evento.data_evento_str.lower() not in ["não especificado", "condicional", "vide fatura", "n/a", ""]:
                        try: data_obj = datetime.strptime(evento.data_evento_str, "%Y-%m-%d").date()
                        except ValueError:
                            try: data_obj = datetime.strptime(evento.data_evento_str, "%d/%m/%Y").date()
                            except ValueError: pass
                    
                    todos_os_eventos_formatados.append({
                        "Arquivo Fonte": nome_arquivo,
                        "Evento": evento.descricao_evento,
                        "Data Informada": evento.data_evento_str,
                        "Data Objeto": data_obj, 
                        "Trecho Relevante": evento.trecho_relevante
                    })
        except Exception as e_main:
            st.warning(f"Erro crítico ao processar datas para '{nome_arquivo}'. Erro: {e_main}")
            todos_os_eventos_formatados.append({
                "Arquivo Fonte": nome_arquivo, "Evento": f"Falha na extração: {e_main}", 
                "Data Informada": "Erro", "Data Objeto": None, "Trecho Relevante": None
            })
            
    barra_progresso.empty()
    if not todos_os_eventos_formatados: st.info("Nenhum evento ou prazo foi extraído dos documentos.")
    else: st.success("Extração de datas e prazos concluída!")
    return todos_os_eventos_formatados

def formatar_chat_para_markdown(mensagens_chat):
    texto_formatado = "# Histórico da Conversa com Analisador-IA\n\n"
    for mensagem in mensagens_chat:
        if mensagem["role"] == "user": texto_formatado += f"## Você:\n{mensagem['content']}\n\n"
        elif mensagem["role"] == "assistant":
            texto_formatado += f"## IA:\n{mensagem['content']}\n"
            if "sources" in mensagem and mensagem["sources"]:
                texto_formatado += "### Fontes Utilizadas:\n"
                for i, doc_fonte in enumerate(mensagem["sources"]):
                    texto_fonte_original = doc_fonte.page_content; sentenca_chave = mensagem.get("sentenca_chave")
                    texto_fonte_md = texto_fonte_original.replace('\n', '  \n')
                    if sentenca_chave and sentenca_chave in texto_fonte_original:
                        texto_formatado_fonte = texto_fonte_md.replace(sentenca_chave, f"**{sentenca_chave}**")
                    else: texto_formatado_fonte = texto_fonte_md
                    texto_formatado += f"- **Fonte {i+1} (Doc: `{doc_fonte.metadata.get('source', 'N/A')}`, Pág: {doc_fonte.metadata.get('page', 'N/A')})**:\n  > {texto_formatado_fonte[:300]}...\n\n"
            texto_formatado += "---\n\n"
    return texto_formatado

# --- INICIALIZAÇÃO DO OBJETO DE EMBEDDINGS ---
if google_api_key:
    embeddings_global = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
else:
    embeddings_global = None

# --- LAYOUT PRINCIPAL E SIDEBAR ---
st.title("🛡️ Analisador-IA Pro")
st.sidebar.header("Gerenciar Documentos")
modo_documento = st.sidebar.radio("Como carregar os documentos?", ("Fazer novo upload de PDFs", "Carregar coleção existente"), key="modo_doc_radio")
arquivos_pdf_upload_sidebar = None
if modo_documento == "Fazer novo upload de PDFs":
    arquivos_pdf_upload_sidebar = st.sidebar.file_uploader("Selecione um ou mais contratos em PDF", type="pdf", accept_multiple_files=True, key="uploader_sidebar")
    if arquivos_pdf_upload_sidebar:
        if st.sidebar.button("Processar Documentos Carregados", key="btn_proc_upload_sidebar"):
            if google_api_key and embeddings_global:
                st.session_state.vector_store, st.session_state.nomes_arquivos = obter_vector_store_de_uploads(arquivos_pdf_upload_sidebar, embeddings_global)
                st.session_state.arquivos_pdf_originais = arquivos_pdf_upload_sidebar
                st.session_state.colecao_ativa = None; st.session_state.messages = []
                st.session_state.pop('df_dashboard', None); st.session_state.pop('resumo_gerado', None)
                st.session_state.pop('analise_riscos_resultados', None); st.session_state.pop('eventos_contratuais_df', None)
                st.rerun()
            else: st.sidebar.error("Chave de API ou Embeddings não configurados.")
elif modo_documento == "Carregar coleção existente":
    colecoes_disponiveis = listar_colecoes_salvas()
    if colecoes_disponiveis:
        colecao_selecionada = st.sidebar.selectbox("Escolha uma coleção:", colecoes_disponiveis, key="select_colecao_sidebar")
        if st.sidebar.button("Carregar Coleção Selecionada", key="btn_load_colecao_sidebar"):
            if google_api_key and embeddings_global:
                vs, nomes_arqs = carregar_colecao(colecao_selecionada, embeddings_global)
                if vs and nomes_arqs:
                    st.session_state.vector_store, st.session_state.nomes_arquivos, st.session_state.colecao_ativa = vs, nomes_arqs, colecao_selecionada
                    st.session_state.arquivos_pdf_originais = None; st.session_state.messages = []
                    st.session_state.pop('df_dashboard', None); st.session_state.pop('resumo_gerado', None)
                    st.session_state.pop('analise_riscos_resultados', None); st.session_state.pop('eventos_contratuais_df', None)
                    st.rerun()
            else: st.sidebar.error("Chave de API ou Embeddings não configurados.")
    else: st.sidebar.info("Nenhuma coleção salva ainda.")

if "vector_store" in st.session_state and st.session_state.vector_store is not None and st.session_state.get("arquivos_pdf_originais"):
    st.sidebar.markdown("---"); st.sidebar.subheader("Salvar Coleção Atual")
    nome_nova_colecao = st.sidebar.text_input("Nome para a nova coleção:", key="input_nome_colecao_sidebar")
    if st.sidebar.button("Salvar Coleção", key="btn_save_colecao_sidebar"):
        if nome_nova_colecao and st.session_state.nomes_arquivos: salvar_colecao_atual(nome_nova_colecao, st.session_state.vector_store, st.session_state.nomes_arquivos)
        else: st.sidebar.warning("Dê um nome e certifique-se de que há docs carregados.")

if "colecao_ativa" in st.session_state and st.session_state.colecao_ativa: st.sidebar.markdown(f"**Coleção Ativa:** `{st.session_state.colecao_ativa}`")
elif "nomes_arquivos" in st.session_state and st.session_state.nomes_arquivos: st.sidebar.markdown(f"**Arquivos Carregados:** {len(st.session_state.nomes_arquivos)}")

st.sidebar.header("Configurações de Idioma"); idioma_selecionado = st.sidebar.selectbox("Idioma para o CHAT:", ("Português", "Inglês", "Espanhol"), key="idioma_chat_key_sidebar")

# --- INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
if "messages" not in st.session_state: st.session_state.messages = []
if "resumo_gerado" not in st.session_state: st.session_state.resumo_gerado = ""
if "arquivo_resumido" not in st.session_state: st.session_state.arquivo_resumido = None
if "df_dashboard" not in st.session_state: st.session_state.df_dashboard = None
if "analise_riscos_resultados" not in st.session_state: st.session_state.analise_riscos_resultados = {}
if "eventos_contratuais_df" not in st.session_state: st.session_state.eventos_contratuais_df = None

# --- LÓGICA DAS ABAS ---
tab_chat, tab_dashboard, tab_resumo, tab_riscos, tab_prazos = st.tabs(["💬 Chat", "📈 Dashboard", "📜 Resumo", "🚩 Riscos", "🗓️ Prazos"])
documentos_prontos = google_api_key and embeddings_global and (st.session_state.get("vector_store") is not None or st.session_state.get("arquivos_pdf_originais") is not None)

if not documentos_prontos:
    st.warning("Por favor, configure sua Chave de API do Google na barra lateral e carregue documentos para habilitar as funcionalidades.")
else:
    vector_store_global = st.session_state.get("vector_store")
    nomes_arquivos_global = st.session_state.get("nomes_arquivos", [])
    arquivos_pdf_originais_global = st.session_state.get("arquivos_pdf_originais")

    with tab_chat:
        st.header("Converse com seus documentos")
        if not vector_store_global: st.warning("Nenhum documento processado para o chat. Por favor, carregue documentos ou uma coleção.")
        else: 
            template_prompt_chat = PromptTemplate.from_template(
                """Use os seguintes trechos de contexto para responder à pergunta no final.
                INSTRUÇÕES DE FORMATAÇÃO DA RESPOSTA: Sua resposta final deve ter duas partes, separadas por '|||'.
                1. Parte 1: A resposta completa e detalhada para a pergunta do usuário, no idioma {language}.
                2. Parte 2: A citação exata e literal da sentença do contexto que foi mais importante para formular a resposta.
                CONTEXTO: {context}
                PERGUNTA: {question}
                RESPOSTA (seguindo o formato acima):""")
            if not st.session_state.messages : st.session_state.messages.append({"role": "assistant", "content": f"Olá! Documentos da coleção '{st.session_state.get('colecao_ativa', 'atual')}' prontos. Qual sua pergunta?"})
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "sources" in message:
                        with st.expander("Ver Fontes Utilizadas"):
                            for doc_fonte in message["sources"]:
                                texto_fonte = doc_fonte.page_content; sentenca_chave = message.get("sentenca_chave")
                                if sentenca_chave and sentenca_chave in texto_fonte: texto_formatado = texto_fonte.replace(sentenca_chave, f"<span style='background-color: #FFFACD; padding: 2px; border-radius: 3px;'>{sentenca_chave}</span>")
                                else: texto_formatado = texto_fonte
                                st.markdown(f"**Fonte: `{doc_fonte.metadata.get('source', 'N/A')}` (Página {doc_fonte.metadata.get('page', 'N/A')})**")
                                st.markdown(texto_formatado, unsafe_allow_html=True)
            if st.session_state.messages :
                chat_exportado_md = formatar_chat_para_markdown(st.session_state.messages)
                agora = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(label="📥 Exportar Conversa",data=chat_exportado_md, file_name=f"conversa_contratos_{agora}.md", mime="text/markdown", key="export_chat_btn_tab")
                st.markdown("---")
            if prompt := st.chat_input("Faça sua pergunta sobre os contratos..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Pesquisando..."):
                        llm_chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
                        qa_chain = RetrievalQA.from_chain_type(llm=llm_chat, chain_type="stuff", retriever=vector_store_global.as_retriever(search_kwargs={"k": 5}), return_source_documents=True, chain_type_kwargs={"prompt": template_prompt_chat.partial(language=idioma_selecionado)})
                        resultado = qa_chain({"query": prompt}); resposta_bruta = resultado["result"]; fontes = resultado["source_documents"]
                        try: resposta_principal, sentenca_chave = resposta_bruta.split('|||'); sentenca_chave = sentenca_chave.strip()
                        except ValueError: resposta_principal, sentenca_chave = resposta_bruta, None
                        st.markdown(resposta_principal)
                        st.session_state.messages.append({"role": "assistant", "content": resposta_principal, "sources": fontes, "sentenca_chave": sentenca_chave})
                        st.rerun()

    with tab_dashboard:
        st.header("Análise Comparativa de Políticas Contratuais")
        st.markdown("Clique no botão para extrair e comparar as políticas chave dos documentos carregados.")
        if not (vector_store_global and nomes_arquivos_global):
            st.warning("
