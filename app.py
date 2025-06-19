import streamlit as st
import os
import pandas as pd
from typing import Optional, List
import re
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import time
import fitz # PyMuPDF
from PIL import Image
import io
import base64
import tempfile
import zipfile

# --- INTEGRACIÓN CON FIREBASE ---
import firebase_admin
from firebase_admin import credentials, firestore, storage

# --- IMPORTAÇÕES DO GOOGLE E LANGCHAIN ---
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document

# --- INICIALIZAÇÃO UNIFICADA (FIREBASE & GOOGLE AI) ---
@st.cache_resource
def initialize_services():
  """
  Initializes Firebase Admin SDK and Google AI using a single service account.
  Returns db client, storage bucket name, and a configured Google AI client.
  """
  try:
    # 1. Carregar as credenciais do serviço a partir dos segredos do Streamlit
    creds_secrets_obj = st.secrets["firebase_credentials"]
    creds_dict = dict(creds_secrets_obj)

    # 2. Inicializar o Firebase Admin SDK com as credenciais
    bucket_name = st.secrets["firebase_config"]["storageBucket"]
    if bucket_name.startswith("gs://"):
      bucket_name = bucket_name.replace("gs://", "")

    if not firebase_admin._apps:
      cred = credentials.Certificate(creds_dict)
      firebase_admin.initialize_app(cred, {'storageBucket': bucket_name})
    
    db = firestore.client()
    
    # 3. Configurar o cliente Google AI (Gemini) com as mesmas credenciais
    # Isto elimina a necessidade de uma chave de API separada
    genai.configure(credentials=credentials.Certificate(creds_dict))

    st.sidebar.success("Serviços Firebase e Google AI conectados!", icon="✔")
    return db, bucket_name
  except (KeyError, FileNotFoundError):
    st.sidebar.error("Credenciais do Firebase (`firebase_credentials`) não configuradas nos Segredos.")
    st.error("ERRO: As credenciais do Firebase não foram encontradas. Configure o `secrets.toml`.")
    return None, None
  except Exception as e:
    st.sidebar.error(f"Erro ao conectar serviços: {e}")
    st.error(f"ERRO: Falha ao conectar com os serviços Google/Firebase. Detalhes: {e}")
    return None, None

# Inicializar os serviços e obter os clientes
db, BUCKET_NAME = initialize_services()
google_api_key = "used_by_langchain_internally" # Valor fictício, a autenticação é gerida pelo `genai.configure`

# --- SCHEMAS DE DADOS ---
class InfoContrato(BaseModel):
  arquivo_fonte: str = Field(description="O nome do arquivo de origem do contrato.")
  nome_banco_emissor: Optional[str] = Field(default="Não encontrado", description="O nome do banco ou instituição financeira principal mencionada.")
  valor_principal_numerico: Optional[float] = Field(default=None, description="Se houver um valor monetário principal claramente definido no contrato (ex: valor total do contrato, valor do empréstimo, limite de crédito principal), extraia apenas o número. Caso contrário, deixe como não encontrado.")
  prazo_total_meses: Optional[int] = Field(default=None, description="Se houver um prazo de vigência total do contrato claramente definido em meses ou anos, converta para meses e extraia apenas o número. Caso contrário, deixe como não encontrado.")
  taxa_juros_anual_numerica: Optional[float] = Field(default=None, description="Se uma taxa de juros principal (anual ou claramente conversível para anual) for mencionada, extraia apenas o número percentual. Caso contrário, deixe como não encontrado.")
  possui_clausula_rescisao_multa: Optional[str] = Field(default="Não claro", description="O contrato menciona explicitamente uma multa em caso de rescisão? Responda 'Sim', 'Não', ou 'Não claro'.")
  condicao_limite_credito: Optional[str] = Field(default="Não encontrado", description="Resumo da política de como o limite de crédito é definido, analisado e alterado.")
  condicao_juros_rotativo: Optional[str] = Field(default="Não encontrado", description="Resumo da regra de como e quando os juros do crédito rotativo são aplicados.")
  condicao_anuidade: Optional[str] = Field(default="Não encontrado", description="Resumo da política de cobrança da anuidade.")
  condicao_cancelamento: Optional[str] = Field(default="Não encontrado", description="Resumo das condições para cancelamento do contrato.")

class EventoContratual(BaseModel):
  descricao_evento: str = Field(description="Uma descrição clara e concisa do evento ou prazo.")
  data_evento_str: Optional[str] = Field(default="Não Especificado", description="A data do evento no formato yyyy-MM-dd. Se não aplicável, use 'Não Especificado'.")
  trecho_relevante: Optional[str] = Field(default=None, description="O trecho exato do contrato que menciona este evento/data.")

class ListaDeEventos(BaseModel):
  eventos: List[EventoContratual] = Field(description="Lista de eventos contratuais com suas datas.")
  arquivo_fonte: str = Field(description="O nome do arquivo de origem do contrato de onde estes eventos foram extraídos.")


# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(layout="wide", page_title="Analisador-IA ProMax", page_icon="💡")
hide_streamlit_style = "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>"
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- FUNÇÕES DE GERENCIAMENTO DE COLEÇÕES ---

def listar_colecoes_salvas():
  if not db: return []
  try:
    colecoes_ref = db.collection('ia_collections').stream()
    return [doc.id for doc in colecoes_ref]
  except Exception as e:
    st.error(f"Erro ao listar coleções do Firebase: {e}")
    return []

def salvar_colecao_atual(nome_colecao, vector_store_atual, nomes_arquivos_atuais):
  if not db or not BUCKET_NAME:
    st.error("Conexão com Firebase não está disponível.")
    return False
  if not nome_colecao.strip():
    st.error("Por favor, forneça um nome para a coleção.")
    return False
    
  with st.spinner(f"A salvar coleção '{nome_colecao}' no Firebase..."):
    with tempfile.TemporaryDirectory() as temp_dir:
      try:
        # O vector store é salvo em uma subpasta 'faiss_index'
        faiss_path = os.path.join(temp_dir, "faiss_index")
        vector_store_atual.save_local(faiss_path)

        # O manifesto é salvo na raiz do temp_dir
        manifest_path = os.path.join(temp_dir, "manifest.json")
        with open(manifest_path, "w") as f:
          json.dump(nomes_arquivos_atuais, f)
        
        # Cria o arquivo zip
        zip_path_temp = os.path.join(tempfile.gettempdir(), f"{nome_colecao}.zip")
        with zipfile.ZipFile(zip_path_temp, 'w', zipfile.ZIP_DEFLATED) as zipf:
          # Itera sobre todos os arquivos e pastas no diretório temporário
          for root, _, files in os.walk(temp_dir):
            for file in files:
              # --- CORREÇÃO APLICADA AQUI ---
              # Obtenha o caminho completo do arquivo
              caminho_completo = os.path.join(root, file)
              # Crie o caminho relativo para preservar a estrutura de pastas no zip
              caminho_relativo = os.path.relpath(caminho_completo, temp_dir)
              # Escreva no zip usando o caminho relativo como nome do arquivo
              zipf.write(caminho_completo, arcname=caminho_relativo)

        # Faz o upload para o Firebase Storage
        bucket = storage.bucket(BUCKET_NAME)
        blob_path = f"collections/{nome_colecao}.zip"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(zip_path_temp)

        # Salva os metadados no Firestore
        doc_ref = db.collection('ia_collections').document(nome_colecao)
        doc_ref.set({
          'nomes_arquivos': nomes_arquivos_atuais,
          'storage_path': blob_path,
          'created_at': firestore.SERVER_TIMESTAMP
        })

        os.remove(zip_path_temp)
        st.success(f"Coleção '{nome_colecao}' salva com sucesso no Firebase!"); return True
      except Exception as e:
        st.error(f"Erro ao salvar coleção no Firebase: {e}"); return False


# --- FUNÇÕES DE PROCESSAMENTO DE DOCUMENTOS ---
@st.cache_resource(show_spinner="A analisar documentos para busca e chat...")
def obter_vector_store_de_uploads(lista_arquivos_pdf_upload, _embeddings_obj):
  if not lista_arquivos_pdf_upload:
    return None, None

  documentos_totais = []
  nomes_arquivos_processados = []
  llm_vision = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1, request_timeout=300)

  for arquivo_pdf_upload in lista_arquivos_pdf_upload:
    nome_arquivo = arquivo_pdf_upload.name
    st.info(f"A processar ficheiro: {nome_arquivo}...")
    documentos_arquivo_atual = []
    texto_extraido_com_sucesso = False
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
      tmp.write(arquivo_pdf_upload.getbuffer())
      temp_file_path = tmp.name

    try:
      # Tentativa 1: PyPDFLoader
      try:
        st.write(f"A tentar PyPDFLoader para {nome_arquivo}...")
        loader = PyPDFLoader(str(temp_file_path))
        pages_pypdf = loader.load()
        if pages_pypdf:
          texto_pypdf_concatenado = ""
          for page_num_pypdf, page_obj_pypdf in enumerate(pages_pypdf):
            if page_obj_pypdf.page_content and page_obj_pypdf.page_content.strip():
              texto_pypdf_concatenado += page_obj_pypdf.page_content + "\n"
              doc = Document(page_content=page_obj_pypdf.page_content,
                      metadata={"source": nome_arquivo, "page": page_obj_pypdf.metadata.get("page", page_num_pypdf), "method": "pypdf"})
              documentos_arquivo_atual.append(doc)
          
          if len(texto_pypdf_concatenado.strip()) > 100:
            st.success(f"Texto extraído com PyPDFLoader para {nome_arquivo}.")
            texto_extraido_com_sucesso = True
      except Exception as e_pypdf:
        st.warning(f"PyPDFLoader falhou para {nome_arquivo}: {e_pypdf}. A tentar PyMuPDF.")

      # Tentativa 2: PyMuPDF
      if not texto_extraido_com_sucesso:
        try:
          st.write(f"A tentar PyMuPDF (fitz) para {nome_arquivo}...")
          documentos_arquivo_atual = []
          doc_fitz = fitz.open(str(temp_file_path))
          texto_fitz_completo = ""
          for num_pagina, pagina_fitz in enumerate(doc_fitz):
            texto_pagina = pagina_fitz.get_text("text")
            if texto_pagina and texto_pagina.strip():
              texto_fitz_completo += texto_pagina + "\n"
              doc = Document(page_content=texto_pagina, metadata={"source": nome_arquivo, "page": num_pagina, "method": "pymupdf"})
              documentos_arquivo_atual.append(doc)
          doc_fitz.close()
          if len(texto_fitz_completo.strip()) > 100:
            st.success(f"Texto extraído com PyMuPDF para {nome_arquivo}.")
            texto_extraido_com_sucesso = True
        except Exception as e_fitz:
          st.warning(f"PyMuPDF (fitz) falhou para {nome_arquivo}: {e_fitz}. A tentar Gemini Vision.")

      # Tentativa 3: Gemini Vision
      if not texto_extraido_com_sucesso and llm_vision:
        st.write(f"A tentar Gemini Vision para {nome_arquivo}...")
        documentos_arquivo_atual = []
        try:
          arquivo_pdf_upload.seek(0)
          pdf_bytes = arquivo_pdf_upload.read()
          doc_fitz_vision = fitz.open(stream=pdf_bytes, filetype="pdf")
          prompt_gemini_ocr = "..."
          
          for page_num_gemini in range(len(doc_fitz_vision)):
            page_fitz_obj = doc_fitz_vision.load_page(page_num_gemini)
            pix = page_fitz_obj.get_pixmap(dpi=300) 
            img_bytes = pix.tobytes("png")
            base64_image = base64.b64encode(img_bytes).decode('utf-8')
            human_message = HumanMessage(content=[...])
            
            with st.spinner(f"Gemini a processar página {page_num_gemini + 1}/{len(doc_fitz_vision)} de {nome_arquivo}..."):
              ai_msg = llm_vision.invoke([human_message])
            
            if isinstance(ai_msg, AIMessage) and ai_msg.content and isinstance(ai_msg.content, str):
              texto_pagina_gemini = ai_msg.content
              if texto_pagina_gemini.strip():
                doc = Document(...)
                documentos_arquivo_atual.append(doc)
                texto_extraido_com_sucesso = True
            time.sleep(2)

          doc_fitz_vision.close()
          if texto_extraido_com_sucesso:
            st.success(f"Texto extraído com Gemini Vision para {nome_arquivo}.")
        except Exception as e_gemini:
          st.error(f"Erro ao usar Gemini Vision para {nome_arquivo}: {str(e_gemini)[:500]}")
      
      if texto_extraido_com_sucesso and documentos_arquivo_atual:
        documentos_totais.extend(documentos_arquivo_atual)
        nomes_arquivos_processados.append(nome_arquivo)
      else:
        st.error(f"Não foi possível extrair texto do ficheiro: {nome_arquivo}.")

    finally:
      if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

  if not documentos_totais:
    st.error("Nenhum texto pôde ser extraído de nenhum dos documentos fornecidos.")
    return None, []

  try:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs_fragmentados = text_splitter.split_documents(documentos_totais)

    if not docs_fragmentados:
       st.error("A fragmentação do texto não resultou em nenhum documento.")
       return None, nomes_arquivos_processados

    st.info(f"A criar vector store com {len(docs_fragmentados)} fragmentos...")
    vector_store = FAISS.from_documents(docs_fragmentados, _embeddings_obj)
    st.success("Vector store criado com sucesso!")
    return vector_store, nomes_arquivos_processados
  except Exception as e_faiss:
    st.error(f"Erro ao criar o vector store com FAISS: {e_faiss}")
    return None, nomes_arquivos_processados


# O resto do seu código (extrair_dados_dos_contratos, etc.) permanece igual.
# As chamadas a ChatGoogleGenerativeAI e GoogleGenerativeAIEmbeddings
# usarão automaticamente as credenciais configuradas com genai.configure().
@st.cache_data(show_spinner="A extrair dados detalhados dos contratos...")
def extrair_dados_dos_contratos(_vector_store: Optional[FAISS], _nomes_arquivos: list) -> list:
  if not _vector_store or not _nomes_arquivos: return []
  llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
  
  resultados_finais = []
  mapa_campos_para_extracao = {
    "nome_banco_emissor": ("Qual o nome principal do banco, instituição financeira ou empresa emissora deste contrato?", "Responda apenas com o nome. Se não encontrar, diga 'Não encontrado'."),
    "valor_principal_numerico": ("Qual o valor monetário principal ou limite de crédito central deste contrato?", "Se encontrar um valor, forneça apenas o número (ex: 10000.50). Se não encontrar, responda 'Não encontrado'."),
    "prazo_total_meses": ("Qual o prazo de vigência total deste contrato em meses? Se estiver em anos, converta para meses.", "Se encontrar, forneça apenas o número de meses. Se não encontrar, responda 'Não encontrado'."),
    "taxa_juros_anual_numerica": ("Qual a principal taxa de juros anual (ou facilmente conversível para anual) mencionada?", "Se encontrar, forneça apenas o número percentual (ex: 12.5). Se não encontrar, responda 'Não encontrado'."),
    "possui_clausula_rescisao_multa": ("Este contrato menciona explicitamente uma multa monetária ou percentual em caso de rescisão?", "Responda apenas com 'Sim', 'Não', ou 'Não claro'."),
    "condicao_limite_credito": ("Qual é a política ou condição para definir o limite de crédito?", "Resuma a política em uma ou duas frases. Se não encontrar, responda 'Não encontrado'."),
    "condicao_juros_rotativo": ("Sob quais condições os juros do crédito rotativo são aplicados?", "Resuma a regra em uma ou duas frases. Se não encontrar, responda 'Não encontrado'."),
    "condicao_anuidade": ("Qual é a política de cobrança da anuidade descrita no contrato?", "Resuma a política em uma ou duas frases. Se não encontrar, responda 'Não encontrado'."),
    "condicao_cancelamento": ("Quais são as regras para o cancelamento ou rescisão do contrato?", "Resuma as regras em uma ou duas frases. Se não encontrar, responda 'Não encontrado'.")
  }
  
  total_campos_a_extrair = len(mapa_campos_para_extracao)
  total_operacoes = len(_nomes_arquivos) * total_campos_a_extrair
  operacao_atual = 0

  barra_progresso_placeholder = st.empty()

  for nome_arquivo in _nomes_arquivos:
    dados_contrato_atual = {"arquivo_fonte": nome_arquivo}
    retriever_arquivo_atual = _vector_store.as_retriever(
      search_type="similarity",
      search_kwargs={'filter': {'source': nome_arquivo}, 'k': 5}
    )
    
    for campo, (pergunta_chave, instrucao_adicional) in mapa_campos_para_extracao.items():
      operacao_atual += 1
      progress_value = operacao_atual / total_operacoes if total_operacoes > 0 else 0
      barra_progresso_placeholder.progress(progress_value,
                     text=f"A extrair '{campo}' de {nome_arquivo} ({operacao_atual}/{total_operacoes})")
      
      docs_relevantes = retriever_arquivo_atual.get_relevant_documents(pergunta_chave + " " + instrucao_adicional)
      contexto = "\n\n---\n\n".join([f"Trecho do documento '{doc.metadata.get('source', 'N/A')}' (página {doc.metadata.get('page', 'N/A')}):\n{doc.page_content}" for doc in docs_relevantes])

      prompt_extracao = PromptTemplate.from_template(
        "Com base no contexto fornecido, responda à seguinte pergunta de forma precisa. {instrucao_adicional}\n\n"
        "Contexto:\n{contexto}\n\n"
        "Pergunta: {pergunta}\n"
        "Resposta:"
      )
      chain_extracao = LLMChain(llm=llm, prompt=prompt_extracao)
      
      if contexto.strip():
        try:
          resultado = chain_extracao.invoke({
            "instrucao_adicional": instrucao_adicional,
            "contexto": contexto, 
            "pergunta": pergunta_chave
          })
          resposta = resultado['text'].strip()

          if campo in ["valor_principal_numerico", "prazo_total_meses", "taxa_juros_anual_numerica"]:
            numeros = re.findall(r"[\d,.]+", resposta)
            if numeros:
              try:
                valor_str = numeros[0].replace('.', '').replace(',', '.')
                if campo == "prazo_total_meses":
                  dados_contrato_atual[campo] = int(float(valor_str))
                else:
                  dados_contrato_atual[campo] = float(valor_str)
              except ValueError: dados_contrato_atual[campo] = None
            else: dados_contrato_atual[campo] = None
          elif campo == "possui_clausula_rescisao_multa":
            if "sim" in resposta.lower(): dados_contrato_atual[campo] = "Sim"
            elif "não" in resposta.lower(): dados_contrato_atual[campo] = "Não"
            else: dados_contrato_atual[campo] = "Não claro"
          else:
            dados_contrato_atual[campo] = resposta if "não encontrado" not in resposta.lower() else "Não encontrado"
        except Exception as e_invoke:
          st.warning(f"Erro ao invocar LLM para '{campo}' em {nome_arquivo}: {e_invoke}")
          dados_contrato_atual[campo] = "Erro na IA"
      else:
        st.warning(f"Contexto não encontrado para '{campo}' em {nome_arquivo} após busca no vector store.")
        dados_contrato_atual[campo] = "Contexto não encontrado"
      
      time.sleep(1)

    try:
      info_validada = InfoContrato(**dados_contrato_atual)
      resultados_finais.append(info_validada.model_dump())
    except Exception as e_pydantic:
      st.error(f"Erro de validação Pydantic para {nome_arquivo}: {e_pydantic}")
      
  barra_progresso_placeholder.empty()
  st.success("Extração detalhada concluída!")
  return resultados_finais


def detectar_anomalias_no_dataframe(df: pd.DataFrame) -> List[str]:
  # ... (seu código permanece o mesmo)
  return []


@st.cache_data(show_spinner="A gerar resumo executivo...")
def gerar_resumo_executivo(arquivo_pdf_bytes, nome_arquivo_original):
  # ... (seu código permanece o mesmo)
  return ""


# --- INICIALIZAÇÃO DO OBJETO DE EMBEDDINGS ---
# Esta verificação é agora mais simples
if BUCKET_NAME: # Se a inicialização dos serviços funcionou
  try:
    embeddings_global = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
  except Exception as e_embed:
    st.sidebar.error(f"Erro ao inicializar embeddings: {e_embed}")
    embeddings_global = None
else:
  embeddings_global = None

# --- LAYOUT PRINCIPAL E SIDEBAR ---
# ... (o resto da sua UI permanece o mesmo)
st.title("💡 Analisador-IA ProMax")
st.sidebar.image("https://i.imgur.com/aozL2jD.png", width=100) # Exemplo de logo
st.sidebar.header("Gerir Documentos")

if db:
  modo_documento = st.sidebar.radio("Como carregar os documentos?", ("Fazer novo upload de PDFs", "Carregar coleção do Firebase"), key="modo_doc_radio_v3", index=0)
  arquivos_pdf_upload_sidebar = None

  if modo_documento == "Fazer novo upload de PDFs":
    arquivos_pdf_upload_sidebar = st.sidebar.file_uploader("Selecione um ou mais contratos em PDF", type="pdf", accept_multiple_files=True, key="uploader_sidebar_v3")
    if arquivos_pdf_upload_sidebar:
      if st.sidebar.button("Processar Documentos Carregados", key="btn_proc_upload_sidebar_v3", use_container_width=True):
        if embeddings_global:
          with st.spinner("A processar e indexar documentos..."):
            vs, nomes_arqs = obter_vector_store_de_uploads(arquivos_pdf_upload_sidebar, embeddings_global)
          
          if vs and nomes_arqs:
            st.session_state.vector_store = vs
            st.session_state.nomes_arquivos = nomes_arqs
            st.session_state.arquivos_pdf_originais = arquivos_pdf_upload_sidebar
            st.session_state.colecao_ativa = None
            st.session_state.messages = []
            st.success(f"{len(nomes_arqs)} Documento(s) processado(s)!")
            st.rerun()
          else:
            st.error("Falha ao processar documentos.")
        else: st.sidebar.error("Embeddings não configurados.")

  elif modo_documento == "Carregar coleção do Firebase":
    colecoes_disponiveis = listar_colecoes_salvas()
    if colecoes_disponiveis:
      colecao_selecionada = st.sidebar.selectbox("Escolha uma coleção:", colecoes_disponiveis, key="select_colecao_sidebar_v3", index=None)
      if colecao_selecionada and st.sidebar.button("Carregar Coleção", key="btn_load_colecao_sidebar_v3", use_container_width=True):
        if embeddings_global:
          vs, nomes_arqs = carregar_colecao(colecao_selecionada, embeddings_global)
          if vs and nomes_arqs:
            st.session_state.vector_store = vs
            st.session_state.nomes_arquivos = nomes_arqs
            st.session_state.colecao_ativa = colecao_selecionada
            st.session_state.arquivos_pdf_originais = None
            st.session_state.messages = []
            st.rerun()
        else: st.sidebar.error("Embeddings não configurados.")
    else: st.sidebar.info("Nenhuma coleção salva.")

  if st.session_state.get("vector_store") and st.session_state.get("arquivos_pdf_originais"):
    st.sidebar.markdown("---")
    nome_nova_colecao = st.sidebar.text_input("Nome para a nova coleção:")
    if st.sidebar.button("Salvar Coleção no Firebase", key="btn_save_colecao_sidebar_v3", use_container_width=True):
      if nome_nova_colecao and st.session_state.nomes_arquivos:
        salvar_colecao_atual(nome_nova_colecao, st.session_state.vector_store, st.session_state.nomes_arquivos)
      else: st.sidebar.warning("Dê um nome e certifique-se de que há docs carregados.")

if "colecao_ativa" in st.session_state and st.session_state.colecao_ativa:
  st.sidebar.markdown(f"**🔥 Coleção Ativa:** `{st.session_state.colecao_ativa}`")
elif "nomes_arquivos" in st.session_state and st.session_state.nomes_arquivos:
  st.sidebar.markdown(f"**📄 Ficheiros em Memória:** {len(st.session_state.nomes_arquivos)}")

st.sidebar.header("Configurações de Idioma")
idioma_selecionado = st.sidebar.selectbox("Idioma para o CHAT:", ("Português", "Inglês", "Espanhol"), key="idioma_chat_key_sidebar_v3")

# --- INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
if "messages" not in st.session_state:
  st.session_state.messages = []

# --- LÓGICA DAS ABAS ---
tab_chat, tab_dashboard, tab_resumo, tab_riscos, tab_prazos, tab_conformidade, tab_anomalias = st.tabs([
  "💬 Chat", "📈 Dashboard", "📜 Resumo", "🚩 Riscos", "🗓️ Prazos", "⚖️ Conformidade", "📊 Anomalias"
])

documentos_prontos = embeddings_global and st.session_state.get("vector_store") and st.session_state.get("nomes_arquivos")

if not documentos_prontos:
  st.info("👈 Por favor, carregue e processe documentos PDF ou uma coleção existente na barra lateral para habilitar as funcionalidades.")
else:
  vector_store_global = st.session_state.get("vector_store")
  nomes_arquivos_global = st.session_state.get("nomes_arquivos", [])
  arquivos_pdf_originais_global = st.session_state.get("arquivos_pdf_originais")

  with tab_chat:
    st.header("Converse com os seus documentos")
    if not st.session_state.messages : 
      st.session_state.messages.append({"role": "assistant", "content": f"Olá! Documentos prontos. Qual a sua pergunta?"})
    
    for message in st.session_state.messages:
      with st.chat_message(message["role"]):
        st.markdown(message["content"])

    if prompt := st.chat_input("Faça a sua pergunta..."):
      st.session_state.messages.append({"role": "user", "content": prompt})
      with st.chat_message("user"): st.markdown(prompt)
      
      with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("A pensar..."):
          llm_chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
          qa_chain = RetrievalQA.from_chain_type(
            llm=llm_chat, 
            chain_type="stuff", 
            retriever=vector_store_global.as_retriever(), 
            return_source_documents=True
          )
          try:
            resultado = qa_chain.invoke({"query": prompt})
            message_placeholder.markdown(resultado['result'])
            st.session_state.messages.append({"role": "assistant", "content": resultado['result']})
          except Exception as e_chat:
            st.error(f"Erro na cadeia de QA: {e_chat}")

  # ... (o resto da UI das abas permanece o mesmo)
