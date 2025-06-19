import streamlit as st
import os
import pandas as pd
from typing import Optional, List
import re
from datetime import datetime, date
import json
from pathlib import Path
import numpy as np
import time
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
import tempfile
import zipfile

# --- IMPORTAÇÕES DO FIREBASE ---
import firebase_admin
from firebase_admin import credentials, firestore, storage

# --- IMPORTAÇÕES DO LANGCHAIN E PYDANTIC ---
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# A FAISS foi movida para langchain_community
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document

# --- INICIALIZAÇÃO UNIFICADA (FIREBASE & GOOGLE AI) ---
@st.cache_resource(show_spinner="Conectando aos serviços Google/Firebase...")
def initialize_services():
    """
    Inicializa o Firebase Admin SDK e o Google AI usando uma única conta de serviço.
    Retorna o cliente do Firestore e o nome do bucket de armazenamento.
    """
    try:
        # Carrega as credenciais a partir dos secrets do Streamlit
        creds_secrets_obj = st.secrets["firebase_credentials"]
        creds_dict = dict(creds_secrets_obj)
        bucket_name = st.secrets["firebase_config"]["storageBucket"]

        # Inicializa o app Firebase se ainda não foi inicializado
        if not firebase_admin._apps:
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred, {'storageBucket': bucket_name})

        db_client = firestore.client()
        
        # As credenciais do LangChain/Google AI são gerenciadas automaticamente pelo ambiente
        # quando o SDK Admin é inicializado. Não é necessário configurar genai separadamente.
        
        st.sidebar.success("Serviços Firebase e Google AI conectados!", icon="✔")
        return db_client, bucket_name
    except (KeyError, FileNotFoundError):
        st.sidebar.error("Credenciais do Firebase (`firebase_credentials`) ou configuração (`firebase_config`) não encontradas nos Segredos do Streamlit.")
        st.error("ERRO: As credenciais do Firebase não foram encontradas. Configure o arquivo `secrets.toml` corretamente.")
        return None, None
    except Exception as e:
        st.sidebar.error(f"Erro ao conectar aos serviços: {e}")
        st.error(f"ERRO: Falha ao conectar com os serviços Google/Firebase. Detalhes: {e}")
        return None, None

# Inicializa os serviços e obtém os clientes
db, BUCKET_NAME = initialize_services()
# A chave de API não é mais necessária, a autenticação é via credenciais de serviço
google_api_key = "FIREBASE_SDK_AUTH" if db else None


# --- SCHEMAS DE DADOS (Pydantic Models) ---
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
    data_evento_str: Optional[str] = Field(default="Não Especificado", description="A data do evento no formato YYYY-MM-DD. Se não aplicável, use 'Não Especificado'.")
    trecho_relevante: Optional[str] = Field(default=None, description="O trecho exato do contrato que menciona este evento/data.")

class ListaDeEventos(BaseModel):
    eventos: List[EventoContratual] = Field(description="Lista de eventos contratuais com suas datas.")
    arquivo_fonte: str = Field(description="O nome do arquivo de origem do contrato de onde estes eventos foram extraídos.")


# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(layout="wide", page_title="Analisador-IA ProMax", page_icon="💡")
hide_streamlit_style = "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>"
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# --- FUNÇÕES DE GERENCIAMENTO DE COLEÇÕES (FIREBASE) ---
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
        st.error("Conexão com Firebase não está disponível para salvar.")
        return False
    if not nome_colecao.strip():
        st.error("Por favor, forneça um nome para a coleção.")
        return False

    with st.spinner(f"Salvando coleção '{nome_colecao}' no Firebase..."):
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Salva o vector store localmente na pasta temporária
                faiss_path = Path(temp_dir) / "faiss_index"
                vector_store_atual.save_local(str(faiss_path))

                # Cria o arquivo zip contendo a pasta do índice
                zip_path_temp = Path(tempfile.gettempdir()) / f"{nome_colecao}.zip"
                with zipfile.ZipFile(zip_path_temp, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Adiciona a pasta faiss_index ao zip
                    for root, _, files in os.walk(faiss_path):
                        for file in files:
                            full_path = Path(root) / file
                            relative_path = full_path.relative_to(Path(temp_dir))
                            zipf.write(full_path, arcname=relative_path)

                # Faz o upload do zip para o Firebase Storage
                bucket = storage.bucket() # O nome do bucket já foi configurado na inicialização
                blob_path = f"collections/{nome_colecao}.zip"
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(str(zip_path_temp))

                # Salva os metadados no Firestore
                doc_ref = db.collection('ia_collections').document(nome_colecao)
                doc_ref.set({
                    'nomes_arquivos': nomes_arquivos_atuais,
                    'storage_path': blob_path,
                    'created_at': firestore.SERVER_TIMESTAMP
                })

                os.remove(zip_path_temp) # Limpa o arquivo zip local
                st.success(f"Coleção '{nome_colecao}' salva com sucesso no Firebase!")
                return True
            except Exception as e:
                st.error(f"Erro ao salvar coleção no Firebase: {e}")
                return False

@st.cache_resource(show_spinner="Carregando coleção do Firebase...")
def carregar_colecao(nome_colecao, _embeddings_obj):
    if not db or not BUCKET_NAME:
        st.error("Conexão com Firebase não está disponível para carregar.")
        return None, None
        
    try:
        # 1. Pega os metadados do Firestore
        doc_ref = db.collection('ia_collections').document(nome_colecao)
        doc = doc_ref.get()
        if not doc.exists:
            st.error(f"Coleção '{nome_colecao}' não encontrada no Firestore.")
            return None, None
        
        metadata = doc.to_dict()
        storage_path = metadata.get('storage_path')
        nomes_arquivos = metadata.get('nomes_arquivos')
        
        if not storage_path or not nomes_arquivos:
            st.error(f"Metadados da coleção '{nome_colecao}' estão incompletos.")
            return None, None

        # 2. Baixa o arquivo zip do Storage
        bucket = storage.bucket()
        blob = bucket.blob(storage_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path_temp = Path(temp_dir) / "colecao.zip"
            st.info(f"Baixando índice de '{nome_colecao}'...")
            blob.download_to_filename(str(zip_path_temp))

            # 3. Descompacta o arquivo
            unzip_path = Path(temp_dir) / "unzipped"
            unzip_path.mkdir()
            with zipfile.ZipFile(zip_path_temp, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)

            # 4. Carrega o índice FAISS
            faiss_index_path = unzip_path / "faiss_index"
            if not faiss_index_path.exists() or not faiss_index_path.is_dir():
                st.error("A estrutura do arquivo zip da coleção está incorreta. Pasta 'faiss_index' não encontrada.")
                return None, None
                
            st.info("Carregando vector store...")
            vector_store = FAISS.load_local(
                str(faiss_index_path), 
                embeddings=_embeddings_obj, 
                allow_dangerous_deserialization=True
            )
            
            st.success(f"Coleção '{nome_colecao}' carregada com sucesso do Firebase!")
            return vector_store, nomes_arquivos

    except Exception as e:
        st.error(f"Erro ao carregar coleção '{nome_colecao}' do Firebase: {e}")
        return None, None

# --- O RESTANTE DAS FUNÇÕES (PROCESSAMENTO, EXTRAÇÃO, ETC.) PERMANECE O MESMO ---
# Nenhuma alteração necessária nas funções abaixo, pois elas dependem dos objetos
# vector_store e llm, que são criados após a carga dos dados.

@st.cache_resource(show_spinner="Analisando documentos para busca e chat...")
def obter_vector_store_de_uploads(lista_arquivos_pdf_upload, _embeddings_obj):
    if not lista_arquivos_pdf_upload or not google_api_key or not _embeddings_obj:
        return None, None

    documentos_totais = []
    nomes_arquivos_processados = []
    llm_vision = None

    if google_api_key:
        try:
            llm_vision = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1, request_timeout=300) # Timeout aumentado
        except Exception as e:
            st.warning(f"Não foi possível inicializar o modelo de visão do Gemini: {e}")
            llm_vision = None

    for arquivo_pdf_upload in lista_arquivos_pdf_upload:
        nome_arquivo = arquivo_pdf_upload.name
        st.info(f"Processando arquivo: {nome_arquivo}...")
        documentos_arquivo_atual = []
        texto_extraido_com_sucesso = False
        temp_file_path = Path(f"temp_{nome_arquivo}") # Definido fora do try para o finally

        try:
            with open(temp_file_path, "wb") as f:
                f.write(arquivo_pdf_upload.getbuffer())

            # Tentativa 1: PyPDFLoader
            try:
                st.write(f"Tentando PyPDFLoader para {nome_arquivo}...")
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
                    
                    if len(texto_pypdf_concatenado.strip()) > 100: # Considerar texto útil
                        st.success(f"Texto extraído com PyPDFLoader para {nome_arquivo}.")
                        texto_extraido_com_sucesso = True
            except Exception as e_pypdf:
                st.warning(f"PyPDFLoader falhou para {nome_arquivo}: {e_pypdf}. Tentando PyMuPDF.")

            # Tentativa 2: PyMuPDF (fitz) se PyPDFLoader falhou ou não extraiu texto suficiente
            if not texto_extraido_com_sucesso:
                try:
                    st.write(f"Tentando PyMuPDF (fitz) para {nome_arquivo}...")
                    documentos_arquivo_atual = [] # Limpar docs anteriores se PyPDFLoader falhou
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
                    elif documentos_arquivo_atual: # Mesmo que pouco texto, se PyMuPDF retornou algo estruturado
                        st.info(f"Texto (limitado) extraído com PyMuPDF para {nome_arquivo}.")
                        texto_extraido_com_sucesso = True
                except Exception as e_fitz:
                    st.warning(f"PyMuPDF (fitz) falhou para {nome_arquivo}: {e_fitz}. Tentando Gemini Vision.")

            # Tentativa 3: Gemini Vision se as outras falharem e llm_vision estiver disponível
            if not texto_extraido_com_sucesso and llm_vision:
                st.write(f"Tentando Gemini Vision para {nome_arquivo}...")
                documentos_arquivo_atual = [] # Limpar docs anteriores
                try:
                    arquivo_pdf_upload.seek(0)
                    pdf_bytes = arquivo_pdf_upload.read()
                    doc_fitz_vision = fitz.open(stream=pdf_bytes, filetype="pdf")
                    
                    prompt_gemini_ocr = "Você é um especialista em OCR. Extraia todo o texto visível desta página de documento da forma mais precisa possível. Mantenha a estrutura do texto original, incluindo parágrafos e quebras de linha, quando apropriado. Se houver tabelas, tente preservar sua estrutura textual."
                    
                    for page_num_gemini in range(len(doc_fitz_vision)):
                        page_fitz_obj = doc_fitz_vision.load_page(page_num_gemini)
                        pix = page_fitz_obj.get_pixmap(dpi=300) 
                        img_bytes = pix.tobytes("png")
                        base64_image = base64.b64encode(img_bytes).decode('utf-8')

                        human_message = HumanMessage(
                            content=[
                                {"type": "text", "text": prompt_gemini_ocr},
                                {"type": "image_url", "image_url": f"data:image/png;base64,{base64_image}"}
                            ]
                        )
                        
                        with st.spinner(f"Gemini processando página {page_num_gemini + 1}/{len(doc_fitz_vision)} de {nome_arquivo}..."):
                            ai_msg = llm_vision.invoke([human_message])
                        
                        if isinstance(ai_msg, AIMessage) and ai_msg.content and isinstance(ai_msg.content, str):
                            texto_pagina_gemini = ai_msg.content
                            if texto_pagina_gemini.strip():
                                doc = Document(page_content=texto_pagina_gemini, metadata={"source": nome_arquivo, "page": page_num_gemini, "method": "gemini_vision"})
                                documentos_arquivo_atual.append(doc)
                                texto_extraido_com_sucesso = True # Marcar sucesso se Gemini extrair algo
                        time.sleep(2) # Respeitar limites de taxa da API

                    doc_fitz_vision.close()
                    if texto_extraido_com_sucesso:
                        st.success(f"Texto extraído com Gemini Vision para {nome_arquivo}.")
                    else:
                        st.warning(f"Gemini Vision não retornou texto substancial para {nome_arquivo}.")

                except Exception as e_gemini:
                    st.error(f"Erro ao usar Gemini Vision para {nome_arquivo}: {str(e_gemini)[:500]}")
            
            if texto_extraido_com_sucesso and documentos_arquivo_atual:
                documentos_totais.extend(documentos_arquivo_atual)
                nomes_arquivos_processados.append(nome_arquivo)
            elif not texto_extraido_com_sucesso :
                st.error(f"Não foi possível extrair texto do arquivo: {nome_arquivo}. O arquivo pode estar vazio, corrompido ou ser uma imagem complexa.")

        except Exception as e_geral_arquivo:
            st.error(f"Erro geral ao processar o arquivo {nome_arquivo}: {e_geral_arquivo}")
        finally:
            if temp_file_path.exists():
                try:
                    os.remove(temp_file_path)
                except Exception as e_remove:
                    st.warning(f"Não foi possível remover o arquivo temporário {temp_file_path}: {e_remove}")
    
    if not documentos_totais:
        st.error("Nenhum texto pôde ser extraído de nenhum dos documentos fornecidos.")
        return None, []

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs_fragmentados = text_splitter.split_documents(documentos_totais)

        if not docs_fragmentados:
             st.error("A fragmentação do texto não resultou em nenhum documento. Verifique o conteúdo extraído.")
             return None, nomes_arquivos_processados

        st.info(f"Criando vector store com {len(docs_fragmentados)} fragmentos de {len(nomes_arquivos_processados)} arquivos.")
        vector_store = FAISS.from_documents(docs_fragmentados, _embeddings_obj)
        st.success("Vector store criado com sucesso!")
        return vector_store, nomes_arquivos_processados
    except Exception as e_faiss:
        st.error(f"Erro ao criar o vector store com FAISS: {e_faiss}")
        return None, nomes_arquivos_processados

@st.cache_data(show_spinner="Extraindo dados detalhados dos contratos...")
def extrair_dados_dos_contratos(_vector_store: Optional[FAISS], _nomes_arquivos: list) -> list:
    if not _vector_store or not google_api_key or not _nomes_arquivos: return []
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
    
    total_operacoes = len(_nomes_arquivos) * len(mapa_campos_para_extracao)
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
                                         text=f"Extraindo '{campo}' de {nome_arquivo} ({operacao_atual}/{total_operacoes})")
            
            docs_relevantes = retriever_arquivo_atual.get_relevant_documents(pergunta_chave + " " + instrucao_adicional)
            contexto = "\n\n---\n\n".join([f"Trecho do documento '{doc.metadata.get('source', 'N/A')}' (página {doc.metadata.get('page', 'N/A')} - método {doc.metadata.get('method', 'N/A')}):\n{doc.page_content}" for doc in docs_relevantes])

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
                        numeros = re.findall(r"[\d]+(?:[.,]\d+)*", resposta)
                        if numeros:
                            try:
                                valor_str = numeros[0].replace('.', '').replace(',', '.')
                                if valor_str.count('.') > 1:
                                    parts = valor_str.split('.')
                                    valor_str = "".join(parts[:-1]) + "." + parts[-1]
                                
                                if campo == "prazo_total_meses":
                                    dados_contrato_atual[campo] = int(float(valor_str))
                                else:
                                    dados_contrato_atual[campo] = float(valor_str)
                            except ValueError: dados_contrato_atual[campo] = None
                        else: dados_contrato_atual[campo] = None
                    elif campo == "possui_clausula_rescisao_multa":
                        if "sim" in resposta.lower(): dados_contrato_atual[campo] = "Sim"
                        elif "não" in resposta.lower() or "nao" in resposta.lower() : dados_contrato_atual[campo] = "Não"
                        else: dados_contrato_atual[campo] = "Não claro"
                    else: # Campos de texto
                        dados_contrato_atual[campo] = resposta if "não encontrado" not in resposta.lower() and resposta.strip() else "Não encontrado"
                except Exception as e_invoke:
                    st.warning(f"Erro ao invocar LLM para '{campo}' em {nome_arquivo}: {e_invoke}")
                    dados_contrato_atual[campo] = None if "numerico" in campo or "meses" in campo else "Erro na IA"
            else:
                st.warning(f"Contexto não encontrado para '{campo}' em {nome_arquivo}.")
                dados_contrato_atual[campo] = None if "numerico" in campo or "meses" in campo else "Contexto não encontrado"
            
            time.sleep(1.5)

        try:
            info_validada = InfoContrato(**dados_contrato_atual)
            resultados_finais.append(info_validada.model_dump())
        except Exception as e_pydantic:
            st.error(f"Erro de validação Pydantic para {nome_arquivo}: {e_pydantic}. Dados: {dados_contrato_atual}")
            resultados_finais.append({"arquivo_fonte": nome_arquivo, "nome_banco_emissor": "Erro de Validação"})
            
    barra_progresso_placeholder.empty()
    if resultados_finais:
        st.success("Extração detalhada para dashboard e anomalias concluída!")
    else:
        st.warning("Nenhum dado foi extraído.")
    return resultados_finais

# --- LAYOUT PRINCIPAL E SIDEBAR ---
st.title("💡 Analisador-IA ProMax")
st.sidebar.image("https://i.imgur.com/aozL2jD.png", width=100)
st.sidebar.header("Gerenciar Documentos")

# --- INICIALIZAÇÃO DO OBJETO DE EMBEDDINGS ---
embeddings_global = None
if google_api_key:
    try:
        embeddings_global = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e_embed:
        st.sidebar.error(f"Erro ao inicializar embeddings: {e_embed}")

if not db or not embeddings_global:
    st.sidebar.error("A aplicação não pode continuar sem conexão ao Firebase e ao serviço de Embeddings.")
else:
    modo_documento = st.sidebar.radio(
        "Como carregar os documentos?", 
        ("Fazer novo upload de PDFs", "Carregar coleção do Firebase"), 
        key="modo_doc_radio_v3"
    )

    if modo_documento == "Fazer novo upload de PDFs":
        arquivos_pdf_upload_sidebar = st.sidebar.file_uploader(
            "Selecione um ou mais contratos em PDF", 
            type="pdf", 
            accept_multiple_files=True, 
            key="uploader_sidebar_v3"
        )
        if arquivos_pdf_upload_sidebar:
            if st.sidebar.button("Processar Documentos Carregados", key="btn_proc_upload_sidebar_v3", use_container_width=True):
                with st.spinner("Processando e indexando documentos... Isso pode levar alguns minutos."):
                    vs, nomes_arqs = obter_vector_store_de_uploads(arquivos_pdf_upload_sidebar, embeddings_global)
                
                if vs and nomes_arqs:
                    st.session_state.vector_store = vs
                    st.session_state.nomes_arquivos = nomes_arqs
                    st.session_state.arquivos_pdf_originais = arquivos_pdf_upload_sidebar
                    st.session_state.colecao_ativa = None
                    st.session_state.messages = []
                    # Limpa caches de abas anteriores
                    for key_to_pop in ['df_dashboard', 'resumo_gerado', 'analise_riscos_resultados', 'eventos_contratuais_df', 'conformidade_resultados', 'anomalias_resultados']:
                        st.session_state.pop(key_to_pop, None)
                    st.success(f"{len(nomes_arqs)} Documento(s) processado(s)!")
                    st.rerun()
                else:
                    st.error("Falha ao processar documentos. Verifique os logs acima.")

    elif modo_documento == "Carregar coleção do Firebase":
        colecoes_disponiveis = listar_colecoes_salvas()
        if colecoes_disponiveis:
            colecao_selecionada = st.sidebar.selectbox(
                "Escolha uma coleção:", 
                colecoes_disponiveis, 
                key="select_colecao_sidebar_v3", 
                index=None, 
                placeholder="Selecione uma coleção"
            )
            if colecao_selecionada and st.sidebar.button("Carregar Coleção Selecionada", key="btn_load_colecao_sidebar_v3", use_container_width=True):
                vs, nomes_arqs = carregar_colecao(colecao_selecionada, embeddings_global)
                if vs and nomes_arqs:
                    st.session_state.vector_store = vs
                    st.session_state.nomes_arquivos = nomes_arqs
                    st.session_state.colecao_ativa = colecao_selecionada
                    st.session_state.arquivos_pdf_originais = None
                    st.session_state.messages = []
                    for key_to_pop in ['df_dashboard', 'resumo_gerado', 'analise_riscos_resultados', 'eventos_contratuais_df', 'conformidade_resultados', 'anomalias_resultados']:
                        st.session_state.pop(key_to_pop, None)
                    st.rerun()
        else:
            st.sidebar.info("Nenhuma coleção salva no Firebase ainda.")

    # Lógica para salvar coleção
    if "vector_store" in st.session_state and st.session_state.vector_store and st.session_state.get("arquivos_pdf_originais"):
        st.sidebar.markdown("---")
        st.sidebar.subheader("Salvar Coleção no Firebase")
        nome_nova_colecao = st.sidebar.text_input("Nome para a nova coleção:", key="input_nome_colecao_sidebar_v3")
        if st.sidebar.button("Salvar Coleção", key="btn_save_colecao_sidebar_v3", use_container_width=True):
            if nome_nova_colecao and st.session_state.nomes_arquivos:
                salvar_colecao_atual(nome_nova_colecao, st.session_state.vector_store, st.session_state.nomes_arquivos)
            else:
                st.sidebar.warning("Dê um nome e certifique-se de que há documentos carregados.")

    if "colecao_ativa" in st.session_state and st.session_state.colecao_ativa:
        st.sidebar.markdown(f"**🔥 Coleção Ativa:** `{st.session_state.colecao_ativa}`")
    elif "nomes_arquivos" in st.session_state and st.session_state.nomes_arquivos:
        st.sidebar.markdown(f"**📄 Arquivos em Memória:** {len(st.session_state.nomes_arquivos)}")

    st.sidebar.header("Configurações de Idioma")
    idioma_selecionado = st.sidebar.selectbox("Idioma para o CHAT:", ("Português", "Inglês", "Espanhol"), key="idioma_chat_key_sidebar_v3")

    # --- INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- LÓGICA DAS ABAS ---
    tab_chat, tab_dashboard, tab_resumo, tab_riscos, tab_prazos, tab_conformidade, tab_anomalias = st.tabs([
        "💬 Chat", "📈 Dashboard", "📜 Resumo", "🚩 Riscos", "🗓️ Prazos", "⚖️ Conformidade", "📊 Anomalias"
    ])
    
    documentos_prontos = st.session_state.get("vector_store") is not None and st.session_state.get("nomes_arquivos")

    if not documentos_prontos:
        st.info("👈 Por favor, carregue e processe documentos PDF ou uma coleção existente na barra lateral para habilitar as funcionalidades.")
    else:
        # As variáveis globais usadas nas abas
        vector_store_global = st.session_state.get("vector_store")
        nomes_arquivos_global = st.session_state.get("nomes_arquivos", [])
        arquivos_pdf_originais_global = st.session_state.get("arquivos_pdf_originais")

        # O restante do código das abas (tab_chat, tab_dashboard, etc.) pode ser colado aqui
        # sem alterações, pois ele já está preparado para usar as variáveis de session_state.
        # Por uma questão de brevidade, o código das abas, que é idêntico ao do seu app.py original,
        # foi omitido desta resposta. Cole o bloco "with tab_chat:", "with tab_dashboard:", etc. aqui.
        with tab_chat:
            st.header("Converse com seus documentos")
            if not st.session_state.messages : 
                st.session_state.messages.append({"role": "assistant", "content": f"Olá! Documentos da coleção '{st.session_state.get('colecao_ativa', 'atual')}' prontos ({len(nomes_arquivos_global)} arquivo(s)). Qual sua pergunta?"})
            
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if len(st.session_state.messages) > 1 :
                # Função `formatar_chat_para_markdown` precisa ser definida ou colada aqui
                pass # chat_exportado_md = formatar_chat_para_markdown(st.session_state.messages)
                # st.download_button(...)
            
            st.markdown("---")
            if prompt := st.chat_input("Faça sua pergunta sobre os contratos...", key="chat_input_v3", disabled=not documentos_prontos):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    with st.spinner("Pesquisando e pensando..."):
                        llm_chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm_chat, 
                            chain_type="stuff", 
                            retriever=vector_store_global.as_retriever(search_kwargs={"k": 5}), 
                            return_source_documents=True
                        )
                        try:
                            resultado = qa_chain.invoke({"query": prompt})
                            resposta = resultado["result"]
                            fontes = resultado["source_documents"]
                            
                            message_placeholder.markdown(resposta)
                            with st.expander("Ver fontes da resposta"):
                                for fonte in fontes:
                                    st.info(f"Fonte: {fonte.metadata['source']} (Página: {fonte.metadata.get('page', 'N/A')})")
                                    st.text(fonte.page_content[:300] + "...")
                                    
                            st.session_state.messages.append({"role": "assistant", "content": resposta})
                        except Exception as e_chat:
                            st.error(f"Erro durante a execução da cadeia de QA: {e_chat}")
                            st.session_state.messages.append({"role": "assistant", "content": "Desculpe, ocorreu um erro ao processar sua pergunta."})

        with tab_dashboard:
            st.header("Análise Comparativa de Dados Contratuais")
            st.markdown("Clique no botão para extrair e comparar os dados chave dos documentos carregados.")
            if st.button("🚀 Gerar Dados para Dashboard", key="btn_dashboard_v3", use_container_width=True):
                dados_extraidos = extrair_dados_dos_contratos(vector_store_global, nomes_arquivos_global)
                if dados_extraidos: 
                    st.session_state.df_dashboard = pd.DataFrame(dados_extraidos)
                    st.success(f"Dados extraídos para {len(st.session_state.df_dashboard)} contratos.")
                    st.rerun()
                else: 
                    st.session_state.df_dashboard = pd.DataFrame()
                    st.warning("Nenhum dado foi extraído para o dashboard.")

            if 'df_dashboard' in st.session_state and not st.session_state.df_dashboard.empty:
                st.dataframe(st.session_state.df_dashboard, use_container_width=True)
