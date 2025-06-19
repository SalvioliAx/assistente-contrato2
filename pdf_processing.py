# pdf_processing.py
"""
Este módulo contém toda a lógica para processamento de arquivos PDF.
Isso inclui extração de texto usando múltiplos métodos e a criação
de um Vector Store com FAISS para busca de similaridade.
"""
import tempfile
import streamlit as st
import os
from pathlib import Path
import fitz  # PyMuPDF
import base64
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document

def _extrair_texto_com_gemini(pdf_bytes, nome_arquivo, llm_vision):
    """Função auxiliar para extrair texto de um PDF usando Gemini Vision."""
    documentos_gemini = []
    texto_extraido = False
    try:
        doc_fitz_vision = fitz.open(stream=pdf_bytes, filetype="pdf")
        prompt_ocr = "Você é um especialista em OCR. Extraia todo o texto visível desta página de documento de forma precisa, mantendo a estrutura original."
        
        for page_num in range(len(doc_fitz_vision)):
            page_obj = doc_fitz_vision.load_page(page_num)
            pix = page_obj.get_pixmap(dpi=300) 
            img_bytes = pix.tobytes("png")
            base64_image = base64.b64encode(img_bytes).decode('utf-8')

            human_message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt_ocr},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{base64_image}"}
                ]
            )
            
            with st.spinner(f"Gemini processando pág. {page_num + 1}/{len(doc_fitz_vision)} de {nome_arquivo}..."):
                ai_msg = llm_vision.invoke([human_message])
            
            if isinstance(ai_msg, AIMessage) and isinstance(ai_msg.content, str) and ai_msg.content.strip():
                doc = Document(page_content=ai_msg.content, metadata={"source": nome_arquivo, "page": page_num, "method": "gemini_vision"})
                documentos_gemini.append(doc)
                texto_extraido = True
            time.sleep(2) # Respeitar limites da API
        
        if texto_extraido:
            st.success(f"Texto extraído com Gemini Vision para {nome_arquivo}.")
        else:
            st.warning(f"Gemini Vision não retornou texto substancial para {nome_arquivo}.")

    except Exception as e_gemini:
        st.error(f"Erro ao usar Gemini Vision em {nome_arquivo}: {e_gemini}")
    
    return documentos_gemini, texto_extraido

@st.cache_resource(show_spinner="Analisando documentos para busca e chat...")
def obter_vector_store_de_uploads(lista_arquivos_pdf_upload, _embeddings_obj):
    """
    Processa uma lista de arquivos PDF, extrai texto e cria um Vector Store FAISS.
    Tenta múltiplos métodos de extração: PyPDFLoader, PyMuPDF e Gemini Vision como fallback.
    """
    if not lista_arquivos_pdf_upload:
        return None, None

    documentos_totais = []
    nomes_arquivos_processados = []
    llm_vision = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1, request_timeout=300)

    for arquivo_pdf in lista_arquivos_pdf_upload:
        nome_arquivo = arquivo_pdf.name
        st.info(f"Processando: {nome_arquivo}...")
        
        docs_arquivo_atual = []
        sucesso = False
        
        # Escreve arquivo em disco temporariamente para os loaders usarem
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(arquivo_pdf.getvalue())
            tmp_path = tmp.name

        # Tentativa 1: PyPDFLoader
        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            if pages and any(p.page_content.strip() for p in pages):
                for p in pages:
                    p.metadata["method"] = "pypdf"
                docs_arquivo_atual.extend(pages)
                sucesso = True
                st.write(f"Texto extraído com PyPDFLoader para {nome_arquivo}.")
        except Exception as e:
            st.write(f"PyPDFLoader falhou para {nome_arquivo}: {e}. Tentando PyMuPDF.")

        # Tentativa 2: PyMuPDF (fitz)
        if not sucesso:
            try:
                doc_fitz = fitz.open(tmp_path)
                for num_pagina, pagina in enumerate(doc_fitz):
                    texto = pagina.get_text("text")
                    if texto.strip():
                        docs_arquivo_atual.append(Document(page_content=texto, metadata={"source": nome_arquivo, "page": num_pagina, "method": "pymupdf"}))
                if docs_arquivo_atual:
                    sucesso = True
                    st.write(f"Texto extraído com PyMuPDF para {nome_arquivo}.")
            except Exception as e:
                st.write(f"PyMuPDF falhou para {nome_arquivo}: {e}. Tentando Gemini Vision.")
        
        # Tentativa 3: Gemini Vision
        if not sucesso and llm_vision:
            st.write(f"Tentando Gemini Vision para {nome_arquivo}...")
            arquivo_pdf.seek(0)
            pdf_bytes = arquivo_pdf.read()
            docs_gemini, sucesso_gemini = _extrair_texto_com_gemini(pdf_bytes, nome_arquivo, llm_vision)
            if sucesso_gemini:
                docs_arquivo_atual = docs_gemini
                sucesso = True

        os.remove(tmp_path) # Remove o arquivo temporário

        if sucesso:
            documentos_totais.extend(docs_arquivo_atual)
            nomes_arquivos_processados.append(nome_arquivo)
        else:
            st.error(f"Falha ao extrair texto de {nome_arquivo} com todos os métodos.")

    if not documentos_totais:
        return None, []

    # Divide os documentos e cria o Vector Store
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs_fragmentados = text_splitter.split_documents(documentos_totais)
        
        st.info(f"Criando base de vetores com {len(docs_fragmentados)} fragmentos...")
        vector_store = FAISS.from_documents(docs_fragmentados, _embeddings_obj)
        st.success("Base de vetores criada com sucesso!")
        return vector_store, nomes_arquivos_processados
    except Exception as e:
        st.error(f"Erro ao criar o Vector Store com FAISS: {e}")
        return None, nomes_arquivos_processados
