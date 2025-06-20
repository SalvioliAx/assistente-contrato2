# pdf_processing.py
import streamlit as st
import os
import tempfile
from pathlib import Path
import fitz  # PyMuPDF
import base64
import time
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
            base64_image = base64.b64encode(img_bytes).decode('UTF-8')

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
            time.sleep(2) # To respect API rate limits
        
        if texto_extraido:
            st.success(f"Texto extraído com Gemini Vision para {nome_arquivo}.")
        else:
            st.warning(f"Gemini Vision não retornou texto substancial para {nome_arquivo}.")

    except Exception as e_gemini:
        st.error(f"Erro ao usar Gemini Vision em {nome_arquivo}: {e_gemini}")
    
    return documentos_gemini, texto_extraido

@st.cache_resource
# CORREÇÃO: Removido o parâmetro 'api_key' da assinatura da função.
def obter_vector_store_de_uploads(_lista_arquivos_pdf_upload, _embeddings_obj):
    """
    Processa uma lista de arquivos PDF, extrai texto e cria um Vector Store FAISS.
    Usa PyMuPDF como método principal e Gemini Vision como fallback.
    """
    if not _lista_arquivos_pdf_upload:
        return None, None

    documentos_totais = []
    nomes_arquivos_processados = []
    
    # CORREÇÃO: Removido 'google_api_key'. A biblioteca usará a 
    # variável de ambiente "GOOGLE_API_KEY" que foi definida no app.py.
    llm_vision = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", 
        temperature=0.1
    )

    for arquivo_pdf in _lista_arquivos_pdf_upload:
        nome_arquivo = arquivo_pdf.name
        st.info(f"Processando: {nome_arquivo}...")
        
        docs_arquivo_atual = []
        sucesso = False
        
        try:
            # Tentativa 1: PyMuPDF (fitz) - método principal
            st.write(f"A extrair texto com PyMuPDF para {nome_arquivo}...")
            arquivo_pdf.seek(0)
            doc_fitz = fitz.open(stream=arquivo_pdf.read(), filetype="pdf")
            for num_pagina, pagina in enumerate(doc_fitz):
                texto = pagina.get_text("text")
                if texto.strip():
                    docs_arquivo_atual.append(Document(page_content=texto, metadata={"source": nome_arquivo, "page": num_pagina, "method": "pymupdf"}))
            if docs_arquivo_atual:
                sucesso = True
                st.success(f"Texto extraído com PyMuPDF para {nome_arquivo}.")

            # Tentativa 2: Gemini Vision como fallback
            if not sucesso:
                st.write(f"PyMuPDF não extraiu texto. A tentar Gemini Vision para {nome_arquivo}...")
                arquivo_pdf.seek(0)
                pdf_bytes = arquivo_pdf.read()
                docs_gemini, sucesso_gemini = _extrair_texto_com_gemini(pdf_bytes, nome_arquivo, llm_vision)
                if sucesso_gemini:
                    docs_arquivo_atual = docs_gemini
                    sucesso = True

            if sucesso:
                documentos_totais.extend(docs_arquivo_atual)
                nomes_arquivos_processados.append(nome_arquivo)
            else:
                st.error(f"Falha ao extrair texto de {nome_arquivo} com todos os métodos disponíveis.")

        except Exception as e:
            st.error(f"Erro geral ao processar o ficheiro {nome_arquivo}: {e}")

    if not documentos_totais:
        return None, []

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
