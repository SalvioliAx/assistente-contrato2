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

def _extrair_texto_com_gemini(pdf_bytes, nome_arquivo, llm_vision, t):
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
            
            spinner_text = t["gemini_processing_page_spinner"].format(
                page_num=page_num + 1, 
                len_doc=len(doc_fitz_vision), 
                nome_arquivo=nome_arquivo
            )
            with st.spinner(spinner_text):
                ai_msg = llm_vision.invoke([human_message])
            
            if isinstance(ai_msg, AIMessage) and isinstance(ai_msg.content, str) and ai_msg.content.strip():
                doc = Document(page_content=ai_msg.content, metadata={"source": nome_arquivo, "page": page_num, "method": "gemini_vision"})
                documentos_gemini.append(doc)
                texto_extraido = True
            time.sleep(2)
        
        if texto_extraido:
            st.success(t["text_extracted_gemini_success"].format(nome_arquivo=nome_arquivo))
        else:
            st.warning(t["gemini_no_text_warning"].format(nome_arquivo=nome_arquivo))

    except Exception as e_gemini:
        st.error(t["gemini_vision_error"].format(nome_arquivo=nome_arquivo, e_gemini=e_gemini))
    
    return documentos_gemini, texto_extraido

@st.cache_resource(show_spinner=False)
def obter_vector_store_de_uploads(lista_arquivos_pdf_upload, _embeddings_obj, _t):
    """
    Processa uma lista de arquivos PDF, extrai texto e cria um Vector Store FAISS.
    """
    with st.spinner(_t["analyzing_documents_spinner"]):
        if not lista_arquivos_pdf_upload:
            return None, None

        documentos_totais = []
        nomes_arquivos_processados = []
        llm_vision = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1, request_timeout=300)

        for arquivo_pdf in lista_arquivos_pdf_upload:
            nome_arquivo = arquivo_pdf.name
            st.info(_t["processing_file_info"].format(nome_arquivo=nome_arquivo))
            
            docs_arquivo_atual = []
            sucesso = False
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(arquivo_pdf.getvalue())
                tmp_path = tmp.name

            try:
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
                if pages and any(p.page_content.strip() for p in pages):
                    for p in pages:
                        p.metadata["method"] = "pypdf"
                    docs_arquivo_atual.extend(pages)
                    sucesso = True
                    st.write(_t["text_extracted_pypdf_success"].format(nome_arquivo=nome_arquivo))
            except Exception as e:
                st.write(_t["pypdf_failed_warning"].format(nome_arquivo=nome_arquivo, e=e))

            if not sucesso:
                try:
                    doc_fitz = fitz.open(tmp_path)
                    for num_pagina, pagina in enumerate(doc_fitz):
                        texto = pagina.get_text("text")
                        if texto.strip():
                            docs_arquivo_atual.append(Document(page_content=texto, metadata={"source": nome_arquivo, "page": num_pagina, "method": "pymupdf"}))
                    if docs_arquivo_atual:
                        sucesso = True
                        st.write(_t["text_extracted_pymupdf_success"].format(nome_arquivo=nome_arquivo))
                except Exception as e:
                    st.write(_t["pymupdf_failed_warning"].format(nome_arquivo=nome_arquivo, e=e))
            
            if not sucesso and llm_vision:
                st.write(_t["gemini_vision_attempt_info"].format(nome_arquivo=nome_arquivo))
                arquivo_pdf.seek(0)
                pdf_bytes = arquivo_pdf.read()
                docs_gemini, sucesso_gemini = _extrair_texto_com_gemini(pdf_bytes, nome_arquivo, llm_vision, _t)
                if sucesso_gemini:
                    docs_arquivo_atual = docs_gemini
                    sucesso = True

            os.remove(tmp_path)

            if sucesso:
                documentos_totais.extend(docs_arquivo_atual)
                nomes_arquivos_processados.append(nome_arquivo)
            else:
                st.error(_t["text_extraction_failed_error"].format(nome_arquivo=nome_arquivo))

        if not documentos_totais:
            return None, []

        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
            docs_fragmentados = text_splitter.split_documents(documentos_totais)
            
            st.info(_t["creating_vector_store_info"].format(len_docs=len(docs_fragmentados)))
            vector_store = FAISS.from_documents(docs_fragmentados, _embeddings_obj)
            st.success(_t["vector_store_created_success"])
            return vector_store, nomes_arquivos_processados
        except Exception as e:
            st.error(_t["vector_store_creation_error"].format(e=e))
            return None, nomes_arquivos_processados
