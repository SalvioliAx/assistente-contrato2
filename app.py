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
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
import tempfile
import zipfile

# --- INTEGRACIÓN CON FIREBASE ---
import firebase_admin
from firebase_admin import credentials, firestore, storage

# Importações do LangChain e Pydantic
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document

# --- INICIALIZAÇÃO DO FIREBASE ---
# Esta função inicializa o Firebase usando as credenciais armazenadas nos Secrets do Streamlit.
# Garanta que seu arquivo .streamlit/secrets.toml está configurado corretamente.
@st.cache_resource
def initialize_firebase():
    """
    Initializes the Firebase Admin SDK using Streamlit secrets.
    Returns the Firestore database client and storage bucket name.
    """
    try:
        # Tenta obter as credenciais do Streamlit Secrets
        creds_dict = st.secrets["firebase_credentials"]
        # Obtém o nome do bucket de armazenamento
        bucket_name = st.secrets["firebase_config"]["storageBucket"]
        
        # Verifica se o app já foi inicializado para evitar erros
        if not firebase_admin._apps:
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred, {'storageBucket': bucket_name})
        
        db = firestore.client()
        st.sidebar.success("Conexão com Firebase estabelecida!", icon="🔥")
        return db, bucket_name
    except (KeyError, FileNotFoundError):
        st.sidebar.error("Credenciais do Firebase não configuradas nos Secrets do Streamlit.")
        st.error("ERRO: As credenciais do Firebase não foram encontradas. Por favor, configure o arquivo .streamlit/secrets.toml como instruído.")
        return None, None
    except Exception as e:
        st.sidebar.error(f"Erro ao conectar com Firebase: {e}")
        st.error(f"ERRO: Falha ao conectar com o Firebase. Detalhes: {e}")
        return None, None

# Inicializa o Firebase e obtém os clientes de DB e Storage
db, BUCKET_NAME = initialize_firebase()


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
    data_evento_str: Optional[str] = Field(default="Não Especificado", description="A data do evento no form
