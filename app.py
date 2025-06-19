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

# --- IMPORTAÇÕES DO GOOGLE E LANGCHAIN ---
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
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
        
        # CORREÇÃO: Limpa o nome do bucket de quaisquer prefixos ou sufixos inválidos.
        # Remove "gs://", "http://", "https://", e a barra "/" no final.
        bucket_name = bucket_name.replace("gs://", "").replace("https://", "").replace("http://", "").rstrip("/")

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
    data_evento_str: Optional[str] = Field(default="Não Especificado", description="A data do evento no form
