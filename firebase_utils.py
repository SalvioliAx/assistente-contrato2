# firebase_utils.py
"""
Este módulo centraliza todas as interações com o Google Firebase.
Versão modificada para funcionar no Google Cloud Run com o Secret Manager.
"""
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, storage
import tempfile
import zipfile
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
import json

# Importar o cliente do Secret Manager
from google.cloud import secretmanager

@st.cache_resource(show_spinner="A ligar aos serviços...")
def initialize_services():
    """
    Inicializa o Firebase Admin SDK usando uma credencial do Google Cloud Secret Manager.
    """
    try:
        if not firebase_admin._apps:
            # --- CÓDIGO ATUALIZADO ---
            # Primeiro, tentamos obter as credenciais do Secret Manager.
            # Isto irá funcionar quando a aplicação estiver a correr no Cloud Run.
            try:
                # Substitua 'seu-id-de-projeto' e 'firebase-credentials' pelos seus valores.
                project_id = "seu-id-de-projeto" # <-- SUBSTITUA PELO SEU ID DE PROJETO
                secret_id = "firebase-credentials" # <-- O NOME QUE DEU AO SEGREDO
                version_id = "latest"
                
                name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
                
                client = secretmanager.SecretManagerServiceClient()
                response = client.access_secret_version(name=name)
                creds_json_str = response.payload.data.decode('UTF-8')
                creds_dict = json.loads(creds_json_str)
                cred = credentials.Certificate(creds_dict)

            # Se falhar (por exemplo, ao correr localmente), tentamos carregar um ficheiro local.
            except Exception as e:
                st.warning(f"Não foi possível carregar as credenciais do Secret Manager ({e}). A tentar carregar a partir de um ficheiro local 'secrets.json' para desenvolvimento.")
                # Coloque o seu ficheiro JSON na mesma pasta e renomeie-o para 'secrets.json' para testar localmente.
                cred = credentials.Certificate("secrets.json")
                with open("secrets.json", 'r') as f:
                    creds_dict = json.load(f)

            app_options = {'storageBucket': creds_dict.get('storageBucket')}
            firebase_admin.initialize_app(cred, app_options)

        db_client = firestore.client()
        bucket_name = storage.bucket().name if firebase_admin.get_app().options.get('storageBucket') else None
        
        return db_client, bucket_name
    except Exception as e:
        st.error(f"ERRO: Falha crítica ao inicializar os serviços do Firebase. Detalhes: {e}")
        return None, None

# O resto das funções (listar_colecoes_salvas, etc.) não precisa de ser alterado.
def listar_colecoes_salvas(db_client, user_id):
    if not db_client or not user_id: return []
    try:
        colecoes_ref = db_client.collection('users').document(user_id).collection('ia_collections').stream()
        return [doc.id for doc in colecoes_ref]
    except Exception as e:
        st.error(f"Erro ao listar coleções do Firebase: {e}")
        return []

def salvar_colecao_atual(db_client, user_id, nome_colecao, vector_store_atual, nomes_arquivos_atuais):
    if not user_id:
        st.error("Utilizador não identificado. Não é possível salvar a coleção.")
        return False
    # ... (código inalterado)

def carregar_colecao(_db_client, _embeddings_obj, user_id, nome_colecao):
    if not user_id:
        st.error("Utilizador não identificado. Não é possível carregar a coleção.")
        return None, None
    # ... (código inalterado)
