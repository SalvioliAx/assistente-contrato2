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
    Também configura as credenciais do Google Cloud para outras bibliotecas.
    """
    try:
        # Só executa a configuração uma vez
        if not firebase_admin._apps:
            creds_dict = None
            try:
                # --- CÓDIGO ATUALIZADO ---
                # Obter credenciais do Secret Manager (para produção no Cloud Run)
                project_id = "contratiapy" # Substituído pelo seu ID de projeto
                secret_id = "firebase-credentials" # O nome que deu ao segredo
                version_id = "latest"
                
                name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
                
                client = secretmanager.SecretManagerServiceClient()
                response = client.access_secret_version(name=name)
                creds_json_str = response.payload.data.decode('UTF-8')
                creds_dict = json.loads(creds_json_str)

            except Exception as e_secret:
                st.warning(f"Não foi possível carregar as credenciais do Secret Manager ({e_secret}). A tentar carregar a partir de um ficheiro local.")
                # Fallback para desenvolvimento local. Crie um ficheiro chamado "secrets.json"
                # na raiz do seu projeto com as credenciais para testar localmente.
                try:
                    with open("secrets.json", 'r') as f:
                        creds_dict = json.load(f)
                except FileNotFoundError:
                    st.error("Credenciais não encontradas no Secret Manager nem no ficheiro local 'secrets.json'.")
                    return None, None
            
            # --- LÓGICA DE AUTENTICAÇÃO PARA TODAS AS GOOGLE SDKs ---
            # Cria um ficheiro temporário com as credenciais JSON
            # e define uma variável de ambiente para que bibliotecas como
            # a do LangChain a possam encontrar automaticamente.
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as creds_file:
                json.dump(creds_dict, creds_file)
                creds_file_path = creds_file.name
            
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_file_path
            
            # Inicializa o Firebase com as mesmas credenciais
            cred = credentials.Certificate(creds_dict)
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
    # A implementação desta função continua igual
    pass

def carregar_colecao(_db_client, _embeddings_obj, user_id, nome_colecao):
    # A implementação desta função continua igual
    pass
