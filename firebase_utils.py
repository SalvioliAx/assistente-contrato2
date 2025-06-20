# firebase_utils.py
"""
Este módulo centraliza todas as interações com o Google Firebase.
Versão modificada para funcionar no Google Cloud Run com o Secret Manager.
"""
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, storage
import json
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
import tempfile
import zipfile  # <-- CORREÇÃO: Módulo importado

# Importar o cliente do Secret Manager
from google.cloud import secretmanager

@st.cache_resource(show_spinner="A ligar aos serviços...")
def initialize_services():
    """
    Inicializa o Firebase Admin SDK. Outras bibliotecas da Google (como a LangChain)
    usarão as credenciais do ambiente fornecidas automaticamente pelo Cloud Run.
    """
    try:
        # Só executa a configuração uma vez
        if not firebase_admin._apps:
            project_id = "contratiapy"
            
            try:
                # Obter credenciais do Secret Manager (para produção no Cloud Run)
                secret_id = "firebase-credentials"
                client = secretmanager.SecretManagerServiceClient()
                name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
                response = client.access_secret_version(name=name)
                creds_json_str = response.payload.data.decode('UTF-8')
                creds_dict = json.loads(creds_json_str)
                
                cred = credentials.Certificate(creds_dict)
                app_options = {'storageBucket': f"{project_id}.appspot.com"}
                firebase_admin.initialize_app(cred, app_options)

            except Exception as e_secret:
                # Se o Secret Manager falhar (por exemplo, ao correr localmente)
                st.warning(f"Não foi possível carregar as credenciais do Secret Manager ({e_secret}). A tentar usar as credenciais padrão do ambiente (ADC).")
                try:
                    cred = credentials.ApplicationDefault()
                    app_options = {'storageBucket': f"{project_id}.appspot.com", 'projectId': project_id}
                    firebase_admin.initialize_app(cred, app_options)
                except Exception as e_default:
                     st.error(f"Falha na inicialização padrão do Firebase. Certifique-se de que está autenticado se estiver a correr localmente. Erro: {e_default}")
                     return None, None

        db_client = firestore.client()
        bucket_name = storage.bucket().name
        
        return db_client, bucket_name
    except Exception as e:
        st.error(f"ERRO: Falha crítica ao inicializar os serviços. Detalhes: {e}")
        return None, None

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
    with st.spinner(f"Salvando coleção '{nome_colecao}'..."):
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                faiss_path = Path(temp_dir) / "faiss_index"
                vector_store_atual.save_local(str(faiss_path))
                zip_path_temp = Path(tempfile.gettempdir()) / f"{nome_colecao}.zip"
                with zipfile.ZipFile(zip_path_temp, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(faiss_path):
                        for file in files:
                            full_path = Path(root) / file
                            relative_path = full_path.relative_to(Path(temp_dir))
                            zipf.write(full_path, arcname=relative_path)
                bucket = storage.bucket()
                blob_path = f"user_collections/{user_id}/{nome_colecao}.zip"
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(str(zip_path_temp))
                doc_ref = db_client.collection('users').document(user_id).collection('ia_collections').document(nome_colecao)
                doc_ref.set({
                    'nomes_arquivos': nomes_arquivos_atuais,
                    'storage_path': blob_path,
                    'created_at': firestore.SERVER_TIMESTAMP
                })
                os.remove(zip_path_temp)
                st.success(f"Coleção '{nome_colecao}' salva com sucesso!")
                return True
            except Exception as e:
                st.error(f"Erro ao salvar coleção no Firebase: {e}")
                return False

def carregar_colecao(_db_client, _embeddings_obj, user_id, nome_colecao):
    if not user_id:
        st.error("Utilizador não identificado. Não é possível carregar a coleção.")
        return None, None
    try:
        doc_ref = _db_client.collection('users').document(user_id).collection('ia_collections').document(nome_colecao)
        doc = doc_ref.get()
        if not doc.exists:
            st.error(f"Coleção '{nome_colecao}' não encontrada.")
            return None, None
        
        metadata = doc.to_dict()
        storage_path = metadata.get('storage_path')
        nomes_arquivos = metadata.get('nomes_arquivos')

        bucket = storage.bucket()
        blob = bucket.blob(storage_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path_temp = Path(temp_dir) / "colecao.zip"
            st.info(f"Baixando índice de '{nome_colecao}'...")
            blob.download_to_filename(str(zip_path_temp))

            unzip_path = Path(temp_dir) / "unzipped"
            unzip_path.mkdir()
            with zipfile.ZipFile(zip_path_temp, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            
            faiss_index_path = unzip_path / "unzipped" / "faiss_index" # Path fix
            vector_store = FAISS.load_local(
                str(faiss_index_path), 
                embeddings=_embeddings_obj, 
                allow_dangerous_deserialization=True
            )
            
            st.success(f"Coleção '{nome_colecao}' carregada com sucesso!")
            return vector_store, nomes_arquivos
    except Exception as e:
        st.error(f"Erro ao carregar coleção '{nome_colecao}': {e}")
        return None, None
