# firebase_utils.py
"""
Este módulo centraliza todas as interações com o Google Firebase,
de forma específica para cada usuário.
"""
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, storage
import tempfile
import zipfile
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS

@st.cache_resource(show_spinner=False) # Spinner será customizado
def initialize_services(t):
    """Inicializa o Firebase Admin SDK."""
    with st.spinner(t["connecting_services_spinner"]):
        try:
            creds_secrets_obj = st.secrets["firebase_credentials"]
            creds_dict = dict(creds_secrets_obj)
            bucket_name = st.secrets["firebase_config"]["storageBucket"]

            if not firebase_admin._apps:
                cred = credentials.Certificate(creds_dict)
                firebase_admin.initialize_app(cred, {'storageBucket': bucket_name})

            db_client = firestore.client()
            return db_client, bucket_name
        except Exception as e:
            st.error(t["firebase_connection_error"].format(e=e))
            return None, None

def listar_colecoes_salvas(db_client, user_id, t):
    """Lista as coleções de um usuário específico."""
    if not db_client or not user_id: return []
    try:
        colecoes_ref = db_client.collection('users').document(user_id).collection('ia_collections').stream()
        return [doc.id for doc in colecoes_ref]
    except Exception as e:
        st.error(t["list_collections_error"].format(e=e))
        return []

def salvar_colecao_atual(db_client, user_id, nome_colecao, vector_store_atual, nomes_arquivos_atuais, t):
    """Salva a coleção para um usuário específico."""
    if not user_id:
        st.error(t["user_not_identified_error"])
        return False

    with st.spinner(t["saving_collection_spinner"].format(nome_colecao=nome_colecao)):
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
                st.success(t["collection_saved_success"].format(nome_colecao=nome_colecao))
                return True
            except Exception as e:
                st.error(t["save_collection_error"].format(e=e))
                return False

@st.cache_resource(show_spinner=False) # Spinner será customizado
def carregar_colecao(_db_client, _embeddings_obj, user_id, nome_colecao, _t):
    """Carrega uma coleção de um usuário específico."""
    if not user_id:
        st.error(_t["user_not_identified_error"])
        return None, None
    
    with st.spinner(_t["loading_collection_spinner"]):
        try:
            doc_ref = _db_client.collection('users').document(user_id).collection('ia_collections').document(nome_colecao)
            doc = doc_ref.get()
            if not doc.exists:
                st.error(_t["collection_not_found_error"].format(nome_colecao=nome_colecao))
                return None, None
            
            metadata = doc.to_dict()
            storage_path = metadata.get('storage_path')
            nomes_arquivos = metadata.get('nomes_arquivos')

            bucket = storage.bucket()
            blob = bucket.blob(storage_path)

            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path_temp = Path(temp_dir) / "colecao.zip"
                st.info(_t["downloading_index_info"].format(nome_colecao=nome_colecao))
                blob.download_to_filename(str(zip_path_temp))

                unzip_path = Path(temp_dir) / "unzipped"
                unzip_path.mkdir()
                with zipfile.ZipFile(zip_path_temp, 'r') as zip_ref:
                    zip_ref.extractall(unzip_path)
                
                faiss_index_path = unzip_path / "faiss_index"
                vector_store = FAISS.load_local(
                    str(faiss_index_path), 
                    embeddings=_embeddings_obj, 
                    allow_dangerous_deserialization=True
                )
                
                st.success(_t["collection_loaded_success"].format(nome_colecao=nome_colecao))
                return vector_store, nomes_arquivos
        except Exception as e:
            st.error(_t["load_collection_error"].format(nome_colecao=nome_colecao, e=e))
            return None, None
