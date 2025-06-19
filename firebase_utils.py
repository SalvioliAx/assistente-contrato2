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

@st.cache_resource(show_spinner="Conectando aos serviços...")
def initialize_services():
    """Inicializa o Firebase Admin SDK."""
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
        st.error(f"ERRO: Falha ao conectar com os serviços Google/Firebase. Detalhes: {e}")
        return None, None

def listar_colecoes_salvas(db_client, user_id):
    """Lista as coleções de um usuário específico."""
    if not db_client or not user_id: return []
    try:
        # O caminho agora inclui o ID do usuário
        colecoes_ref = db_client.collection('users').document(user_id).collection('ia_collections').stream()
        return [doc.id for doc in colecoes_ref]
    except Exception as e:
        st.error(f"Erro ao listar coleções do Firebase: {e}")
        return []

def salvar_colecao_atual(db_client, user_id, nome_colecao, vector_store_atual, nomes_arquivos_atuais):
    """Salva a coleção para um usuário específico."""
    if not user_id:
        st.error("Usuário não identificado. Não é possível salvar a coleção.")
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
                # O caminho no Storage também inclui o ID do usuário para isolamento
                blob_path = f"user_collections/{user_id}/{nome_colecao}.zip"
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(str(zip_path_temp))

                # O caminho no Firestore agora é aninhado sob o usuário
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

@st.cache_resource(show_spinner="Carregando coleção do Firebase...")
def carregar_colecao(_db_client, _embeddings_obj, user_id, nome_colecao):
    """Carrega uma coleção de um usuário específico."""
    if not user_id:
        st.error("Usuário não identificado. Não é possível carregar a coleção.")
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
            
            faiss_index_path = unzip_path / "faiss_index"
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
