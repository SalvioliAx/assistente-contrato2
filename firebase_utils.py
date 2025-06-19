# firebase_utils.py
"""
Este módulo centraliza todas as interações com o Google Firebase,
incluindo inicialização, e operações de salvar/carregar no Firestore e Storage.
"""
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, storage
import tempfile
import zipfile
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS # Apenas para type hinting

@st.cache_resource(show_spinner="Conectando aos serviços Google/Firebase...")
def initialize_services():
    """
    Inicializa o Firebase Admin SDK usando credenciais do Streamlit secrets.
    Retorna o cliente do Firestore e o nome do bucket de armazenamento.
    """
    try:
        creds_secrets_obj = st.secrets["firebase_credentials"]
        creds_dict = dict(creds_secrets_obj)
        bucket_name = st.secrets["firebase_config"]["storageBucket"]

        if not firebase_admin._apps:
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred, {'storageBucket': bucket_name})

        db_client = firestore.client()
        
        st.sidebar.success("Serviços Firebase e Google AI conectados!", icon="✔")
        return db_client, bucket_name
    except (KeyError, FileNotFoundError):
        st.sidebar.error("Credenciais/configuração do Firebase não encontradas nos Segredos do Streamlit.")
        st.error("ERRO: Configure o arquivo `secrets.toml` corretamente.")
        return None, None
    except Exception as e:
        st.sidebar.error(f"Erro na conexão com os serviços: {e}")
        st.error(f"ERRO: Falha ao conectar com os serviços Google/Firebase. Detalhes: {e}")
        return None, None

def listar_colecoes_salvas(db_client):
    """Lista os nomes de todas as coleções salvas no Firestore."""
    if not db_client: return []
    try:
        colecoes_ref = db_client.collection('ia_collections').stream()
        return [doc.id for doc in colecoes_ref]
    except Exception as e:
        st.error(f"Erro ao listar coleções do Firebase: {e}")
        return []

def salvar_colecao_atual(db_client, bucket_name, nome_colecao, vector_store_atual, nomes_arquivos_atuais):
    """Salva o vector store no Storage e seus metadados no Firestore."""
    if not db_client or not bucket_name:
        st.error("Conexão com Firebase não disponível para salvar.")
        return False
    if not nome_colecao.strip():
        st.error("Por favor, forneça um nome para a coleção.")
        return False

    with st.spinner(f"Salvando coleção '{nome_colecao}' no Firebase..."):
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
                blob_path = f"collections/{nome_colecao}.zip"
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(str(zip_path_temp))

                doc_ref = db_client.collection('ia_collections').document(nome_colecao)
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

# --- CORREÇÃO APLICADA AQUI ---
# Adicionado '_' aos parâmetros _db_client e _embeddings_obj para indicar ao
# Streamlit que eles não devem ser "hasheados" pelo cache.
@st.cache_resource(show_spinner="Carregando coleção do Firebase...")
def carregar_colecao(_db_client, _embeddings_obj, nome_colecao):
    """Baixa uma coleção do Storage e a carrega em memória."""
    if not _db_client:
        st.error("Conexão com Firebase não disponível para carregar.")
        return None, None
        
    try:
        doc_ref = _db_client.collection('ia_collections').document(nome_colecao)
        doc = doc_ref.get()
        if not doc.exists:
            st.error(f"Coleção '{nome_colecao}' não encontrada no Firestore.")
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
