# firebase_utils.py
"""
Este módulo centraliza todas as interações com o Google Firebase.
Versão modificada para funcionar no PythonAnywhere.
"""
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, storage
import tempfile
import zipfile
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS

@st.cache_resource(show_spinner="A ligar aos serviços...")
def initialize_services():
    """
    Inicializa o Firebase Admin SDK usando um ficheiro de credenciais no servidor.
    """
    try:
        # --- ALTERAÇÃO PRINCIPAL AQUI ---
        # Substitua 'seu_username' e 'nome_do_seu_ficheiro.json' pelos seus valores.
        # Este é o caminho para o ficheiro JSON que carregou no PythonAnywhere.
        cred_path = '/home/seu_username/nome_do_seu_ficheiro.json'
        
        # O nome do bucket ainda pode ser obtido do ficheiro JSON se estiver lá,
        # ou pode especificá-lo diretamente.
        # Para simplificar, vamos assumir que está no ficheiro de credenciais.
        
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            # Precisamos de obter o nome do bucket do ficheiro de configuração ou definir manualmente
            # Se o seu ficheiro de credenciais do Firebase não tiver o storageBucket,
            # adicione-o manualmente, por exemplo: {'storageBucket': 'seu-app.appspot.com'}
            app_options = {'storageBucket': st.secrets["firebase_config"]["storageBucket"]} if "firebase_config" in st.secrets else {}
            firebase_admin.initialize_app(cred, app_options)

        db_client = firestore.client()
        bucket_name = storage.bucket().name if firebase_admin.get_app().options.get('storageBucket') else None
        
        return db_client, bucket_name
    except Exception as e:
        st.error(f"ERRO: Falha ao ligar com os serviços Google/Firebase. Detalhes: {e}")
        return None, None

# O resto das funções (listar_colecoes_salvas, salvar_colecao_atual, carregar_colecao)
# não precisam de ser alteradas, pois dependem do db_client e do bucket_name.
