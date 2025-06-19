# auth_utils.py
"""
Este módulo contém as funções para autenticação de usuários,
incluindo registro, login e gerenciamento de senhas.
"""
import streamlit as st
from werkzeug.security import generate_password_hash, check_password_hash

def register_user(db_client, username, password):
    """
    Registra um novo usuário no Firestore se o nome de usuário não existir.
    Armazena a senha de forma segura usando hash.
    """
    if not username or not password:
        st.error("Usuário e senha não podem estar em branco.")
        return False
        
    users_ref = db_client.collection('users')
    # Verifica se o usuário já existe
    if users_ref.document(username).get().exists:
        st.error(f"O nome de usuário '{username}' já existe. Por favor, escolha outro.")
        return False
    
    # Gera o hash da senha
    hashed_password = generate_password_hash(password)
    
    # Salva o novo usuário
    users_ref.document(username).set({
        'username': username,
        'password_hash': hashed_password
    })
    st.success("Usuário registrado com sucesso! Por favor, faça o login.")
    return True

def login_user(db_client, username, password):
    """
    Verifica as credenciais do usuário contra os dados no Firestore.
    """
    if not username or not password:
        st.error("Por favor, insira o nome de usuário e a senha.")
        return False

    users_ref = db_client.collection('users')
    user_doc = users_ref.document(username).get()
    
    if not user_doc.exists:
        st.error("Nome de usuário ou senha incorretos.")
        return False
        
    user_data = user_doc.to_dict()
    stored_hash = user_data.get('password_hash')
    
    # Verifica se a senha fornecida corresponde ao hash armazenado
    if check_password_hash(stored_hash, password):
        st.success("Login realizado com sucesso!")
        return True
    else:
        st.error("Nome de usuário ou senha incorretos.")
        return False
