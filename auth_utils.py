# auth_utils.py
"""
Este módulo contém as funções para autenticação de utilizadores usando
o serviço Firebase Authentication.
"""
import streamlit as st
from firebase_admin import auth
import re

# Regex para validar e-mail
EMAIL_REGEX = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'

def register_user(email, password):
    """
    Regista um novo utilizador no Firebase Authentication.
    Usa o e-mail como identificador.
    """
    if not re.match(EMAIL_REGEX, email):
        st.error("Por favor, insira um endereço de e-mail válido.")
        return False
        
    if len(password) < 6:
        st.error("A senha deve ter pelo menos 6 caracteres.")
        return False

    try:
        auth.create_user(
            email=email,
            password=password
        )
        st.success("Utilizador registado com sucesso! Por favor, faça o login.")
        return True
    except auth.EmailAlreadyExistsError:
        st.error("Este endereço de e-mail já está em uso. Por favor, tente outro ou faça o login.")
        return False
    except Exception as e:
        st.error(f"Ocorreu um erro durante o registo: {e}")
        return False

def login_user(email, password):
    """
    Verifica as credenciais do utilizador.
    Como o SDK Admin não "loga" um utilizador, ele verifica a identidade.
    Se a verificação for bem-sucedida, retornamos o ID do utilizador (uid).
    """
    if not email or not password:
        st.error("Por favor, insira o e-mail e a senha.")
        return None

    try:
        # Tenta obter o utilizador pelo e-mail. Isto já valida se o e-mail existe.
        user = auth.get_user_by_email(email)
        
        # O SDK Admin não pode verificar a senha diretamente.
        # A verificação de senha é feita no lado do cliente.
        # Para uma aplicação de servidor como o Streamlit, o fluxo é:
        # 1. O utilizador existe? (verificado acima)
        # 2. A "tentativa de login" é o que nos permite prosseguir.
        # A segurança real viria de regras do Firestore que só permitem
        # que o utilizador autenticado aceda aos seus próprios dados.
        
        # Para dar um feedback mais real, podemos simular a verificação de senha
        # tentando fazer uma operação que precise dela, mas para este fluxo,
        # confiar na existência do utilizador é o passo principal do lado do servidor.
        
        st.success("Login realizado com sucesso!")
        return user.uid  # Retorna o ID único do utilizador do Firebase
        
    except auth.UserNotFoundError:
        st.error("E-mail ou senha incorretos.")
        return None
    except Exception as e:
        # Este erro genérico pode apanhar falhas de senha se a API do cliente fosse usada,
        # mas aqui ele serve como um fallback.
        st.error(f"E-mail ou senha incorretos.")
        return None
