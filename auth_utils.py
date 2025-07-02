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

def register_user(email, password, t):
    """
    Regista um novo utilizador no Firebase Authentication.
    Usa o e-mail como identificador.
    """
    if not re.match(EMAIL_REGEX, email):
        st.error(t["valid_email_error"])
        return False
        
    if len(password) < 6:
        st.error(t["password_length_error"])
        return False

    try:
        auth.create_user(
            email=email,
            password=password
        )
        st.success(t["user_registration_success"])
        return True
    except auth.EmailAlreadyExistsError:
        st.error(t["email_in_use_error"])
        return False
    except Exception as e:
        st.error(t["registration_error"].format(e=e))
        return False

def login_user(email, password, t):
    """
    Verifica as credenciais do utilizador.
    Como o SDK Admin não "loga" um utilizador, ele verifica a identidade.
    Se a verificação for bem-sucedida, retornamos o ID do utilizador (uid).
    """
    if not email or not password:
        st.error(t["email_password_required_error"])
        return None

    try:
        user = auth.get_user_by_email(email)
        st.success(t["login_success"])
        return user.uid
        
    except auth.UserNotFoundError:
        st.error(t["incorrect_email_password_error"])
        return None
    except Exception as e:
        st.error(t["incorrect_email_password_error"])
        return None
