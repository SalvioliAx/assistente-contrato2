# ui_tabs.py
"""
Este módulo contém funções para renderizar cada uma das abas (tabs)
da interface do utilizador do Streamlit.
"""
import streamlit as st
import pandas as pd
from datetime import datetime
import time
import fitz # PyMuPDF

from llm_utils import (
    extrair_dados_dos_contratos, 
    gerar_resumo_executivo, 
    analisar_documento_para_riscos,
    extrair_eventos_dos_contratos,
    verificar_conformidade_documento,
    detectar_anomalias_no_dataframe
)
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

def _get_full_text_from_vector_store(vector_store, nome_arquivo, t):
    """
    Reconstrói o texto completo de um ficheiro a partir dos documentos no vector store.
    """
    if not hasattr(vector_store, 'docstore') or not hasattr(vector_store.docstore, '_dict'):
        st.error(t["vector_store_incompatible_error"])
        return ""
        
    docs_arquivo = []
    for doc_id, doc in vector_store.docstore._dict.items():
        if doc.metadata.get('source') == nome_arquivo:
            docs_arquivo.append(doc)
    
    if not docs_arquivo:
        return ""
        
    docs_arquivo.sort(key=lambda x: x.metadata.get('page', 0))
    
    return "\n".join([doc.page_content for doc in docs_arquivo])

def render_chat_tab(vector_store, nomes_arquivos, t):
    """Renderiza a aba de Chat Interativo."""
    st.header(t["chat_header"])
    
    if "messages" not in st.session_state or not st.session_state.messages: 
        colecao = st.session_state.get('colecao_ativa', 'Sessão Atual')
        welcome_msg = t["chat_welcome_message"].format(colecao=colecao, len_files=len(nomes_arquivos))
        st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if user_prompt := st.chat_input(t["chat_input_placeholder"]):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner(t["chat_spinner_text"]):
                llm_chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
                
                prompt_template = """
                Use os seguintes trechos de contexto para responder à pergunta no final.
                A sua tarefa é sintetizar a informação e fornecer uma resposta precisa e direta.
                Se não souber a resposta ou se a informação não estiver no contexto, diga apenas que não encontrou a informação, não tente inventar uma resposta.
                Responda sempre no idioma da pergunta.

                Contexto:
                {context}

                Pergunta:
                {question}

                Resposta Útil:"""
                
                PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                chain_type_kwargs = {"prompt": PROMPT}
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm_chat, 
                    chain_type="stuff", 
                    retriever=vector_store.as_retriever(search_kwargs={"k": 5}), 
                    chain_type_kwargs=chain_type_kwargs,
                    return_source_documents=True
                )
                
                try:
                    resultado = qa_chain.invoke({"query": user_prompt})
                    resposta = resultado["result"]
                    fontes = resultado.get("source_documents")
                    
                    message_placeholder.markdown(resposta)
                    if fontes:
                        with st.expander(t["chat_answer_source_expander"]):
                            for fonte in fontes:
                                source_info = t["chat_source_info"].format(
                                    source=fonte.metadata.get('source', 'N/A'),
                                    page=fonte.metadata.get('page', 'N/A')
                                )
                                st.info(source_info)
                                st.text(fonte.page_content[:300] + "...")
                                    
                    st.session_state.messages.append({"role": "assistant", "content": resposta})
                except Exception as e:
                    st.error(t["chat_error_message"].format(e=e))
                    st.session_state.messages.append({"role": "assistant", "content": t["chat_default_error"]})

def render_dashboard_tab(vector_store, nomes_arquivos, t):
    st.header(t["dashboard_header"])
    st.markdown(t["dashboard_button_label"])
    if st.button(t["dashboard_button_label"], key="btn_dashboard", use_container_width=True):
        dados_extraidos = extrair_dados_dos_contratos(vector_store, nomes_arquivos, t)
        if dados_extraidos:
            st.session_state.df_dashboard = pd.DataFrame(dados_extraidos)
            st.success(t["dashboard_data_extracted_success"].format(num_contracts=len(st.session_state.df_dashboard)))
        else:
            st.session_state.df_dashboard = pd.DataFrame()
            st.warning(t["dashboard_no_data_warning"])
        st.rerun()
    if 'df_dashboard' in st.session_state and not st.session_state.df_dashboard.empty:
        st.dataframe(st.session_state.df_dashboard, use_container_width=True)

def render_resumo_tab(vector_store, nomes_arquivos, t):
    st.header(t["summary_header"])

    arquivo_selecionado = st.selectbox(
        t["summary_selectbox_label"], 
        options=nomes_arquivos, 
        key="select_resumo", 
        index=None
    )
    
    if st.button(t["summary_button_label"], key="btn_resumo", use_container_width=True, disabled=not arquivo_selecionado):
        spinner_text = t["summary_spinner_text"].format(arquivo_selecionado=arquivo_selecionado)
        with st.spinner(spinner_text):
            texto_completo = _get_full_text_from_vector_store(vector_store, arquivo_selecionado, t)
        
        if texto_completo:
            resumo = gerar_resumo_executivo(texto_completo, arquivo_selecionado, t)
            st.session_state.resumo_gerado = resumo
            st.session_state.arquivo_resumido = arquivo_selecionado
        else:
            st.error(t["summary_text_reconstruction_error"].format(arquivo_selecionado=arquivo_selecionado))

    if 'arquivo_resumido' in st.session_state and st.session_state.arquivo_resumido == arquivo_selecionado:
        st.subheader(t["summary_result_header"].format(arquivo_resumido=st.session_state.arquivo_resumido))
        st.markdown(st.session_state.resumo_gerado)

def render_riscos_tab(vector_store, nomes_arquivos, t):
    st.header(t["risks_header"])
    
    arquivo_selecionado = st.selectbox(
        t["risks_selectbox_label"], 
        options=nomes_arquivos, 
        key="select_riscos", 
        index=None
    )
    
    if st.button(t["risks_button_label"], key="btn_riscos", use_container_width=True, disabled=not arquivo_selecionado):
        spinner_text = t["risks_spinner_text"].format(arquivo_selecionado=arquivo_selecionado)
        with st.spinner(spinner_text):
            texto_completo = _get_full_text_from_vector_store(vector_store, arquivo_selecionado, t)

        if texto_completo:
            analise = analisar_documento_para_riscos(texto_completo, arquivo_selecionado, t)
            st.session_state.analise_riscos_resultado = {
                "nome_arquivo": arquivo_selecionado,
                "analise": analise
            }
        else:
            st.error(t["risks_text_reconstruction_error"].format(arquivo_selecionado=arquivo_selecionado))

    if 'analise_riscos_resultado' in st.session_state and st.session_state.analise_riscos_resultado['nome_arquivo'] == arquivo_selecionado:
        resultado = st.session_state.analise_riscos_resultado
        expander_label = t["risks_result_expander_label"].format(nome_arquivo=resultado['nome_arquivo'])
        with st.expander(expander_label, expanded=True):
            st.markdown(resultado['analise'])

def render_prazos_tab(vector_store, nomes_arquivos, t):
    st.header(t["deadlines_header"])
    st.info(t["deadlines_info_all_contracts"])
    
    if st.button(t["deadlines_button_label"], key="btn_prazos", use_container_width=True):
        textos_docs = []
        for nome_arquivo in nomes_arquivos:
            spinner_text = t["deadlines_reconstructing_text_spinner"].format(nome_arquivo=nome_arquivo)
            with st.spinner(spinner_text):
                texto = _get_full_text_from_vector_store(vector_store, nome_arquivo, t)
                if texto:
                    textos_docs.append({"nome": nome_arquivo, "texto": texto})
        
        if textos_docs:
            eventos_extraidos = extrair_eventos_dos_contratos(textos_docs, t)
            if eventos_extraidos:
                df = pd.DataFrame(eventos_extraidos)
                st.session_state.eventos_contratuais_df = df
            else:
                st.warning(t["deadlines_no_events_warning"])
        else:
            st.error(t["deadlines_reconstruction_failed_error"])

    if 'eventos_contratuais_df' in st.session_state and not st.session_state.eventos_contratuais_df.empty:
        st.dataframe(st.session_state.eventos_contratuais_df, use_container_width=True)

def render_conformidade_tab(vector_store, nomes_arquivos, t):
    st.header(t["compliance_header"])
    if len(nomes_arquivos) < 2:
        st.info(t["compliance_need_two_docs_info"])
        return

    col1, col2 = st.columns(2)
    with col1:
        doc_ref_nome = st.selectbox(t["compliance_ref_doc_selectbox"], nomes_arquivos, key="ref_conf", index=None)
    with col2:
        doc_ana_nome = st.selectbox(t["compliance_ana_doc_selectbox"], [n for n in nomes_arquivos if n != doc_ref_nome], key="ana_conf", index=None)

    if st.button(t["compliance_button_label"], key="btn_conf", use_container_width=True, disabled=not (doc_ref_nome and doc_ana_nome)):
        with st.spinner(t["compliance_spinner_text"]):
            texto_ref = _get_full_text_from_vector_store(vector_store, doc_ref_nome, t)
            texto_ana = _get_full_text_from_vector_store(vector_store, doc_ana_nome, t)

        if texto_ref and texto_ana:
            resultado = verificar_conformidade_documento(texto_ref, doc_ref_nome, texto_ana, doc_ana_nome, t)
            st.session_state.conformidade_resultados = resultado
        else:
            st.error(t["compliance_reconstruction_error"])
            
    if 'conformidade_resultados' in st.session_state:
        st.markdown("---")
        st.subheader(t["compliance_report_header"])
        st.markdown(st.session_state.conformidade_resultados)

def render_anomalias_tab(t):
    st.header(t["anomalies_header"])
    
    if 'df_dashboard' not in st.session_state or st.session_state.df_dashboard.empty:
        st.warning(t["anomalies_no_data_warning"])
        return

    if st.button(t["anomalies_button_label"], key="btn_anomalias", use_container_width=True):
        resultados = detectar_anomalias_no_dataframe(st.session_state.df_dashboard, t)
        st.session_state.anomalias_resultados = resultados

    if 'anomalias_resultados' in st.session_state:
        st.subheader(t["anomalies_results_header"])
        for item in st.session_state.anomalias_resultados:
            st.markdown(f"- {item}")
