# ui_tabs.py
import streamlit as st
import pandas as pd
from llm_utils import *
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

def _get_full_text_from_vector_store(vector_store, nome_arquivo):
    if not hasattr(vector_store, 'docstore') or not hasattr(vector_store.docstore, '_dict'):
        st.error("Vector store com formato incompatível.")
        return ""
    docs_arquivo = [doc for doc in vector_store.docstore._dict.values() if doc.metadata.get('source') == nome_arquivo]
    docs_arquivo.sort(key=lambda x: x.metadata.get('page', 0))
    return "\n".join([doc.page_content for doc in docs_arquivo])

def render_chat_tab(vector_store, nomes_arquivos, api_key):
    st.header("💬 Converse com os seus documentos")
    
    if "messages" not in st.session_state or not st.session_state.messages: 
        colecao = st.session_state.get('colecao_ativa', 'Sessão Atual')
        st.session_state.messages = [{"role": "assistant", "content": f"Olá! Documentos da coleção '{colecao}' prontos. Qual é a sua pergunta?"}]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if user_prompt := st.chat_input("Faça a sua pergunta..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("A pensar..."):
                llm_chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2, google_api_key=api_key)
                
                prompt_template = """Use os seguintes trechos de contexto para responder à pergunta. Se não souber a resposta, diga que não encontrou a informação. Responda em português.
                Contexto: {context}
                Pergunta: {question}
                Resposta:"""
                PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm_chat, chain_type="stuff", retriever=vector_store.as_retriever(),
                    chain_type_kwargs={"prompt": PROMPT}, return_source_documents=True
                )
                try:
                    resultado = qa_chain.invoke({"query": user_prompt})
                    resposta = resultado["result"]
                    st.session_state.messages.append({"role": "assistant", "content": resposta})
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao processar: {e}")

def render_dashboard_tab(vector_store, nomes_arquivos, api_key):
    st.header("📈 Análise Comparativa")
    if st.button("🚀 Gerar Dados para o Dashboard", use_container_width=True):
        dados_extraidos = extrair_dados_dos_contratos(vector_store, nomes_arquivos, api_key)
        if dados_extraidos: 
            st.session_state.df_dashboard = pd.DataFrame(dados_extraidos)
    if 'df_dashboard' in st.session_state and not st.session_state.df_dashboard.empty:
        st.dataframe(st.session_state.df_dashboard, use_container_width=True)

def render_resumo_tab(vector_store, nomes_arquivos, api_key):
    st.header("📜 Resumo Executivo")
    arquivo_selecionado = st.selectbox("Escolha um contrato para resumir:", nomes_arquivos, index=None)
    if st.button("✍️ Gerar Resumo", use_container_width=True, disabled=not arquivo_selecionado):
        texto_completo = _get_full_text_from_vector_store(vector_store, arquivo_selecionado)
        if texto_completo:
            resumo = gerar_resumo_executivo(texto_completo, arquivo_selecionado, api_key)
            st.session_state.resumo_gerado = resumo
            st.session_state.arquivo_resumido = arquivo_selecionado
    if 'arquivo_resumido' in st.session_state and st.session_state.arquivo_resumido == arquivo_selecionado:
        st.markdown(st.session_state.resumo_gerado)

def render_riscos_tab(vector_store, nomes_arquivos, api_key):
    st.header("🚩 Análise de Riscos")
    arquivo_selecionado = st.selectbox("Escolha um contrato para analisar:", nomes_arquivos, index=None)
    if st.button("🔎 Analisar Riscos", use_container_width=True, disabled=not arquivo_selecionado):
        texto_completo = _get_full_text_from_vector_store(vector_store, arquivo_selecionado)
        if texto_completo:
            analise = analisar_documento_para_riscos(texto_completo, arquivo_selecionado, api_key)
            st.session_state.analise_riscos_resultado = {"nome_arquivo": arquivo_selecionado, "analise": analise}
    if 'analise_riscos_resultado' in st.session_state and st.session_state.analise_riscos_resultado['nome_arquivo'] == arquivo_selecionado:
        st.markdown(st.session_state.analise_riscos_resultado['analise'])

def render_prazos_tab(vector_store, nomes_arquivos, api_key):
    st.header("🗓️ Monitorização de Prazos")
    if st.button("🔍 Analisar Prazos em Todos os Contratos", use_container_width=True):
        textos_docs = [{"nome": arq, "texto": _get_full_text_from_vector_store(vector_store, arq)} for arq in nomes_arquivos]
        eventos_extraidos = extrair_eventos_dos_contratos(textos_docs, api_key)
        st.session_state.eventos_contratuais_df = pd.DataFrame(eventos_extraidos) if eventos_extraidos else pd.DataFrame()
    if 'eventos_contratuais_df' in st.session_state and not st.session_state.eventos_contratuais_df.empty:
        st.dataframe(st.session_state.eventos_contratuais_df, use_container_width=True)

def render_conformidade_tab(vector_store, nomes_arquivos, api_key):
    st.header("⚖️ Verificador de Conformidade")
    if len(nomes_arquivos) < 2:
        st.info("É necessário ter pelo menos dois documentos.")
        return
    col1, col2 = st.columns(2)
    doc_ref_nome = col1.selectbox("Documento de Referência:", nomes_arquivos, key="ref_conf", index=None)
    doc_ana_nome = col2.selectbox("Documento a Analisar:", [n for n in nomes_arquivos if n != doc_ref_nome], key="ana_conf", index=None)
    if st.button("🔎 Verificar Conformidade", use_container_width=True, disabled=not (doc_ref_nome and doc_ana_nome)):
        texto_ref = _get_full_text_from_vector_store(vector_store, doc_ref_nome)
        texto_ana = _get_full_text_from_vector_store(vector_store, doc_ana_nome)
        if texto_ref and texto_ana:
            resultado = verificar_conformidade_documento(texto_ref, doc_ref_nome, texto_ana, doc_ana_nome, api_key)
            st.session_state.conformidade_resultados = resultado
    if 'conformidade_resultados' in st.session_state:
        st.markdown(st.session_state.conformidade_resultados)

def render_anomalias_tab():
    st.header("📊 Deteção de Anomalias")
    if 'df_dashboard' not in st.session_state or st.session_state.df_dashboard.empty:
        st.warning("Gere os dados no 'Dashboard' primeiro.")
        return
    if st.button("🚨 Detetar Anomalias", use_container_width=True):
        resultados = detectar_anomalias_no_dataframe(st.session_state.df_dashboard)
        st.session_state.anomalias_resultados = resultados
    if 'anomalias_resultados' in st.session_state:
        for item in st.session_state.anomalias_resultados:
            st.markdown(f"- {item}")
