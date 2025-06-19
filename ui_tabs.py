# ui_tabs.py
"""
Este módulo contém funções para renderizar o conteúdo de cada aba.
Os títulos e o layout principal são geridos pelo app.py.
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

def _get_full_text_from_vector_store(vector_store, nome_arquivo):
    if not hasattr(vector_store, 'docstore') or not hasattr(vector_store.docstore, '_dict'):
        st.error("Vector store com formato incompatível.")
        return ""
    docs_arquivo = [doc for doc in vector_store.docstore._dict.values() if doc.metadata.get('source') == nome_arquivo]
    docs_arquivo.sort(key=lambda x: x.metadata.get('page', 0))
    return "\n".join([doc.page_content for doc in docs_arquivo])

def render_chat_tab(vector_store, nomes_arquivos):
    st.header("💬 Converse com os seus documentos")
    
    if "messages" not in st.session_state or not st.session_state.messages: 
        colecao = st.session_state.get('colecao_ativa', 'Sessão Atual')
        st.session_state.messages = [{"role": "assistant", "content": f"Olá! Documentos da coleção '{colecao}' ({len(nomes_arquivos)} ficheiro(s)) prontos. Qual é a sua pergunta?"}]
    
    # Área de chat com altura definida e scroll
    with st.container():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Input de chat no fundo
    if user_prompt := st.chat_input("Faça a sua pergunta sobre os contratos..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        st.rerun() # Reroda para exibir a mensagem do utilizador imediatamente

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("A pesquisar e a pensar..."):
                llm_chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
                prompt_template = """
                Use os seguintes trechos de contexto para responder à pergunta.
                Sintetize a informação e forneça uma resposta precisa e direta.
                Se a informação não estiver no contexto, diga que não encontrou a informação.
                Responda sempre em português do Brasil.
                Contexto: {context}
                Pergunta: {question}
                Resposta Útil:"""
                PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm_chat, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
                    chain_type_kwargs={"prompt": PROMPT}, return_source_documents=True
                )
                try:
                    resultado = qa_chain.invoke({"query": user_prompt})
                    resposta = resultado["result"]
                    fontes = resultado.get("source_documents")
                    message_placeholder.markdown(resposta)
                    if fontes:
                        with st.expander("Ver fontes da resposta"):
                            for fonte in fontes:
                                st.info(f"Fonte: {fonte.metadata.get('source', 'N/A')} (Página: {fonte.metadata.get('page', 'N/A')})")
                                st.text(fonte.page_content[:300] + "...")
                    st.session_state.messages.append({"role": "assistant", "content": resposta})
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao processar a sua pergunta: {e}")

def render_dashboard_tab(vector_store, nomes_arquivos):
    st.header("📈 Análise Comparativa de Dados Contratuais")
    st.markdown("Clique no botão para extrair e comparar os dados chave dos documentos carregados.")
    if st.button("🚀 Gerar Dados para o Dashboard", key="btn_dashboard", use_container_width=True):
        with st.spinner("A extrair dados... Isto pode demorar alguns minutos."):
            dados_extraidos = extrair_dados_dos_contratos(vector_store, nomes_arquivos)
        if dados_extraidos:
            st.session_state.df_dashboard = pd.DataFrame(dados_extraidos)
            st.success(f"Dados extraídos para {len(st.session_state.df_dashboard)} contratos.")
        else:
            st.session_state.df_dashboard = pd.DataFrame()
            st.warning("Nenhum dado foi extraído para o dashboard.")
        st.rerun()
    if 'df_dashboard' in st.session_state and not st.session_state.df_dashboard.empty:
        st.dataframe(st.session_state.df_dashboard, use_container_width=True)

def render_resumo_tab(vector_store, nomes_arquivos):
    st.header("📜 Resumo Executivo de um Contrato")
    arquivo_selecionado = st.selectbox("Escolha um contrato para resumir:", options=nomes_arquivos, key="select_resumo", index=None)
    if st.button("✍️ Gerar Resumo Executivo", key="btn_resumo", use_container_width=True, disabled=not arquivo_selecionado):
        with st.spinner(f"A preparar o texto de '{arquivo_selecionado}'..."):
            texto_completo = _get_full_text_from_vector_store(vector_store, arquivo_selecionado)
        if texto_completo:
            resumo = gerar_resumo_executivo(texto_completo, arquivo_selecionado)
            st.session_state.resumo_gerado = resumo
            st.session_state.arquivo_resumido = arquivo_selecionado
        else:
            st.error(f"Não foi possível reconstruir o texto do contrato '{arquivo_selecionado}'.")
    if 'arquivo_resumido' in st.session_state and st.session_state.arquivo_resumido == arquivo_selecionado:
        st.subheader(f"Resumo do Contrato: {st.session_state.arquivo_resumido}")
        st.markdown(st.session_state.resumo_gerado)

def render_riscos_tab(vector_store, nomes_arquivos):
    st.header("🚩 Análise de Cláusulas de Risco")
    arquivo_selecionado = st.selectbox("Escolha um contrato para analisar os riscos:", options=nomes_arquivos, key="select_riscos", index=None)
    if st.button("🔎 Analisar Riscos", key="btn_riscos", use_container_width=True, disabled=not arquivo_selecionado):
        with st.spinner(f"A analisar os riscos de '{arquivo_selecionado}'..."):
            texto_completo = _get_full_text_from_vector_store(vector_store, arquivo_selecionado)
            if texto_completo:
                analise = analisar_documento_para_riscos(texto_completo, arquivo_selecionado)
                st.session_state.analise_riscos_resultado = {"nome_arquivo": arquivo_selecionado, "analise": analise}
            else:
                st.error(f"Não foi possível reconstruir o texto do contrato '{arquivo_selecionado}'.")
    if 'analise_riscos_resultado' in st.session_state and st.session_state.analise_riscos_resultado['nome_arquivo'] == arquivo_selecionado:
        resultado = st.session_state.analise_riscos_resultado
        with st.container():
            st.markdown(resultado['analise'])

def render_prazos_tab(vector_store, nomes_arquivos):
    st.header("🗓️ Monitorização de Prazos e Vencimentos")
    st.info("Esta funcionalidade analisa todos os contratos da coleção de uma vez.")
    if st.button("🔍 Analisar Prazos e Datas em Todos os Contratos", key="btn_prazos", use_container_width=True):
        textos_docs = []
        with st.spinner("A reconstruir e analisar documentos..."):
            for nome_arquivo in nomes_arquivos:
                texto = _get_full_text_from_vector_store(vector_store, nome_arquivo)
                if texto:
                    textos_docs.append({"nome": nome_arquivo, "texto": texto})
        if textos_docs:
            eventos_extraidos = extrair_eventos_dos_contratos(textos_docs)
            st.session_state.eventos_contratuais_df = pd.DataFrame(eventos_extraidos) if eventos_extraidos else pd.DataFrame()
    if 'eventos_contratuais_df' in st.session_state and not st.session_state.eventos_contratuais_df.empty:
        st.dataframe(st.session_state.eventos_contratuais_df, use_container_width=True)

def render_conformidade_tab(vector_store, nomes_arquivos):
    st.header("⚖️ Verificador de Conformidade Contratual")
    if len(nomes_arquivos) < 2:
        st.info("É necessário ter pelo menos dois documentos para usar esta função.")
        return
    col1, col2 = st.columns(2)
    doc_ref_nome = col1.selectbox("Documento de Referência:", nomes_arquivos, key="ref_conf", index=None)
    doc_ana_nome = col2.selectbox("Documento a Analisar:", [n for n in nomes_arquivos if n != doc_ref_nome], key="ana_conf", index=None)
    if st.button("🔎 Verificar Conformidade", key="btn_conf", use_container_width=True, disabled=not (doc_ref_nome and doc_ana_nome)):
        with st.spinner("A preparar e comparar textos..."):
            texto_ref = _get_full_text_from_vector_store(vector_store, doc_ref_nome)
            texto_ana = _get_full_text_from_vector_store(vector_store, doc_ana_nome)
        if texto_ref and texto_ana:
            resultado = verificar_conformidade_documento(texto_ref, doc_ref_nome, texto_ana, doc_ana_nome)
            st.session_state.conformidade_resultados = resultado
    if 'conformidade_resultados' in st.session_state:
        st.markdown("---")
        st.subheader("Relatório de Conformidade")
        st.markdown(st.session_state.conformidade_resultados)

def render_anomalias_tab(vector_store, nomes_arquivos):
    st.header("📊 Deteção de Anomalias Contratuais")
    if 'df_dashboard' not in st.session_state or st.session_state.df_dashboard.empty:
        st.warning("Gere os dados no 'Dashboard' primeiro para poder detetar anomalias.")
        return
    if st.button("🚨 Detetar Anomalias Agora", key="btn_anomalias", use_container_width=True):
        resultados = detectar_anomalias_no_dataframe(st.session_state.df_dashboard)
        st.session_state.anomalias_resultados = resultados
    if 'anomalias_resultados' in st.session_state:
        st.subheader("Resultados da Deteção de Anomalias:")
        for item in st.session_state.anomalias_resultados:
            st.markdown(f"- {item}")
