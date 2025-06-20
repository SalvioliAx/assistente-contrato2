# ui_tabs.py
"""
Este módulo contém funções para renderizar o conteúdo de cada aba
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

def _get_full_text_from_vector_store(vector_store, nome_arquivo):
    """
    Reconstrói o texto completo de um ficheiro a partir dos documentos no vector store.
    """
    if not hasattr(vector_store, 'docstore') or not hasattr(vector_store.docstore, '_dict'):
        st.error("Vector store com formato incompatível ou vazio para reconstrução de texto.")
        return ""
        
    docs_arquivo = []
    for doc_id, doc in vector_store.docstore._dict.items():
        if doc.metadata.get('source') == nome_arquivo:
            docs_arquivo.append(doc)
    
    if not docs_arquivo:
        return ""
        
    docs_arquivo.sort(key=lambda x: x.metadata.get('page', 0))
    
    return "\n".join([doc.page_content for doc in docs_arquivo])

def render_chat_tab(vector_store, nomes_arquivos):
    """Renderiza a aba de Chat Interativo."""
    st.header("💬 Converse com os seus documentos")
    
    if "messages" not in st.session_state or not st.session_state.messages: 
        colecao = st.session_state.get('colecao_ativa', 'Sessão Atual')
        st.session_state.messages = [{"role": "assistant", "content": f"Olá! Documentos da coleção '{colecao}' ({len(nomes_arquivos)} ficheiro(s)) prontos. Qual é a sua pergunta?"}]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if user_prompt := st.chat_input("Faça a sua pergunta sobre os contratos..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("A pesquisar e a pensar..."):
                # A chave de API já foi definida como variável de ambiente no app.py
                llm_chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
                
                prompt_template = """
                Use os seguintes trechos de contexto para responder à pergunta no final.
                A sua tarefa é sintetizar a informação e fornecer uma resposta precisa e direta.
                Se não souber a resposta ou se a informação não estiver no contexto, diga apenas que não encontrou a informação, não tente inventar uma resposta.
                Responda sempre em português do Brasil.

                Contexto:
                {context}

                Pergunta:
                {question}

                Resposta Útil:"""
                
                PROMPT = PromptTemplate(
                    template=prompt_template, input_variables=["context", "question"]
                )
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm_chat, 
                    chain_type="stuff", 
                    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
                    chain_type_kwargs={"prompt": PROMPT},
                    return_source_documents=True
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
                except Exception as e:
                    st.error(f"Erro ao processar a sua pergunta: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": "Desculpe, ocorreu um erro."})

def render_dashboard_tab(vector_store, nomes_arquivos):
    st.header("📈 Análise Comparativa de Dados Contratuais")
    st.markdown("Clique no botão para extrair e comparar os dados chave dos documentos carregados.")
    if st.button("🚀 Gerar Dados para o Dashboard", key="btn_dashboard", use_container_width=True):
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

    arquivo_selecionado = st.selectbox(
        "Escolha um contrato para resumir:", 
        options=nomes_arquivos, 
        key="select_resumo", 
        index=None
    )
    
    if st.button("✍️ Gerar Resumo Executivo", key="btn_resumo", use_container_width=True, disabled=not arquivo_selecionado):
        with st.spinner(f"A preparar o texto de '{arquivo_selecionado}' para resumo..."):
            texto_completo = _get_full_text_from_vector_store(vector_store, arquivo_selecionado)
        
        if texto_completo:
            resumo = gerar_resumo_executivo(texto_completo, arquivo_selecionado)
            st.session_state.resumo_gerado = resumo
            st.session_state.arquivo_resumido = arquivo_selecionado
        else:
            st.error(f"Não foi possível reconstruir o texto do contrato '{arquivo_selecionado}' a partir da coleção.")

    if 'arquivo_resumido' in st.session_state and st.session_state.arquivo_resumido == arquivo_selecionado:
        st.subheader(f"Resumo do Contrato: {st.session_state.arquivo_resumido}")
        st.markdown(st.session_state.resumo_gerado)

def render_riscos_tab(vector_store, nomes_arquivos):
    st.header("🚩 Análise de Cláusulas de Risco")
    
    arquivo_selecionado = st.selectbox(
        "Escolha um contrato para analisar os riscos:", 
        options=nomes_arquivos, 
        key="select_riscos", 
        index=None
    )
    
    if st.button("🔎 Analisar Riscos", key="btn_riscos", use_container_width=True, disabled=not arquivo_selecionado):
        with st.spinner(f"A preparar o texto de '{arquivo_selecionado}' para análise de riscos..."):
            texto_completo = _get_full_text_from_vector_store(vector_store, arquivo_selecionado)

        if texto_completo:
            analise = analisar_documento_para_riscos(texto_completo, arquivo_selecionado)
            st.session_state.analise_riscos_resultado = {
                "nome_arquivo": arquivo_selecionado,
                "analise": analise
            }
        else:
            st.error(f"Não foi possível reconstruir o texto para análise de riscos do contrato '{arquivo_selecionado}'.")

    if 'analise_riscos_resultado' in st.session_state and st.session_state.analise_riscos_resultado['nome_arquivo'] == arquivo_selecionado:
        resultado = st.session_state.analise_riscos_resultado
        with st.expander(f"Riscos Identificados em: {resultado['nome_arquivo']}", expanded=True):
            st.markdown(resultado['analise'])

def render_prazos_tab(vector_store, nomes_arquivos):
    st.header("🗓️ Monitorização de Prazos e Vencimentos")
    st.info("Esta funcionalidade analisa todos os contratos da coleção de uma vez.")
    
    if st.button("🔍 Analisar Prazos e Datas em Todos os Contratos", key="btn_prazos", use_container_width=True):
        textos_docs = []
        for nome_arquivo in nomes_arquivos:
            with st.spinner(f"A reconstruir o texto de '{nome_arquivo}'..."):
                texto = _get_full_text_from_vector_store(vector_store, nome_arquivo)
                if texto:
                    textos_docs.append({"nome": nome_arquivo, "texto": texto})
        
        if textos_docs:
            eventos_extraidos = extrair_eventos_dos_contratos(textos_docs)
            if eventos_extraidos:
                df = pd.DataFrame(eventos_extraidos)
                st.session_state.eventos_contratuais_df = df
            else:
                st.warning("Nenhum evento ou prazo foi extraído dos documentos.")
        else:
            st.error("Falha ao reconstruir os textos dos documentos da coleção.")

    if 'eventos_contratuais_df' in st.session_state and not st.session_state.eventos_contratuais_df.empty:
        st.dataframe(st.session_state.eventos_contratuais_df, use_container_width=True)

def render_conformidade_tab(vector_store, nomes_arquivos):
    st.header("⚖️ Verificador de Conformidade Contratual")
    if len(nomes_arquivos) < 2:
        st.info("É necessário ter pelo menos dois documentos na coleção para usar esta função.")
        return

    col1, col2 = st.columns(2)
    with col1:
        doc_ref_nome = st.selectbox("Documento de Referência:", nomes_arquivos, key="ref_conf", index=None)
    with col2:
        doc_ana_nome = st.selectbox("Documento a Analisar:", [n for n in nomes_arquivos if n != doc_ref_nome], key="ana_conf", index=None)

    if st.button("🔎 Verificar Conformidade", key="btn_conf", use_container_width=True, disabled=not (doc_ref_nome and doc_ana_nome)):
        with st.spinner("A preparar os textos para comparação..."):
            texto_ref = _get_full_text_from_vector_store(vector_store, doc_ref_nome)
            texto_ana = _get_full_text_from_vector_store(vector_store, doc_ana_nome)

        if texto_ref and texto_ana:
            resultado = verificar_conformidade_documento(texto_ref, doc_ref_nome, texto_ana, doc_ana_nome)
            st.session_state.conformidade_resultados = resultado
        else:
            st.error("Não foi possível reconstruir o texto de um ou de ambos os documentos para comparação.")
            
    if 'conformidade_resultados' in st.session_state:
        st.markdown("---")
        st.subheader("Relatório de Conformidade")
        st.markdown(st.session_state.conformidade_resultados)

def render_anomalias_tab():
    st.header("📊 Deteção de Anomalias Contratuais")
    
    if 'df_dashboard' not in st.session_state or st.session_state.df_dashboard.empty:
        st.warning("Os dados para análise ainda não foram gerados. Vá para a aba '📈 Dashboard' e gere os dados primeiro.")
        return

    if st.button("🚨 Detetar Anomalias Agora", key="btn_anomalias", use_container_width=True):
        resultados = detectar_anomalias_no_dataframe(st.session_state.df_dashboard)
        st.session_state.anomalias_resultados = resultados

    if 'anomalias_resultados' in st.session_state:
        st.subheader("Resultados da Deteção de Anomalias:")
        for item in st.session_state.anomalias_resultados:
            st.markdown(f"- {item}")
