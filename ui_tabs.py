# ui_tabs.py
"""
Este módulo contém funções para renderizar o conteúdo de cada aba.
"""
import streamlit as st
import pandas as pd
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
                llm_chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
                
                # --- CORREÇÃO APLICADA AQUI ---
                # O template do prompt foi preenchido corretamente.
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

def render_dashboard_tab(vector_store, nomes_arquivos):
    st.header("📈 Análise Comparativa de Dados Contratuais")
    if st.button("🚀 Gerar Dados para o Dashboard", key="btn_dashboard", use_container_width=True):
        dados_extraidos = extrair_dados_dos_contratos(vector_store, nomes_arquivos)
        if dados_extraidos: 
            st.session_state.df_dashboard = pd.DataFrame(dados_extraidos)
    if 'df_dashboard' in st.session_state and not st.session_state.df_dashboard.empty:
        st.dataframe(st.session_state.df_dashboard, use_container_width=True)

def render_resumo_tab(vector_store, nomes_arquivos):
    st.header("📜 Resumo Executivo de um Contrato")
    arquivo_selecionado = st.selectbox("Escolha um contrato para resumir:", options=nomes_arquivos, key="select_resumo", index=None)
    if st.button("✍️ Gerar Resumo Executivo", key="btn_resumo", use_container_width=True, disabled=not arquivo_selecionado):
        texto_completo = _get_full_text_from_vector_store(vector_store, arquivo_selecionado)
        if texto_completo:
            resumo = gerar_resumo_executivo(texto_completo, arquivo_selecionado)
            st.session_state.resumo_gerado = resumo
            st.session_state.arquivo_resumido = arquivo_selecionado
    if 'arquivo_resumido' in st.session_state and st.session_state.arquivo_resumido == arquivo_selecionado:
        st.markdown(st.session_state.resumo_gerado)

def render_riscos_tab(vector_store, nomes_arquivos):
    st.header("🚩 Análise de Cláusulas de Risco")
    arquivo_selecionado = st.selectbox("Escolha um contrato para analisar os riscos:", options=nomes_arquivos, key="select_riscos", index=None)
    if st.button("🔎 Analisar Riscos", key="btn_riscos", use_container_width=True, disabled=not arquivo_selecionado):
        texto_completo = _get_full_text_from_vector_store(vector_store, arquivo_selecionado)
        if texto_completo:
            analise = analisar_documento_para_riscos(texto_completo, arquivo_selecionado)
            st.session_state.analise_riscos_resultado = {"nome_arquivo": arquivo_selecionado, "analise": analise}
    if 'analise_riscos_resultado' in st.session_state and st.session_state.analise_riscos_resultado['nome_arquivo'] == arquivo_selecionado:
        st.markdown(st.session_state.analise_riscos_resultado['analise'])

def render_prazos_tab(vector_store, nomes_arquivos):
    st.header("🗓️ Monitorização de Prazos e Vencimentos")
    if st.button("🔍 Analisar Prazos e Datas em Todos os Contratos", key="btn_prazos", use_container_width=True):
        textos_docs = [{"nome": arq, "texto": _get_full_text_from_vector_store(vector_store, arq)} for arq in nomes_arquivos]
        eventos_extraidos = extrair_eventos_dos_contratos(textos_docs)
        st.session_state.eventos_contratuais_df = pd.DataFrame(eventos_extraidos) if eventos_extraidos else pd.DataFrame()
    if 'eventos_contratuais_df' in st.session_state and not st.session_state.eventos_contratuais_df.empty:
        st.dataframe(st.session_state.eventos_contratuais_df, use_container_width=True)

def render_conformidade_tab(vector_store, nomes_arquivos):
    st.header("⚖️ Verificador de Conformidade Contratual")
    # ... (código inalterado)

def render_anomalias_tab():
    st.header("📊 Deteção de Anomalias Contratuais")
    # ... (código inalterado)
