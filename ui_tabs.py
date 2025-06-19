# ui_tabs.py
"""
Este módulo contém funções para renderizar cada uma das abas (tabs)
da interface do usuário do Streamlit.
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
from langchain_google_genai import ChatGoogleGenerativeAI

def render_chat_tab(vector_store, nomes_arquivos):
    """Renderiza a aba de Chat Interativo."""
    st.header("💬 Converse com seus documentos")
    
    if "messages" not in st.session_state or not st.session_state.messages: 
        colecao = st.session_state.get('colecao_ativa', 'Sessão Atual')
        st.session_state.messages = [{"role": "assistant", "content": f"Olá! Documentos da coleção '{colecao}' ({len(nomes_arquivos)} arquivo(s)) prontos. Qual sua pergunta?"}]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if user_prompt := st.chat_input("Faça sua pergunta sobre os contratos..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Pesquisando e pensando..."):
                llm_chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm_chat, 
                    chain_type="stuff", 
                    retriever=vector_store.as_retriever(search_kwargs={"k": 5}), 
                    return_source_documents=True
                )
                try:
                    # --- CORREÇÃO APLICADA AQUI ---
                    # Adiciona a instrução para responder em português ao prompt do usuário.
                    prompt_em_portugues = f"{user_prompt}\n\n**Instrução:** Responda em português do Brasil."
                    resultado = qa_chain.invoke({"query": prompt_em_portugues})
                    
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
                    st.error(f"Erro ao processar sua pergunta: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": "Desculpe, ocorreu um erro."})

def render_dashboard_tab(vector_store, nomes_arquivos):
    """Renderiza a aba do Dashboard com dados comparativos."""
    st.header("📈 Análise Comparativa de Dados Contratuais")
    st.markdown("Clique no botão para extrair e comparar os dados chave dos documentos carregados.")
    if st.button("🚀 Gerar Dados para Dashboard", key="btn_dashboard", use_container_width=True):
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

def _get_full_text(uploaded_file):
    """Função auxiliar para ler o texto completo de um PDF."""
    try:
        uploaded_file.seek(0)
        pdf_bytes = uploaded_file.read()
        texto_completo = ""
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                texto_completo += page.get_text() + "\n"
        return texto_completo
    except Exception as e:
        st.error(f"Erro ao ler o arquivo {uploaded_file.name}: {e}")
        return ""

def render_resumo_tab(arquivos_originais, nomes_arquivos):
    """Renderiza a aba de Resumo Executivo."""
    st.header("📜 Resumo Executivo de um Contrato")
    if not arquivos_originais:
        st.info("A função de resumo é otimizada para novos uploads. Para coleções salvas, use o chat para pedir resumos.")
        return

    arquivo_selecionado = st.selectbox("Escolha um contrato para resumir:", options=nomes_arquivos, key="select_resumo", index=None)
    
    if st.button("✍️ Gerar Resumo Executivo", key="btn_resumo", use_container_width=True, disabled=not arquivo_selecionado):
        arquivo_obj = next((arq for arq in arquivos_originais if arq.name == arquivo_selecionado), None)
        if arquivo_obj:
            texto_completo = _get_full_text(arquivo_obj)
            if texto_completo:
                resumo = gerar_resumo_executivo(texto_completo, arquivo_obj.name)
                st.session_state.resumo_gerado = resumo
                st.session_state.arquivo_resumido = arquivo_selecionado
    
    if 'arquivo_resumido' in st.session_state and st.session_state.arquivo_resumido:
        st.subheader(f"Resumo do Contrato: {st.session_state.arquivo_resumido}")
        st.markdown(st.session_state.resumo_gerado)


def render_riscos_tab(arquivos_originais):
    """Renderiza a aba de Análise de Riscos."""
    st.header("🚩 Análise de Cláusulas de Risco")
    if not arquivos_originais:
        st.info("A análise de riscos é otimizada para novos uploads.")
        return
        
    if st.button("🔎 Analisar Riscos em Todos os Documentos", key="btn_riscos", use_container_width=True):
        resultados = {}
        for arquivo in arquivos_originais:
            texto_completo = _get_full_text(arquivo)
            if texto_completo:
                resultados[arquivo.name] = analisar_documento_para_riscos(texto_completo, arquivo.name)
        st.session_state.analise_riscos_resultados = resultados
    
    if 'analise_riscos_resultados' in st.session_state and st.session_state.analise_riscos_resultados:
        for nome_arquivo, analise in st.session_state.analise_riscos_resultados.items():
            with st.expander(f"Riscos Identificados em: {nome_arquivo}", expanded=True):
                st.markdown(analise)

def render_prazos_tab(arquivos_originais):
    """Renderiza a aba de Monitoramento de Prazos."""
    st.header("🗓️ Monitoramento de Prazos e Vencimentos")
    if not arquivos_originais:
        st.info("A extração de prazos é otimizada para novos uploads.")
        return

    if st.button("🔍 Analisar Prazos e Datas Importantes", key="btn_prazos", use_container_width=True):
        textos_docs = [{"nome": arq.name, "texto": _get_full_text(arq)} for arq in arquivos_originais]
        eventos_extraidos = extrair_eventos_dos_contratos(textos_docs)
        if eventos_extraidos:
            df = pd.DataFrame(eventos_extraidos)
            # Tentar converter datas - isso pode precisar de uma função de parsing mais robusta
            # df['Data Objeto'] = pd.to_datetime(df['Data Informada'], errors='coerce')
            st.session_state.eventos_contratuais_df = df
    
    if 'eventos_contratuais_df' in st.session_state and not st.session_state.eventos_contratuais_df.empty:
        st.dataframe(st.session_state.eventos_contratuais_df, use_container_width=True)

def render_conformidade_tab(arquivos_originais):
    """Renderiza a aba de Verificação de Conformidade."""
    st.header("⚖️ Verificador de Conformidade Contratual")
    if not arquivos_originais or len(arquivos_originais) < 2:
        st.info("Faça o upload de pelo menos dois documentos para usar esta função.")
        return

    nomes_arquivos = [f.name for f in arquivos_originais]
    col1, col2 = st.columns(2)
    with col1:
        doc_ref_nome = st.selectbox("Documento de Referência:", nomes_arquivos, key="ref_conf", index=None)
    with col2:
        doc_ana_nome = st.selectbox("Documento a Analisar:", [n for n in nomes_arquivos if n != doc_ref_nome], key="ana_conf", index=None)

    if st.button("🔎 Verificar Conformidade", key="btn_conf", use_container_width=True, disabled=not (doc_ref_nome and doc_ana_nome)):
        doc_ref_obj = next(f for f in arquivos_originais if f.name == doc_ref_nome)
        doc_ana_obj = next(f for f in arquivos_originais if f.name == doc_ana_nome)
        
        texto_ref = _get_full_text(doc_ref_obj)
        texto_ana = _get_full_text(doc_ana_obj)

        if texto_ref and texto_ana:
            resultado = verificar_conformidade_documento(texto_ref, doc_ref_nome, texto_ana, doc_ana_nome)
            st.session_state.conformidade_resultados = resultado
            
    if 'conformidade_resultados' in st.session_state and st.session_state.conformidade_resultados:
        st.markdown("---")
        st.subheader("Relatório de Conformidade")
        st.markdown(st.session_state.conformidade_resultados)


def render_anomalias_tab():
    """Renderiza a aba de Detecção de Anomalias."""
    st.header("📊 Detecção de Anomalias Contratuais")
    
    if 'df_dashboard' not in st.session_state or st.session_state.df_dashboard.empty:
        st.warning("Os dados para análise ainda não foram gerados. Vá para a aba '📈 Dashboard' e gere os dados primeiro.")
        return

    if st.button("🚨 Detectar Anomalias Agora", key="btn_anomalias", use_container_width=True):
        resultados = detectar_anomalias_no_dataframe(st.session_state.df_dashboard)
        st.session_state.anomalias_resultados = resultados

    if 'anomalias_resultados' in st.session_state and st.session_state.anomalias_resultados:
        st.subheader("Resultados da Detecção de Anomalias:")
        for item in st.session_state.anomalias_resultados:
            st.markdown(f"- {item}")
