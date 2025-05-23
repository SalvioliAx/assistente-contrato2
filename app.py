import streamlit as st
import os
import pandas as pd
from typing import Optional, List
import re
from datetime import datetime, date
import json
from pathlib import Path
import numpy as np # Para cálculos estatísticos

# Importações do LangChain e Pydantic
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.messages import AIMessage

# --- DEFINIÇÕES GLOBAIS ---
COLECOES_DIR = Path("colecoes_ia")
COLECOES_DIR.mkdir(exist_ok=True)

# --- SCHEMAS DE DADOS ---
# ATUALIZAÇÃO DO InfoContrato para incluir campos mais quantificáveis/categorizáveis
class InfoContrato(BaseModel):
    arquivo_fonte: str = Field(description="O nome do arquivo de origem do contrato.")
    nome_banco_emissor: Optional[str] = Field(default="Não encontrado", description="O nome do banco ou instituição financeira principal mencionada.")
    
    # Campos para Detecção de Anomalias (Exemplos - Adapte à sua necessidade!)
    valor_principal_numerico: Optional[float] = Field(default=None, description="Se houver um valor monetário principal claramente definido no contrato (ex: valor total do contrato, valor do empréstimo, limite de crédito principal), extraia apenas o número. Caso contrário, deixe como não encontrado.")
    prazo_total_meses: Optional[int] = Field(default=None, description="Se houver um prazo de vigência total do contrato claramente definido em meses ou anos, converta para meses e extraia apenas o número. Caso contrário, deixe como não encontrado.")
    taxa_juros_anual_numerica: Optional[float] = Field(default=None, description="Se uma taxa de juros principal (anual ou claramente conversível para anual) for mencionada, extraia apenas o número percentual. Caso contrário, deixe como não encontrado.")
    possui_clausula_rescisao_multa: Optional[str] = Field(default="Não claro", description="O contrato menciona explicitamente uma multa em caso de rescisão? Responda 'Sim', 'Não', ou 'Não claro'.")
    
    # Campos de política (mantidos do dashboard anterior)
    condicao_limite_credito: Optional[str] = Field(default="Não encontrado", description="Resumo da política de como o limite de crédito é definido, analisado e alterado.")
    condicao_juros_rotativo: Optional[str] = Field(default="Não encontrado", description="Resumo da regra de como e quando os juros do crédito rotativo são aplicados.")
    condicao_anuidade: Optional[str] = Field(default="Não encontrado", description="Resumo da política de cobrança da anuidade.")
    condicao_cancelamento: Optional[str] = Field(default="Não encontrado", description="Resumo das condições para cancelamento do contrato.")


class EventoContratual(BaseModel):
    # ... (schema como antes)
    descricao_evento: str = Field(description="Uma descrição clara e concisa do evento ou prazo.")
    data_evento_str: Optional[str] = Field(default="Não Especificado", description="A data do evento no formato YYYY-MM-DD. Se não aplicável, use 'Não Especificado'.")
    trecho_relevante: Optional[str] = Field(default=None, description="O trecho exato do contrato que menciona este evento/data.")

class ListaDeEventos(BaseModel):
    # ... (schema como antes)
    eventos: List[EventoContratual] = Field(description="Lista de eventos contratuais com suas datas.")
    arquivo_fonte: str = Field(description="O nome do arquivo de origem do contrato de onde estes eventos foram extraídos.")

# --- CONFIGURAÇÃO DA PÁGINA E DA CHAVE DE API ---
st.set_page_config(layout="wide", page_title="Analisador-IA ProMax", page_icon="💡") # Novo ícone
# ... (resto da configuração como antes) ...
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = google_api_key
except (KeyError, FileNotFoundError):
    st.sidebar.warning("Chave de API do Google não configurada nos Secrets.")
    google_api_key = st.sidebar.text_input("(OU) Cole sua Chave de API do Google aqui:", type="password", key="api_key_input_main")
    if google_api_key: os.environ["GOOGLE_API_KEY"] = google_api_key
    else: google_api_key = None
hide_streamlit_style = "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>"
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- FUNÇÕES DE GERENCIAMENTO DE COLEÇÕES ---
# (Funções listar_colecoes_salvas, salvar_colecao_atual, carregar_colecao permanecem as mesmas)
def listar_colecoes_salvas():
    if not COLECOES_DIR.exists(): return []
    return [d.name for d in COLECOES_DIR.iterdir() if d.is_dir()]
def salvar_colecao_atual(nome_colecao, vector_store_atual, nomes_arquivos_atuais):
    if not nome_colecao.strip(): st.error("Por favor, forneça um nome para a coleção."); return False
    caminho_colecao = COLECOES_DIR / nome_colecao
    try:
        caminho_colecao.mkdir(parents=True, exist_ok=True)
        vector_store_atual.save_local(str(caminho_colecao / "faiss_index"))
        with open(caminho_colecao / "manifest.json", "w") as f: json.dump(nomes_arquivos_atuais, f)
        st.success(f"Coleção '{nome_colecao}' salva com sucesso!"); return True
    except Exception as e: st.error(f"Erro ao salvar coleção: {e}"); return False
@st.cache_resource(show_spinner="Carregando coleção...")
def carregar_colecao(nome_colecao, _embeddings_obj):
    caminho_colecao = COLECOES_DIR / nome_colecao; caminho_indice = caminho_colecao / "faiss_index"; caminho_manifesto = caminho_colecao / "manifest.json"
    if not caminho_indice.exists() or not caminho_manifesto.exists(): st.error(f"Coleção '{nome_colecao}' incompleta."); return None, None
    try:
        vector_store = FAISS.load_local(str(caminho_indice), embeddings=_embeddings_obj, allow_dangerous_deserialization=True)
        with open(caminho_manifesto, "r") as f: nomes_arquivos = json.load(f)
        st.success(f"Coleção '{nome_colecao}' carregada!"); return vector_store, nomes_arquivos
    except Exception as e: st.error(f"Erro ao carregar coleção '{nome_colecao}': {e}"); return None, None

# --- FUNÇÕES DE PROCESSAMENTO DE DOCUMENTOS ---
@st.cache_resource(show_spinner="Analisando documentos para busca e chat...")
def obter_vector_store_de_uploads(lista_arquivos_pdf_upload, _embeddings_obj):
    if not lista_arquivos_pdf_upload or not google_api_key or not _embeddings_obj : return None, None
    documentos_totais = [];
    for arquivo_pdf in lista_arquivos_pdf_upload:
        temp_file_path = Path(f"temp_{arquivo_pdf.name}")
        with open(temp_file_path, "wb") as f: f.write(arquivo_pdf.getbuffer())
        loader = PyPDFLoader(str(temp_file_path)); pages = loader.load()
        for page in pages: page.metadata["source"] = arquivo_pdf.name
        documentos_totais.extend(pages); os.remove(temp_file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs_fragmentados = text_splitter.split_documents(documentos_totais)
    vector_store = FAISS.from_documents(docs_fragmentados, _embeddings_obj)
    return vector_store, [f.name for f in lista_arquivos_pdf_upload]

# ATUALIZADA para o novo InfoContrato e com tentativa de extrair números
@st.cache_data(show_spinner="Extraindo dados detalhados dos contratos...")
def extrair_dados_dos_contratos(_vector_store: FAISS, _nomes_arquivos: list) -> list:
    if not _vector_store or not google_api_key or not _nomes_arquivos: return []
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    
    resultados_finais = []
    barra_progresso = st.progress(0, text="Iniciando extração detalhada...")

    # Mapa de campos para perguntas mais diretas, focando nos novos campos
    mapa_campos_perguntas = {
        "nome_banco_emissor": "Qual o nome principal do banco, instituição financeira ou empresa emissora deste contrato?",
        "valor_principal_numerico": "Qual o valor monetário principal ou limite de crédito central deste contrato? Forneça apenas o número.",
        "prazo_total_meses": "Qual o prazo de vigência total deste contrato em meses? Se estiver em anos, converta para meses. Forneça apenas o número.",
        "taxa_juros_anual_numerica": "Qual a principal taxa de juros anual (ou facilmente conversível para anual) mencionada? Forneça apenas o número percentual.",
        "possui_clausula_rescisao_multa": "Este contrato menciona explicitamente uma multa em caso de rescisão? Responda 'Sim', 'Não', ou 'Não claro'.",
        "condicao_limite_credito": "Qual é a política ou condição para definir o limite de crédito?",
        "condicao_juros_rotativo": "Sob quais condições os juros do crédito rotativo são aplicados?",
        "condicao_anuidade": "Qual é a política de cobrança da anuidade descrita no contrato?",
        "condicao_cancelamento": "Quais são as regras para o cancelamento ou rescisão do contrato?"
    }

    for i, nome_arquivo in enumerate(_nomes_arquivos):
        dados_contrato_atual = {"arquivo_fonte": nome_arquivo}
        retriever_arquivo_atual = _vector_store.as_retriever(search_kwargs={'filter': {'source': nome_arquivo}, 'k': 3}) # Menos chunks para maior precisão na extração de valores
        
        for campo, pergunta_chave in mapa_campos_perguntas.items():
            barra_progresso.progress((i + (list(mapa_campos_perguntas.keys()).index(campo) / len(mapa_campos_perguntas))) / len(_nomes_arquivos),
                                     text=f"Extraindo '{campo}' de {nome_arquivo}")
            
            docs_relevantes = retriever_arquivo_atual.get_relevant_documents(pergunta_chave)
            contexto = "\n\n".join([doc.page_content for doc in docs_relevantes])

            prompt_extracao_especifica = PromptTemplate.from_template(
                "Com base no contexto fornecido, responda à seguinte pergunta de forma concisa. {instrucao_adicional}\n\n"
                "Contexto:\n{contexto}\n\n"
                "Pergunta: {pergunta}\n"
                "Resposta:"
            )
            chain_extracao = LLMChain(llm=llm, prompt=prompt_extracao_especifica)
            
            instrucao_adicional = ""
            if "numerico" in campo or "meses" in campo:
                instrucao_adicional = "Se encontrar um valor, forneça apenas o número (ex: 10.5 ou 12). Se não encontrar, responda 'Não encontrado'."
            elif "possui_clausula" in campo:
                instrucao_adicional = "Responda apenas com 'Sim', 'Não', ou 'Não claro'."
            else: # campos de condição (texto)
                 instrucao_adicional = "Resuma a política em uma ou duas frases. Se não encontrar, responda 'Não encontrado'."


            if contexto:
                try:
                    resultado = chain_extracao.invoke({
                        "instrucao_adicional": instrucao_adicional,
                        "contexto": contexto, 
                        "pergunta": pergunta_chave
                    })
                    resposta = resultado['text'].strip()

                    # Limpeza e conversão para campos numéricos
                    if campo in ["valor_principal_numerico", "prazo_total_meses", "taxa_juros_anual_numerica"]:
                        numeros = re.findall(r"[\d\.,]+", resposta)
                        if numeros:
                            try:
                                valor_limpo = numeros[0].replace('.', '').replace(',', '.')
                                dados_contrato_atual[campo] = float(valor_limpo) if "." in valor_limpo else int(valor_limpo)
                            except ValueError:
                                dados_contrato_atual[campo] = None # Não conseguiu converter
                        else:
                            dados_contrato_atual[campo] = None # Não encontrou número
                    elif campo == "possui_clausula_rescisao_multa":
                        if "sim" in resposta.lower(): dados_contrato_atual[campo] = "Sim"
                        elif "não" in resposta.lower() or "nao" in resposta.lower() : dados_contrato_atual[campo] = "Não"
                        else: dados_contrato_atual[campo] = "Não claro"
                    else:
                        dados_contrato_atual[campo] = resposta if "não encontrado" not in resposta.lower() else "Não encontrado"

                except Exception as e_invoke:
                    st.warning(f"Erro ao invocar LLM para '{campo}' em {nome_arquivo}: {e_invoke}")
                    dados_contrato_atual[campo] = None if "numerico" in campo or "meses" in campo else "Erro na IA"
            else:
                dados_contrato_atual[campo] = None if "numerico" in campo or "meses" in campo else "Contexto não encontrado"
        
        # Validação com Pydantic antes de adicionar
        try:
            info_validada = InfoContrato(**dados_contrato_atual)
            resultados_finais.append(info_validada.dict())
        except Exception as e_pydantic:
            st.error(f"Erro de validação Pydantic para {nome_arquivo}: {e_pydantic}. Dados: {dados_contrato_atual}")
            # Adiciona um registro de erro, preenchendo com defaults para manter a estrutura do DataFrame
            resultados_finais.append(InfoContrato(arquivo_fonte=nome_arquivo).dict())


    barra_progresso.empty()
    st.success("Extração detalhada para dashboard e anomalias concluída!")
    return resultados_finais

# --- NOVA FUNÇÃO PARA DETECÇÃO DE ANOMALIAS ---
def detectar_anomalias_no_dataframe(df: pd.DataFrame) -> List[str]:
    anomalias_encontradas = []
    if df.empty:
        return ["Nenhum dado para analisar anomalias."]

    # Campos numéricos para análise de outliers
    campos_numericos = ["valor_principal_numerico", "prazo_total_meses", "taxa_juros_anual_numerica"]
    for campo in campos_numericos:
        if campo in df.columns:
            # Converte para numérico, erros viram NaN, depois remove NaNs
            serie = pd.to_numeric(df[campo], errors='coerce').dropna()
            if not serie.empty and len(serie) > 1: # Precisa de pelo menos 2 pontos para desvio padrão
                media = serie.mean()
                desvio_pad = serie.std()
                limite_superior = media + 2 * desvio_pad
                limite_inferior = media - 2 * desvio_pad

                # Identifica outliers
                outliers = df[
                    (pd.to_numeric(df[campo], errors='coerce') > limite_superior) |
                    (pd.to_numeric(df[campo], errors='coerce') < limite_inferior)
                ]
                for _, linha in outliers.iterrows():
                    anomalias_encontradas.append(
                        f"**Anomalia Numérica em `{linha['arquivo_fonte']}`:** Campo '{campo}' com valor `{linha[campo]}` "
                        f"está distante da média ({media:.2f} ± {2*desvio_pad:.2f})."
                    )
            elif len(serie) == 1:
                 anomalias_encontradas.append(f"**Info:** Campo '{campo}' possui apenas um valor numérico (`{serie.iloc[0]}`), não sendo possível análise de desvio.")


    # Campos categóricos para análise de frequência
    campos_categoricos = ["possui_clausula_rescisao_multa", "nome_banco_emissor"] # Adicionar outros se relevante
    for campo in campos_categoricos:
        if campo in df.columns:
            contagem_valores = df[campo].value_counts(normalize=True) # normalize=True para obter percentuais
            # Considera anômalo se uma categoria aparece em menos de 10% dos casos (e há mais de 5 contratos)
            if len(df) > 5:
                categorias_raras = contagem_valores[contagem_valores < 0.1]
                for categoria, freq in categorias_raras.items():
                    documentos_com_categoria_rara = df[df[campo] == categoria]['arquivo_fonte'].tolist()
                    anomalias_encontradas.append(
                        f"**Anomalia Categórica:** O valor/categoria '`{categoria}`' para o campo '{campo}' "
                        f"é incomum (presente em {freq*100:.1f}% dos contratos: {', '.join(documentos_com_categoria_rara[:3])}{'...' if len(documentos_com_categoria_rara) > 3 else ''})."
                    )
    
    if not anomalias_encontradas:
        return ["Nenhuma anomalia significativa detectada com os critérios atuais."]
    return anomalias_encontradas


# ... (gerar_resumo_executivo, analisar_documento_para_riscos, extrair_eventos_dos_contratos, formatar_chat_para_markdown permanecem os mesmos)
@st.cache_data(show_spinner="Gerando resumo executivo...")
def gerar_resumo_executivo(arquivo_pdf_bytes, nome_arquivo_original):
    if not arquivo_pdf_bytes or not google_api_key: return "Erro: Arquivo ou chave de API não fornecidos."
    temp_file_path = Path(f"temp_resumo_{nome_arquivo_original}")
    with open(temp_file_path, "wb") as f: f.write(arquivo_pdf_bytes)
    loader = PyPDFLoader(str(temp_file_path)); documento_completo_paginas = loader.load(); os.remove(temp_file_path)
    texto_completo = "\n\n".join([page.page_content for page in documento_completo_paginas])
    llm_resumo = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    template_prompt_resumo = PromptTemplate.from_template(
        "Você é um assistente especializado em analisar e resumir documentos jurídicos, como contratos.\n"
        "Com base no texto do contrato fornecido abaixo, crie um resumo executivo em 5 a 7 tópicos concisos (bullet points).\n"
        "Destaque os seguintes aspectos, se presentes: as partes principais envolvidas, o objeto principal do contrato, "
        "prazo de vigência (se houver), principais obrigações financeiras ou condições de pagamento, e as "
        "principais condições ou motivos para rescisão ou cancelamento do contrato.\n"
        "Seja claro e direto.\n\nTEXTO DO CONTRATO:\n{texto_contrato}\n\nRESUMO EXECUTIVO:")
    chain_resumo = LLMChain(llm=llm_resumo, prompt=template_prompt_resumo)
    try: resultado = chain_resumo.invoke({"texto_contrato": texto_completo}); return resultado['text']
    except Exception as e: return f"Erro ao gerar resumo: {e}"

@st.cache_data(show_spinner="Analisando riscos no documento...")
def analisar_documento_para_riscos(texto_completo_doc, nome_arquivo_doc):
    if not texto_completo_doc or not google_api_key: return f"Não foi possível analisar riscos para '{nome_arquivo_doc}': Texto ou Chave API ausente."
    llm_riscos = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
    prompt_riscos_template = PromptTemplate.from_template(
        "Você é um advogado especialista em análise de riscos contratuais. "
        "Analise o texto do contrato fornecido abaixo e identifique cláusulas ou omissões que possam representar riscos significativos. "
        "Para cada risco identificado, por favor:\n1. Descreva o risco de forma clara e concisa.\n"
        "2. Cite o trecho exato da cláusula relevante (ou mencione a ausência de uma cláusula esperada).\n"
        "3. Classifique o risco (ex: Financeiro, Operacional, Legal, Rescisão, Propriedade Intelectual, Confidencialidade, etc.).\n"
        "Concentre-se nos riscos mais impactantes. Se nenhum risco significativo for encontrado, declare isso explicitamente.\n"
        "Use formatação Markdown para sua resposta, com um título para cada risco.\n\n"
        "TEXTO DO CONTRATO ({nome_arquivo}):\n{texto_contrato}\n\nANÁLISE DE RISCOS:")
    chain_riscos = LLMChain(llm=llm_riscos, prompt=prompt_riscos_template)
    try: resultado = chain_riscos.invoke({"nome_arquivo": nome_arquivo_doc, "texto_contrato": texto_completo_doc}); return resultado['text']
    except Exception as e: return f"Erro ao analisar riscos para '{nome_arquivo_doc}': {e}"

@st.cache_data(show_spinner="Extraindo datas e prazos dos contratos...")
def extrair_eventos_dos_contratos(textos_completos_docs: List[dict]) -> List[dict]:
    if not textos_completos_docs or not google_api_key: return []
    llm_eventos = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, request_timeout=120)
    parser = PydanticOutputParser(pydantic_object=ListaDeEventos)
    prompt_eventos_template_str = """Analise o texto do contrato abaixo. Sua tarefa é identificar TODOS os eventos, datas, prazos e períodos importantes mencionados.
Para cada evento encontrado, extraia as seguintes informações:
1.  'descricao_evento': Uma descrição clara e concisa do evento (ex: 'Data de assinatura do contrato', 'Vencimento da primeira parcela', 'Prazo final para entrega do produto', 'Início da vigência', 'Período de carência para alteração de vencimento').
2.  'data_evento_str': A data específica do evento no formato<y_bin_46>MM-DD. Se uma data EXATA não puder ser determinada ou não se aplicar (ex: '10 dias antes do vencimento', 'prazo indeterminado', 'na fatura mensal'), preencha este campo OBRIGATORIAMENTE com la string 'Não Especificado'. NUNCA use null, None ou deixe o campo vazio.
3.  'trecho_relevante': O trecho curto e exato do contrato que menciona este evento/data.

{format_instructions}

TEXTO DO CONTRATO ({arquivo_fonte}):
{texto_contrato}

ATENÇÃO: O campo 'data_evento_str' DEVE SEMPRE ser uma string. Se não houver data específica, use 'Não Especificado'.
LISTA DE EVENTOS ENCONTRADOS:"""
    prompt_eventos = PromptTemplate(
        template=prompt_eventos_template_str,
        input_variables=["texto_contrato", "arquivo_fonte"],
        partial_variables={"format_instructions": parser.get_format_instructions().replace("```json", "").replace("```", "").strip()}
    )
    output_fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0))
    chain_eventos_llm_only = prompt_eventos | llm_eventos
    todos_os_eventos_formatados = []
    barra_progresso = st.progress(0, text="Iniciando extração de datas...")
    for i, doc_info in enumerate(textos_completos_docs):
        nome_arquivo, texto_contrato = doc_info["nome"], doc_info["texto"]
        barra_progresso.progress((i + 1) / len(textos_completos_docs), text=f"Analisando datas em: {nome_arquivo}")
        try:
            resposta_ia_obj = chain_eventos_llm_only.invoke({"texto_contrato": texto_contrato, "arquivo_fonte": nome_arquivo})
            resposta_ia_str = resposta_ia_obj.content
            try: resultado_parseado = parser.parse(resposta_ia_str)
            except Exception as e_parse:
                st.write(f"Parser Pydantic inicial falhou para {nome_arquivo}, tentando com OutputFixingParser. Erro: {e_parse}")
                st.write(f"Resposta da IA que causou o erro (primeiros 500 chars): {resposta_ia_str[:500]}...")
                resultado_parseado = output_fixing_parser.parse(resposta_ia_str)
            if resultado_parseado and isinstance(resultado_parseado, ListaDeEventos):
                for evento in resultado_parseado.eventos:
                    data_obj = None
                    if evento.data_evento_str and evento.data_evento_str.lower() not in ["não especificado", "condicional", "vide fatura", "n/a", ""]:
                        try: data_obj = datetime.strptime(evento.data_evento_str, "%Y-%m-%d").date()
                        except ValueError:
                            try: data_obj = datetime.strptime(evento.data_evento_str, "%d/%m/%Y").date()
                            except ValueError: pass
                    todos_os_eventos_formatados.append({
                        "Arquivo Fonte": nome_arquivo, "Evento": evento.descricao_evento,
                        "Data Informada": evento.data_evento_str, "Data Objeto": data_obj,
                        "Trecho Relevante": evento.trecho_relevante})
        except Exception as e_main:
            st.warning(f"Erro crítico ao processar datas para '{nome_arquivo}'. Erro: {e_main}")
            todos_os_eventos_formatados.append({
                "Arquivo Fonte": nome_arquivo, "Evento": f"Falha na extração: {e_main}", 
                "Data Informada": "Erro", "Data Objeto": None, "Trecho Relevante": None})
    barra_progresso.empty()
    if not todos_os_eventos_formatados: st.info("Nenhum evento ou prazo foi extraído dos documentos.")
    else: st.success("Extração de datas e prazos concluída!")
    return todos_os_eventos_formatados

@st.cache_data(show_spinner="Verificando conformidade do documento...")
def verificar_conformidade_documento(texto_doc_referencia, nome_doc_referencia, texto_doc_analisar, nome_doc_analisar):
    # (sem alterações)
    if not texto_doc_referencia or not texto_doc_analisar or not google_api_key: return "Erro: Textos dos documentos ou Chave API ausentes para verificação de conformidade."
    # ... (código da função como antes) ...
    llm_conformidade = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1, request_timeout=180)
    prompt_conformidade_template = PromptTemplate.from_template(
        "Você é um especialista em conformidade e auditoria contratual. Sua tarefa é comparar o 'DOCUMENTO A ANALISAR' com o 'DOCUMENTO DE REFERÊNCIA' (ex: um Código de Ética ou Política Interna).\n\n"
        "DOCUMENTO DE REFERÊNCIA ({nome_doc_referencia}):\n--------------------------------------------\n{texto_doc_referencia}\n--------------------------------------------\n\n"
        "DOCUMENTO A ANALISAR ({nome_doc_analisar}):\n--------------------------------------------\n{texto_doc_analisar}\n--------------------------------------------\n\n"
        "Por favor, identifique e liste quaisquer cláusulas, termos ou omissões significativas no 'DOCUMENTO A ANALISAR' que possam:\n"
        "1. Contradizer diretamente os princípios ou regras estabelecidas no 'DOCUMENTO DE REFERÊNCIA'.\n"
        "2. Estar em desalinhamento ético ou de conduta com o 'DOCUMENTO DE REFERÊNCIA'.\n"
        "3. Representar um risco de não conformidade com as diretrizes do 'DOCUMENTO DE REFERÊNCIA'.\n\n"
        "Para cada ponto de não conformidade ou potencial conflito identificado:\n"
        "a. Descreva o problema/conflito de forma clara e objetiva.\n"
        "b. Cite o trecho exato (ou o número da cláusula, se aplicável) do 'DOCUMENTO A ANALISAR' que levanta a questão.\n"
        "c. Cite o trecho exato ou princípio (ou o número da cláusula/item, se aplicável) do 'DOCUMENTO DE REFERÊNCIA' que está sendo potencialmente violado ou que serve de base para a comparação.\n"
        "d. Ofereça uma breve explicação sobre o motivo do conflito ou desalinhamento.\n\n"
        "Se nenhum conflito ou ponto de não conformidade significativo for encontrado, declare isso explicitamente como 'Nenhum conflito de conformidade significativo identificado'.\n"
        "Use formatação Markdown para sua resposta, organizando os pontos claramente. Use títulos (###) para cada conflito encontrado.\n\n"
        "RELATÓRIO DE ANÁLISE DE CONFORMIDADE:")
    chain_conformidade = LLMChain(llm=llm_conformidade, prompt=prompt_conformidade_template)
    try:
        resultado = chain_conformidade.invoke({
            "nome_doc_referencia": nome_doc_referencia, "texto_doc_referencia": texto_doc_referencia,
            "nome_doc_analisar": nome_doc_analisar, "texto_doc_analisar": texto_doc_analisar})
        return resultado['text']
    except Exception as e: return f"Erro ao gerar análise de conformidade para '{nome_doc_analisar}' vs '{nome_doc_referencia}': {e}"

def formatar_chat_para_markdown(mensagens_chat):
    # (sem alterações)
    texto_formatado = "# Histórico da Conversa com Analisador-IA\n\n"
    # ... (código da função como antes) ...
    for mensagem in mensagens_chat:
        if mensagem["role"] == "user": texto_formatado += f"## Você:\n{mensagem['content']}\n\n"
        elif mensagem["role"] == "assistant":
            texto_formatado += f"## IA:\n{mensagem['content']}\n"
            if "sources" in mensagem and mensagem["sources"]:
                texto_formatado += "### Fontes Utilizadas:\n"
                for i, doc_fonte in enumerate(mensagem["sources"]):
                    texto_fonte_original = doc_fonte.page_content; sentenca_chave = mensagem.get("sentenca_chave")
                    texto_fonte_md = texto_fonte_original.replace('\n', '  \n')
                    if sentenca_chave and sentenca_chave in texto_fonte_original:
                        texto_formatado_fonte = texto_fonte_md.replace(sentenca_chave, f"**{sentenca_chave}**")
                    else: texto_formatado_fonte = texto_fonte_md
                    texto_formatado += f"- **Fonte {i+1} (Doc: `{doc_fonte.metadata.get('source', 'N/A')}`, Pág: {doc_fonte.metadata.get('page', 'N/A')})**:\n  > {texto_formatado_fonte[:300]}...\n\n"
            texto_formatado += "---\n\n"
    return texto_formatado

# --- INICIALIZAÇÃO DO OBJETO DE EMBEDDINGS ---
if google_api_key:
    embeddings_global = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
else:
    embeddings_global = None

# --- LAYOUT PRINCIPAL E SIDEBAR ---
st.title("💡 Analisador-IA ProMax")
st.sidebar.header("Gerenciar Documentos")
# ... (sidebar como antes) ...
modo_documento = st.sidebar.radio("Como carregar os documentos?", ("Fazer novo upload de PDFs", "Carregar coleção existente"), key="modo_doc_radio")
arquivos_pdf_upload_sidebar = None
if modo_documento == "Fazer novo upload de PDFs":
    arquivos_pdf_upload_sidebar = st.sidebar.file_uploader("Selecione um ou mais contratos em PDF", type="pdf", accept_multiple_files=True, key="uploader_sidebar")
    if arquivos_pdf_upload_sidebar:
        if st.sidebar.button("Processar Documentos Carregados", key="btn_proc_upload_sidebar"):
            if google_api_key and embeddings_global:
                st.session_state.vector_store, st.session_state.nomes_arquivos = obter_vector_store_de_uploads(arquivos_pdf_upload_sidebar, embeddings_global)
                st.session_state.arquivos_pdf_originais = arquivos_pdf_upload_sidebar
                st.session_state.colecao_ativa = None; st.session_state.messages = []
                st.session_state.pop('df_dashboard', None); st.session_state.pop('resumo_gerado', None)
                st.session_state.pop('analise_riscos_resultados', None); st.session_state.pop('eventos_contratuais_df', None)
                st.session_state.pop('conformidade_resultados', None)
                st.session_state.pop('anomalias_resultados', None) # Limpa anomalias
                st.rerun()
            else: st.sidebar.error("Chave de API ou Embeddings não configurados.")
elif modo_documento == "Carregar coleção existente":
    colecoes_disponiveis = listar_colecoes_salvas()
    if colecoes_disponiveis:
        colecao_selecionada = st.sidebar.selectbox("Escolha uma coleção:", colecoes_disponiveis, key="select_colecao_sidebar")
        if st.sidebar.button("Carregar Coleção Selecionada", key="btn_load_colecao_sidebar"):
            if google_api_key and embeddings_global:
                vs, nomes_arqs = carregar_colecao(colecao_selecionada, embeddings_global)
                if vs and nomes_arqs:
                    st.session_state.vector_store, st.session_state.nomes_arquivos, st.session_state.colecao_ativa = vs, nomes_arqs, colecao_selecionada
                    st.session_state.arquivos_pdf_originais = None; st.session_state.messages = []
                    st.session_state.pop('df_dashboard', None); st.session_state.pop('resumo_gerado', None)
                    st.session_state.pop('analise_riscos_resultados', None); st.session_state.pop('eventos_contratuais_df', None)
                    st.session_state.pop('conformidade_resultados', None)
                    st.session_state.pop('anomalias_resultados', None) # Limpa anomalias
                    st.rerun()
            else: st.sidebar.error("Chave de API ou Embeddings não configurados.")
    else: st.sidebar.info("Nenhuma coleção salva ainda.")

if "vector_store" in st.session_state and st.session_state.vector_store is not None and st.session_state.get("arquivos_pdf_originais"):
    st.sidebar.markdown("---"); st.sidebar.subheader("Salvar Coleção Atual")
    nome_nova_colecao = st.sidebar.text_input("Nome para a nova coleção:", key="input_nome_colecao_sidebar")
    if st.sidebar.button("Salvar Coleção", key="btn_save_colecao_sidebar"):
        if nome_nova_colecao and st.session_state.nomes_arquivos: salvar_colecao_atual(nome_nova_colecao, st.session_state.vector_store, st.session_state.nomes_arquivos)
        else: st.sidebar.warning("Dê um nome e certifique-se de que há docs carregados.")

if "colecao_ativa" in st.session_state and st.session_state.colecao_ativa: st.sidebar.markdown(f"**Coleção Ativa:** `{st.session_state.colecao_ativa}`")
elif "nomes_arquivos" in st.session_state and st.session_state.nomes_arquivos: st.sidebar.markdown(f"**Arquivos Carregados:** {len(st.session_state.nomes_arquivos)}")

st.sidebar.header("Configurações de Idioma"); idioma_selecionado = st.sidebar.selectbox("Idioma para o CHAT:", ("Português", "Inglês", "Espanhol"), key="idioma_chat_key_sidebar")

# --- INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
if "messages" not in st.session_state: st.session_state.messages = []
if "resumo_gerado" not in st.session_state: st.session_state.resumo_gerado = ""
if "arquivo_resumido" not in st.session_state: st.session_state.arquivo_resumido = None
if "df_dashboard" not in st.session_state: st.session_state.df_dashboard = None
if "analise_riscos_resultados" not in st.session_state: st.session_state.analise_riscos_resultados = {}
if "eventos_contratuais_df" not in st.session_state: st.session_state.eventos_contratuais_df = None
if "conformidade_resultados" not in st.session_state: st.session_state.conformidade_resultados = {}
if "anomalias_resultados" not in st.session_state: st.session_state.anomalias_resultados = [] # Novo estado

# --- LÓGICA DAS ABAS ---
tab_chat, tab_dashboard, tab_resumo, tab_riscos, tab_prazos, tab_conformidade, tab_anomalias = st.tabs([
    "💬 Chat", "📈 Dashboard", "📜 Resumo", "🚩 Riscos", "🗓️ Prazos", "⚖️ Conformidade", "📊 Anomalias"
])
documentos_prontos = google_api_key and embeddings_global and (st.session_state.get("vector_store") is not None or st.session_state.get("arquivos_pdf_originais") is not None)

if not documentos_prontos:
    st.warning("Por favor, configure sua Chave de API do Google na barra lateral e carregue documentos para habilitar as funcionalidades.")
else:
    vector_store_global = st.session_state.get("vector_store")
    nomes_arquivos_global = st.session_state.get("nomes_arquivos", [])
    arquivos_pdf_originais_global = st.session_state.get("arquivos_pdf_originais")

    with tab_chat:
        # ... (código do chat completo como na versão anterior) ...
        st.header("Converse com seus documentos")
        if not vector_store_global: st.warning("Nenhum documento processado para o chat. Por favor, carregue documentos ou uma coleção.")
        else: 
            template_prompt_chat = PromptTemplate.from_template(
                """Use os seguintes trechos de contexto para responder à pergunta no final.
                INSTRUÇÕES DE FORMATAÇÃO DA RESPOSTA: Sua resposta final deve ter duas partes, separadas por '|||'.
                1. Parte 1: A resposta completa e detalhada para a pergunta do usuário, no idioma {language}.
                2. Parte 2: A citação exata e literal da sentença do contexto que foi mais importante para formular a resposta.
                CONTEXTO: {context}
                PERGUNTA: {question}
                RESPOSTA (seguindo o formato acima):""")
            if not st.session_state.messages : st.session_state.messages.append({"role": "assistant", "content": f"Olá! Documentos da coleção '{st.session_state.get('colecao_ativa', 'atual')}' prontos. Qual sua pergunta?"})
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "sources" in message:
                        with st.expander("Ver Fontes Utilizadas"):
                            for doc_fonte in message["sources"]:
                                texto_fonte = doc_fonte.page_content; sentenca_chave = message.get("sentenca_chave")
                                if sentenca_chave and sentenca_chave in texto_fonte: texto_formatado = texto_fonte.replace(sentenca_chave, f"<span style='background-color: #FFFACD; padding: 2px; border-radius: 3px;'>{sentenca_chave}</span>")
                                else: texto_formatado = texto_fonte
                                st.markdown(f"**Fonte: `{doc_fonte.metadata.get('source', 'N/A')}` (Página {doc_fonte.metadata.get('page', 'N/A')})**")
                                st.markdown(texto_formatado, unsafe_allow_html=True)
            if st.session_state.messages :
                chat_exportado_md = formatar_chat_para_markdown(st.session_state.messages)
                agora = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(label="📥 Exportar Conversa",data=chat_exportado_md, file_name=f"conversa_contratos_{agora}.md", mime="text/markdown", key="export_chat_btn_tab")
                st.markdown("---")
            if prompt := st.chat_input("Faça sua pergunta sobre os contratos..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Pesquisando..."):
                        llm_chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
                        qa_chain = RetrievalQA.from_chain_type(llm=llm_chat, chain_type="stuff", retriever=vector_store_global.as_retriever(search_kwargs={"k": 5}), return_source_documents=True, chain_type_kwargs={"prompt": template_prompt_chat.partial(language=idioma_selecionado)})
                        resultado = qa_chain({"query": prompt}); resposta_bruta = resultado["result"]; fontes = resultado["source_documents"]
                        try: resposta_principal, sentenca_chave = resposta_bruta.split('|||'); sentenca_chave = sentenca_chave.strip()
                        except ValueError: resposta_principal, sentenca_chave = resposta_bruta, None
                        st.markdown(resposta_principal)
                        st.session_state.messages.append({"role": "assistant", "content": resposta_principal, "sources": fontes, "sentenca_chave": sentenca_chave})
                        st.rerun()

    with tab_dashboard:
        st.header("Análise Comparativa de Políticas Contratuais")
        st.markdown("Clique no botão para extrair e comparar as políticas chave dos documentos carregados.")
        if not (vector_store_global and nomes_arquivos_global):
            st.warning("Carregue documentos ou uma coleção válida para usar o dashboard.")
        else:
            if st.button("🚀 Gerar Dados para Dashboard e Anomalias", key="btn_dashboard_e_anomalias_tab"):
                # Este botão agora vai popular o df_dashboard que será usado por esta aba e pela de anomalias
                dados_extraidos = extrair_dados_dos_contratos(vector_store_global, nomes_arquivos_global)
                if dados_extraidos: st.session_state.df_dashboard = pd.DataFrame(dados_extraidos)
                else: st.session_state.df_dashboard = pd.DataFrame() # Cria um DF vazio se nada for extraído
                st.session_state.pop('anomalias_resultados', None) # Limpa resultados de anomalias anteriores
            
            if 'df_dashboard' in st.session_state and st.session_state.df_dashboard is not None:
                if not st.session_state.df_dashboard.empty:
                    st.info("Tabela de dados extraídos dos contratos. Use a barra de rolagem horizontal.")
                    st.dataframe(st.session_state.df_dashboard)
                elif ("btn_dashboard_e_anomalias_tab" in st.session_state and st.session_state.btn_dashboard_e_anomalias_tab): 
                    st.warning("Nenhum dado foi extraído para o dashboard.")
            elif ("btn_dashboard_e_anomalias_tab" in st.session_state and st.session_state.btn_dashboard_e_anomalias_tab and st.session_state.df_dashboard is None) :
                 st.warning("A extração de dados para o dashboard não retornou resultados ou falhou.")
    
    with tab_resumo: # (Mantido como antes)
        # ...
        st.header("📜 Resumo Executivo de um Contrato")
        if arquivos_pdf_originais_global:
            lista_nomes_arquivos_resumo = [f.name for f in arquivos_pdf_originais_global]
            if lista_nomes_arquivos_resumo:
                arquivo_selecionado_nome_resumo = st.selectbox("Escolha um contrato para resumir:", options=lista_nomes_arquivos_resumo, key="select_resumo_tab")
                if st.button("✍️ Gerar Resumo Executivo", key="btn_resumo_tab"):
                    arquivo_obj_selecionado = next((arq for arq in arquivos_pdf_originais_global if arq.name == arquivo_selecionado_nome_resumo), None)
                    if arquivo_obj_selecionado:
                        resumo = gerar_resumo_executivo(arquivo_obj_selecionado.getvalue(), arquivo_obj_selecionado.name)
                        st.session_state.resumo_gerado = resumo; st.session_state.arquivo_resumido = arquivo_selecionado_nome_resumo
                    else: st.error("Arquivo selecionado não encontrado.")
                if st.session_state.get("arquivo_resumido") == arquivo_selecionado_nome_resumo and st.session_state.resumo_gerado:
                    st.subheader(f"Resumo do Contrato: {st.session_state.arquivo_resumido}"); st.markdown(st.session_state.resumo_gerado)
            else: st.info("Nenhum arquivo carregado disponível para resumo.")
        elif nomes_arquivos_global: st.info("A função de resumo funciona melhor com arquivos recém-carregados.")
        else: st.warning("Carregue documentos para usar a função de resumo.")


    with tab_riscos: # (Mantido como antes)
        # ...
        st.header("🚩 Análise de Cláusulas de Risco")
        st.markdown("Analisa os documentos carregados na sessão atual em busca de cláusulas potencialmente arriscadas.")
        if arquivos_pdf_originais_global:
            if st.button("🔎 Analisar Riscos em Todos os Documentos Carregados", key="btn_analise_riscos"):
                st.session_state.analise_riscos_resultados = {}
                textos_completos_docs = []
                for arquivo_pdf_obj in arquivos_pdf_originais_global:
                    temp_path = Path(f"temp_risco_{arquivo_pdf_obj.name}")
                    with open(temp_path, "wb") as f: f.write(arquivo_pdf_obj.getbuffer())
                    loader = PyPDFLoader(str(temp_path)); texto_doc = "\n\n".join([page.page_content for page in loader.load()]); textos_completos_docs.append({"nome": arquivo_pdf_obj.name, "texto": texto_doc}); os.remove(temp_path)
                resultados_analise = {}
                for doc_info in textos_completos_docs:
                    st.info(f"Analisando riscos em: {doc_info['nome']}...")
                    resultado_risco = analisar_documento_para_riscos(doc_info["texto"], doc_info["nome"])
                    resultados_analise[doc_info["nome"]] = resultado_risco
                st.session_state.analise_riscos_resultados = resultados_analise
            if st.session_state.analise_riscos_resultados:
                st.markdown("---")
                for nome_arquivo, analise in st.session_state.analise_riscos_resultados.items():
                    with st.expander(f"Riscos Identificados em: {nome_arquivo}", expanded=True): st.markdown(analise)
        elif "colecao_ativa" in st.session_state and st.session_state.colecao_ativa: st.warning("A Análise de Riscos detalhada funciona melhor com arquivos recém-carregados.")
        else: st.info("Faça o upload de documentos para ativar a análise de riscos.")


    with tab_prazos: # (Mantido como antes)
        # ...
        st.header("🗓️ Monitoramento de Prazos e Vencimentos")
        st.markdown("Extrai e organiza datas e prazos importantes dos documentos carregados na sessão atual.")
        if arquivos_pdf_originais_global:
            if st.button("🔍 Analisar Prazos e Datas Importantes", key="btn_analise_prazos"):
                textos_completos_para_datas = []
                for arquivo_pdf_obj in arquivos_pdf_originais_global:
                    temp_path = Path(f"temp_prazo_{arquivo_pdf_obj.name}")
                    with open(temp_path, "wb") as f: f.write(arquivo_pdf_obj.getbuffer())
                    loader = PyPDFLoader(str(temp_path)); texto_doc = "\n\n".join([page.page_content for page in loader.load()]); textos_completos_para_datas.append({"nome": arquivo_pdf_obj.name, "texto": texto_doc}); os.remove(temp_path)
                eventos_extraidos = extrair_eventos_dos_contratos(textos_completos_para_datas)
                if eventos_extraidos:
                    df_eventos = pd.DataFrame(eventos_extraidos)
                    df_eventos['Data Objeto'] = pd.to_datetime(df_eventos['Data Objeto'], errors='coerce')
                    st.session_state.eventos_contratuais_df = df_eventos.sort_values(by="Data Objeto", ascending=True, na_position='last')
                else: st.session_state.eventos_contratuais_df = pd.DataFrame()
            if 'eventos_contratuais_df' in st.session_state and st.session_state.eventos_contratuais_df is not None:
                df_display = st.session_state.eventos_contratuais_df.copy()
                if not df_display.empty:
                    if 'Data Objeto' in df_display.columns and df_display['Data Objeto'].notna().any():
                         df_display['Data Formatada'] = df_display['Data Objeto'].dt.strftime('%d/%m/%Y').fillna('N/A')
                    else: df_display['Data Formatada'] = df_display.get('Data Informada', pd.Series(['N/A'] * len(df_display)))
                    st.subheader("Todos os Eventos e Prazos Identificados")
                    colunas_para_exibir_eventos = ['Arquivo Fonte', 'Evento', 'Data Informada', 'Data Formatada', 'Trecho Relevante']
                    colunas_existentes_eventos = [col for col in colunas_para_exibir_eventos if col in df_display.columns]
                    st.dataframe(df_display[colunas_existentes_eventos], height=400)
                    if 'Data Objeto' in df_display.columns and df_display['Data Objeto'].notna().any():
                        st.subheader("Próximos Eventos (Próximos 90 dias)")
                        hoje_datetime = datetime.now()
                        df_display_com_datetime = df_display[df_display['Data Objeto'].notna()].copy()
                        if not df_display_com_datetime.empty:
                            proximos_eventos = df_display_com_datetime[
                                (df_display_com_datetime['Data Objeto'] >= hoje_datetime) &
                                (df_display_com_datetime['Data Objeto'] <= (hoje_datetime + pd.Timedelta(days=90)))]
                            if not proximos_eventos.empty: st.table(proximos_eventos[['Arquivo Fonte', 'Evento', 'Data Formatada']])
                            else: st.info("Nenhum evento encontrado para os próximos 90 dias.")
                        else: st.info("Nenhuma data válida encontrada para filtrar próximos eventos.")
                    else: st.warning("Coluna 'Data Objeto' não contém datas válidas para filtrar próximos eventos.")
                else: st.info("Nenhum evento ou prazo foi extraído dos documentos ou a extração falhou.")
            elif ("btn_analise_prazos" in st.session_state and st.session_state.btn_analise_prazos):
                 st.warning("A extração de datas não retornou resultados. Verifique os avisos na função de extração.")
        elif "colecao_ativa" in st.session_state and st.session_state.colecao_ativa: st.warning("O Monitoramento de Prazos funciona melhor com arquivos recém-carregados.")
        else: st.info("Faça o upload de documentos para ativar o monitoramento de prazos.")

    with tab_conformidade: # (Mantido como antes)
        # ... (código da verificação de conformidade completo)
        st.header("⚖️ Verificador de Conformidade Contratual")
        st.markdown("Compare um documento com um documento de referência para identificar desalinhamentos.")
        if arquivos_pdf_originais_global and len(arquivos_pdf_originais_global) >= 1:
            nomes_arquivos_para_selecao = [f.name for f in arquivos_pdf_originais_global]
            col_ref, col_ana = st.columns(2)
            with col_ref:
                doc_referencia_nome = st.selectbox("1. Documento de Referência:", options=nomes_arquivos_para_selecao, key="select_doc_ref_conf")
            opcoes_docs_analisar = [n for n in nomes_arquivos_para_selecao if n != doc_referencia_nome]
            if opcoes_docs_analisar:
                with col_ana:
                    docs_a_analisar_nomes = st.multiselect("2. Documento(s) a Analisar:", options=opcoes_docs_analisar, key="multiselect_docs_ana_conf")
            else: st.warning("Carregue pelo menos dois documentos para comparação.")

            if st.button("🔎 Verificar Conformidade", key="btn_ver_conf") and doc_referencia_nome and docs_a_analisar_nomes:
                st.session_state.conformidade_resultados = {}
                doc_referencia_obj = next((arq for arq in arquivos_pdf_originais_global if arq.name == doc_referencia_nome), None)
                texto_doc_referencia = ""
                if doc_referencia_obj:
                    temp_path_ref = Path(f"temp_conf_ref_{doc_referencia_obj.name}")
                    with open(temp_path_ref, "wb") as f: f.write(doc_referencia_obj.getbuffer())
                    loader_ref = PyPDFLoader(str(temp_path_ref)); texto_doc_referencia = "\n\n".join([page.page_content for page in loader_ref.load()]); os.remove(temp_path_ref)
                if not texto_doc_referencia: st.error(f"Não foi possível ler: {doc_referencia_nome}")
                else:
                    for nome_doc_analisar in docs_a_analisar_nomes:
                        doc_analisar_obj = next((arq for arq in arquivos_pdf_originais_global if arq.name == nome_doc_analisar), None)
                        if doc_analisar_obj:
                            temp_path_ana = Path(f"temp_conf_ana_{doc_analisar_obj.name}")
                            with open(temp_path_ana, "wb") as f: f.write(doc_analisar_obj.getbuffer())
                            loader_ana = PyPDFLoader(str(temp_path_ana)); texto_doc_analisar = "\n\n".join([page.page_content for page in loader_ana.load()]); os.remove(temp_path_ana)
                            if texto_doc_analisar:
                                st.info(f"Analisando conformidade de '{nome_doc_analisar}' vs '{doc_referencia_nome}'...")
                                resultado_conformidade = verificar_conformidade_documento(texto_doc_referencia, doc_referencia_nome, texto_doc_analisar, nome_doc_analisar)
                                st.session_state.conformidade_resultados[f"{nome_doc_analisar}_vs_{doc_referencia_nome}"] = resultado_conformidade
                            else: st.error(f"Não foi possível ler: {nome_doc_analisar}")
                        else: st.error(f"Objeto do arquivo '{nome_doc_analisar}' não encontrado.")
            if st.session_state.conformidade_resultados:
                st.markdown("---")
                for chave_analise, relatorio in st.session_state.conformidade_resultados.items():
                    with st.expander(f"Relatório: {chave_analise.replace('_vs_', ' vs ')}", expanded=True): st.markdown(relatorio)
        elif "colecao_ativa" in st.session_state and st.session_state.colecao_ativa:
            st.warning("A Verificação de Conformidade funciona melhor com arquivos recém-carregados.")
        else:
            st.info("Faça o upload de documentos para ativar a verificação de conformidade.")
            
    # --- NOVA ABA DE DETECÇÃO DE ANOMALIAS ---
    with tab_anomalias:
        st.header("📊 Detecção de Anomalias Contratuais")
        st.markdown("Identifica dados que fogem do padrão no conjunto de contratos carregados. "
                    "**Nota:** Esta funcionalidade depende da qualidade e consistência da extração de dados do 'Dashboard Analítico'.")

        # Verifica se o df_dashboard existe e foi populado
        df_para_anomalias = st.session_state.get("df_dashboard")

        if df_para_anomalias is None or df_para_anomalias.empty:
            st.warning("Os dados para análise de anomalias ainda não foram gerados. "
                       "Por favor, vá para a aba '📈 Dashboard Analítico' e clique em "
                       "'🚀 Gerar Dados para Dashboard e Anomalias' primeiro.")
        else:
            st.info("Analisando os dados extraídos da aba 'Dashboard Analítico' em busca de anomalias.")
            if st.button("🚨 Detectar Anomalias Agora", key="btn_detectar_anomalias"):
                st.session_state.anomalias_resultados = detectar_anomalias_no_dataframe(df_para_anomalias.copy()) # Passa uma cópia
            
            if st.session_state.get("anomalias_resultados"):
                st.subheader("Resultados da Detecção de Anomalias:")
                if isinstance(st.session_state.anomalias_resultados, list) and len(st.session_state.anomalias_resultados) > 0:
                    for anomalia in st.session_state.anomalias_resultados:
                        st.markdown(f"- {anomalia}")
                else:
                    st.info("Nenhuma anomalia significativa detectada com os critérios atuais, ou os dados não foram suficientes para a análise.")
