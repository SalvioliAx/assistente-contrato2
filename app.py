import streamlit as st
import os
import pandas as pd
from typing import Optional, List
import re
from datetime import datetime, date
import json
from pathlib import Path
import numpy as np
import time
import fitz  # PyMuPDF
from PIL import Image
import io
import base64

# Importações do LangChain e Pydantic
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document # Adicionado para criar Documentos a partir do texto do Gemini


# --- DEFINIÇÕES GLOBAIS ---
COLECOES_DIR = Path("colecoes_ia")
COLECOES_DIR.mkdir(exist_ok=True)

# --- SCHEMAS DE DADOS ---
class InfoContrato(BaseModel):
    arquivo_fonte: str = Field(description="O nome do arquivo de origem do contrato.")
    nome_banco_emissor: Optional[str] = Field(default="Não encontrado", description="O nome do banco ou instituição financeira principal mencionada.")
    valor_principal_numerico: Optional[float] = Field(default=None, description="Se houver um valor monetário principal claramente definido no contrato (ex: valor total do contrato, valor do empréstimo, limite de crédito principal), extraia apenas o número. Caso contrário, deixe como não encontrado.")
    prazo_total_meses: Optional[int] = Field(default=None, description="Se houver um prazo de vigência total do contrato claramente definido em meses ou anos, converta para meses e extraia apenas o número. Caso contrário, deixe como não encontrado.")
    taxa_juros_anual_numerica: Optional[float] = Field(default=None, description="Se uma taxa de juros principal (anual ou claramente conversível para anual) for mencionada, extraia apenas o número percentual. Caso contrário, deixe como não encontrado.")
    possui_clausula_rescisao_multa: Optional[str] = Field(default="Não claro", description="O contrato menciona explicitamente uma multa em caso de rescisão? Responda 'Sim', 'Não', ou 'Não claro'.")
    condicao_limite_credito: Optional[str] = Field(default="Não encontrado", description="Resumo da política de como o limite de crédito é definido, analisado e alterado.")
    condicao_juros_rotativo: Optional[str] = Field(default="Não encontrado", description="Resumo da regra de como e quando os juros do crédito rotativo são aplicados.")
    condicao_anuidade: Optional[str] = Field(default="Não encontrado", description="Resumo da política de cobrança da anuidade.")
    condicao_cancelamento: Optional[str] = Field(default="Não encontrado", description="Resumo das condições para cancelamento do contrato.")

class EventoContratual(BaseModel):
    descricao_evento: str = Field(description="Uma descrição clara e concisa do evento ou prazo.")
    data_evento_str: Optional[str] = Field(default="Não Especificado", description="A data do evento no formato YYYY-MM-DD. Se não aplicável, use 'Não Especificado'.")
    trecho_relevante: Optional[str] = Field(default=None, description="O trecho exato do contrato que menciona este evento/data.")

class ListaDeEventos(BaseModel):
    eventos: List[EventoContratual] = Field(description="Lista de eventos contratuais com suas datas.")
    arquivo_fonte: str = Field(description="O nome do arquivo de origem do contrato de onde estes eventos foram extraídos.")

# --- CONFIGURAÇÃO DA PÁGINA E DA CHAVE DE API ---
st.set_page_config(layout="wide", page_title="Analisador-IA ProMax", page_icon="💡")
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
    if not lista_arquivos_pdf_upload or not google_api_key or not _embeddings_obj:
        return None, None

    documentos_totais = []
    nomes_arquivos_processados = []
    llm_vision = None

    if google_api_key:
        try:
            llm_vision = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1, request_timeout=300) # Timeout aumentado
        except Exception as e:
            st.warning(f"Não foi possível inicializar o modelo de visão do Gemini: {e}")
            llm_vision = None

    for arquivo_pdf_upload in lista_arquivos_pdf_upload:
        nome_arquivo = arquivo_pdf_upload.name
        st.info(f"Processando arquivo: {nome_arquivo}...")
        documentos_arquivo_atual = []
        texto_extraido_com_sucesso = False
        temp_file_path = Path(f"temp_{nome_arquivo}") # Definido fora do try para o finally

        try:
            with open(temp_file_path, "wb") as f:
                f.write(arquivo_pdf_upload.getbuffer())

            # Tentativa 1: PyPDFLoader
            try:
                st.write(f"Tentando PyPDFLoader para {nome_arquivo}...")
                loader = PyPDFLoader(str(temp_file_path))
                pages_pypdf = loader.load()
                if pages_pypdf:
                    texto_pypdf_concatenado = ""
                    for page_num_pypdf, page_obj_pypdf in enumerate(pages_pypdf):
                        if page_obj_pypdf.page_content and page_obj_pypdf.page_content.strip():
                            texto_pypdf_concatenado += page_obj_pypdf.page_content + "\n"
                            doc = Document(page_content=page_obj_pypdf.page_content,
                                           metadata={"source": nome_arquivo, "page": page_obj_pypdf.metadata.get("page", page_num_pypdf), "method": "pypdf"})
                            documentos_arquivo_atual.append(doc)
                    
                    if len(texto_pypdf_concatenado.strip()) > 100: # Considerar texto útil
                        st.success(f"Texto extraído com PyPDFLoader para {nome_arquivo}.")
                        texto_extraido_com_sucesso = True
            except Exception as e_pypdf:
                st.warning(f"PyPDFLoader falhou para {nome_arquivo}: {e_pypdf}. Tentando PyMuPDF.")

            # Tentativa 2: PyMuPDF (fitz) se PyPDFLoader falhou ou não extraiu texto suficiente
            if not texto_extraido_com_sucesso:
                try:
                    st.write(f"Tentando PyMuPDF (fitz) para {nome_arquivo}...")
                    documentos_arquivo_atual = [] # Limpar docs anteriores se PyPDFLoader falhou
                    doc_fitz = fitz.open(str(temp_file_path))
                    texto_fitz_completo = ""
                    for num_pagina, pagina_fitz in enumerate(doc_fitz):
                        texto_pagina = pagina_fitz.get_text("text")
                        if texto_pagina and texto_pagina.strip():
                            texto_fitz_completo += texto_pagina + "\n"
                            doc = Document(page_content=texto_pagina, metadata={"source": nome_arquivo, "page": num_pagina, "method": "pymupdf"})
                            documentos_arquivo_atual.append(doc)
                    doc_fitz.close()
                    if len(texto_fitz_completo.strip()) > 100:
                        st.success(f"Texto extraído com PyMuPDF para {nome_arquivo}.")
                        texto_extraido_com_sucesso = True
                    elif documentos_arquivo_atual: # Mesmo que pouco texto, se PyMuPDF retornou algo estruturado
                        st.info(f"Texto (limitado) extraído com PyMuPDF para {nome_arquivo}.")
                        texto_extraido_com_sucesso = True
                except Exception as e_fitz:
                    st.warning(f"PyMuPDF (fitz) falhou para {nome_arquivo}: {e_fitz}. Tentando Gemini Vision.")

            # Tentativa 3: Gemini Vision se as outras falharem e llm_vision estiver disponível
            if not texto_extraido_com_sucesso and llm_vision:
                st.write(f"Tentando Gemini Vision para {nome_arquivo}...")
                documentos_arquivo_atual = [] # Limpar docs anteriores
                try:
                    arquivo_pdf_upload.seek(0)
                    pdf_bytes = arquivo_pdf_upload.read()
                    doc_fitz_vision = fitz.open(stream=pdf_bytes, filetype="pdf")
                    
                    prompt_gemini_ocr = "Você é um especialista em OCR. Extraia todo o texto visível desta página de documento da forma mais precisa possível. Mantenha a estrutura do texto original, incluindo parágrafos e quebras de linha, quando apropriado. Se houver tabelas, tente preservar sua estrutura textual."
                    
                    for page_num_gemini in range(len(doc_fitz_vision)):
                        page_fitz_obj = doc_fitz_vision.load_page(page_num_gemini)
                        # Aumentar DPI para melhor qualidade de imagem para OCR
                        pix = page_fitz_obj.get_pixmap(dpi=300) 
                        img_bytes = pix.tobytes("png")
                        base64_image = base64.b64encode(img_bytes).decode('utf-8')

                        human_message = HumanMessage(
                            content=[
                                {"type": "text", "text": prompt_gemini_ocr},
                                {"type": "image_url", "image_url": f"data:image/png;base64,{base64_image}"}
                            ]
                        )
                        
                        with st.spinner(f"Gemini processando página {page_num_gemini + 1}/{len(doc_fitz_vision)} de {nome_arquivo}..."):
                            ai_msg = llm_vision.invoke([human_message])
                        
                        if isinstance(ai_msg, AIMessage) and ai_msg.content and isinstance(ai_msg.content, str):
                            texto_pagina_gemini = ai_msg.content
                            if texto_pagina_gemini.strip():
                                doc = Document(page_content=texto_pagina_gemini, metadata={"source": nome_arquivo, "page": page_num_gemini, "method": "gemini_vision"})
                                documentos_arquivo_atual.append(doc)
                                texto_extraido_com_sucesso = True # Marcar sucesso se Gemini extrair algo
                        time.sleep(2) # Respeitar limites de taxa da API (ajuste conforme necessário)

                    doc_fitz_vision.close()
                    if texto_extraido_com_sucesso:
                        st.success(f"Texto extraído com Gemini Vision para {nome_arquivo}.")
                    else:
                        st.warning(f"Gemini Vision não retornou texto substancial para {nome_arquivo}.")

                except Exception as e_gemini:
                    st.error(f"Erro ao usar Gemini Vision para {nome_arquivo}: {str(e_gemini)[:500]}") # Log mais curto
            
            if texto_extraido_com_sucesso and documentos_arquivo_atual:
                documentos_totais.extend(documentos_arquivo_atual)
                nomes_arquivos_processados.append(nome_arquivo)
            elif not texto_extraido_com_sucesso : # Se nenhuma tentativa funcionou
                st.error(f"Não foi possível extrair texto do arquivo: {nome_arquivo}. O arquivo pode estar vazio, corrompido ou ser uma imagem complexa demais para os métodos atuais.")

        except Exception as e_geral_arquivo:
            st.error(f"Erro geral ao processar o arquivo {nome_arquivo}: {e_geral_arquivo}")
        finally:
            if temp_file_path.exists():
                try:
                    os.remove(temp_file_path)
                except Exception as e_remove:
                    st.warning(f"Não foi possível remover o arquivo temporário {temp_file_path}: {e_remove}")
    # Fim do loop de arquivos

    if not documentos_totais:
        st.error("Nenhum texto pôde ser extraído de nenhum dos documentos fornecidos.")
        return None, [] # Retornar lista vazia para nomes_arquivos

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs_fragmentados = text_splitter.split_documents(documentos_totais)

        if not docs_fragmentados:
             st.error("A fragmentação do texto não resultou em nenhum documento. Verifique o conteúdo extraído.")
             return None, nomes_arquivos_processados # Retornar arquivos que tiveram algum processamento

        st.info(f"Criando vector store com {len(docs_fragmentados)} fragmentos de {len(nomes_arquivos_processados)} arquivos.")
        vector_store = FAISS.from_documents(docs_fragmentados, _embeddings_obj)
        st.success("Vector store criado com sucesso!")
        return vector_store, nomes_arquivos_processados
    except Exception as e_faiss:
        st.error(f"Erro ao criar o vector store com FAISS: {e_faiss}")
        st.error(f"Número de documentos totais (pré-split): {len(documentos_totais)}")
        st.error(f"Número de documentos fragmentados: {len(docs_fragmentados) if 'docs_fragmentados' in locals() else 'N/A'}")
        if 'docs_fragmentados' in locals() and docs_fragmentados:
            st.json(docs_fragmentados[0].metadata) # Mostrar metadados do primeiro fragmento para debug
        return None, nomes_arquivos_processados


@st.cache_data(show_spinner="Extraindo dados detalhados dos contratos...")
def extrair_dados_dos_contratos(_vector_store: Optional[FAISS], _nomes_arquivos: list) -> list:
    if not _vector_store or not google_api_key or not _nomes_arquivos: return []
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    
    resultados_finais = []
    mapa_campos_para_extracao = {
        "nome_banco_emissor": ("Qual o nome principal do banco, instituição financeira ou empresa emissora deste contrato?", "Responda apenas com o nome. Se não encontrar, diga 'Não encontrado'."),
        "valor_principal_numerico": ("Qual o valor monetário principal ou limite de crédito central deste contrato?", "Se encontrar um valor, forneça apenas o número (ex: 10000.50). Se não encontrar, responda 'Não encontrado'."),
        "prazo_total_meses": ("Qual o prazo de vigência total deste contrato em meses? Se estiver em anos, converta para meses.", "Se encontrar, forneça apenas o número de meses. Se não encontrar, responda 'Não encontrado'."),
        "taxa_juros_anual_numerica": ("Qual a principal taxa de juros anual (ou facilmente conversível para anual) mencionada?", "Se encontrar, forneça apenas o número percentual (ex: 12.5). Se não encontrar, responda 'Não encontrado'."),
        "possui_clausula_rescisao_multa": ("Este contrato menciona explicitamente uma multa monetária ou percentual em caso de rescisão?", "Responda apenas com 'Sim', 'Não', ou 'Não claro'."),
        "condicao_limite_credito": ("Qual é a política ou condição para definir o limite de crédito?", "Resuma a política em uma ou duas frases. Se não encontrar, responda 'Não encontrado'."),
        "condicao_juros_rotativo": ("Sob quais condições os juros do crédito rotativo são aplicados?", "Resuma a regra em uma ou duas frases. Se não encontrar, responda 'Não encontrado'."),
        "condicao_anuidade": ("Qual é a política de cobrança da anuidade descrita no contrato?", "Resuma a política em uma ou duas frases. Se não encontrar, responda 'Não encontrado'."),
        "condicao_cancelamento": ("Quais são as regras para o cancelamento ou rescisão do contrato?", "Resuma as regras em uma ou duas frases. Se não encontrar, responda 'Não encontrado'.")
    }
    
    total_campos_a_extrair = len(mapa_campos_para_extracao)
    total_operacoes = len(_nomes_arquivos) * total_campos_a_extrair
    operacao_atual = 0

    barra_progresso_placeholder = st.empty()

    for nome_arquivo in _nomes_arquivos:
        dados_contrato_atual = {"arquivo_fonte": nome_arquivo}
        # Garante que o retriever só busque no arquivo correto.
        retriever_arquivo_atual = _vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={'filter': {'source': nome_arquivo}, 'k': 5} # Aumentar k para mais contexto
        )
        
        for campo, (pergunta_chave, instrucao_adicional) in mapa_campos_para_extracao.items():
            operacao_atual += 1
            progress_value = operacao_atual / total_operacoes if total_operacoes > 0 else 0
            barra_progresso_placeholder.progress(progress_value,
                                         text=f"Extraindo '{campo}' de {nome_arquivo} ({operacao_atual}/{total_operacoes})")
            
            docs_relevantes = retriever_arquivo_atual.get_relevant_documents(pergunta_chave + " " + instrucao_adicional)
            contexto = "\n\n---\n\n".join([f"Trecho do documento '{doc.metadata.get('source', 'N/A')}' (página {doc.metadata.get('page', 'N/A')} - método {doc.metadata.get('method', 'N/A')}):\n{doc.page_content}" for doc in docs_relevantes])


            prompt_extracao = PromptTemplate.from_template(
                "Com base no contexto fornecido, responda à seguinte pergunta de forma precisa. {instrucao_adicional}\n\n"
                "Contexto:\n{contexto}\n\n"
                "Pergunta: {pergunta}\n"
                "Resposta:"
            )
            chain_extracao = LLMChain(llm=llm, prompt=prompt_extracao)
            
            if contexto.strip():
                try:
                    resultado = chain_extracao.invoke({
                        "instrucao_adicional": instrucao_adicional,
                        "contexto": contexto, 
                        "pergunta": pergunta_chave
                    })
                    resposta = resultado['text'].strip()

                    if campo in ["valor_principal_numerico", "prazo_total_meses", "taxa_juros_anual_numerica"]:
                        # Regex aprimorada para capturar números com ou sem decimais, com ou sem separadores de milhar
                        numeros = re.findall(r"[\d]+(?:[.,]\d+)*", resposta)
                        if numeros:
                            try:
                                # Pega o primeiro número encontrado, remove pontos de milhar, substitui vírgula decimal por ponto
                                valor_str = numeros[0].replace('.', '').replace(',', '.')
                                # Se ainda houver vírgula após a primeira (caso de formato europeu tipo 1,234.56), remover.
                                if valor_str.count('.') > 1: # Ex: 1.234.56 -> 1234.56
                                    parts = valor_str.split('.')
                                    valor_str = "".join(parts[:-1]) + "." + parts[-1]
                                
                                if campo == "prazo_total_meses":
                                    dados_contrato_atual[campo] = int(float(valor_str))
                                else:
                                    dados_contrato_atual[campo] = float(valor_str)
                            except ValueError: dados_contrato_atual[campo] = None
                        else: dados_contrato_atual[campo] = None
                    elif campo == "possui_clausula_rescisao_multa":
                        if "sim" in resposta.lower(): dados_contrato_atual[campo] = "Sim"
                        elif "não" in resposta.lower() or "nao" in resposta.lower() : dados_contrato_atual[campo] = "Não"
                        else: dados_contrato_atual[campo] = "Não claro"
                    else: # Campos de texto/condição
                        dados_contrato_atual[campo] = resposta if "não encontrado" not in resposta.lower() and resposta.strip() else "Não encontrado"
                except Exception as e_invoke:
                    st.warning(f"Erro ao invocar LLM para '{campo}' em {nome_arquivo}: {e_invoke}")
                    dados_contrato_atual[campo] = None if "numerico" in campo or "meses" in campo else "Erro na IA"
            else:
                st.warning(f"Contexto não encontrado para '{campo}' em {nome_arquivo} após busca no vector store.")
                dados_contrato_atual[campo] = None if "numerico" in campo or "meses" in campo else "Contexto não encontrado"
            
            time.sleep(1.5) # Pausa entre chamadas de campo

        try:
            info_validada = InfoContrato(**dados_contrato_atual)
            resultados_finais.append(info_validada.model_dump()) # Usar model_dump para Pydantic v2
        except Exception as e_pydantic:
            st.error(f"Erro de validação Pydantic para {nome_arquivo}: {e_pydantic}. Dados: {dados_contrato_atual}")
            # Adiciona um registro parcial mesmo em caso de erro de validação para não perder o arquivo_fonte
            resultados_finais.append({"arquivo_fonte": nome_arquivo, "nome_banco_emissor": "Erro de Validação"})
            
    barra_progresso_placeholder.empty()
    if resultados_finais:
        st.success("Extração detalhada para dashboard e anomalias concluída!")
    else:
        st.warning("Nenhum dado foi extraído. Verifique os logs e os arquivos.")
    return resultados_finais


def detectar_anomalias_no_dataframe(df: pd.DataFrame) -> List[str]:
    anomalias_encontradas = []
    if df.empty: return ["Nenhum dado para analisar anomalias."]
    
    campos_numericos = ["valor_principal_numerico", "prazo_total_meses", "taxa_juros_anual_numerica"]
    for campo in campos_numericos:
        if campo in df.columns:
            # Tentar converter para numérico, tratando erros e strings vazias/None
            serie_original = df[campo]
            serie_numerica = []
            for val in serie_original:
                if isinstance(val, (int, float)) and not np.isnan(val):
                    serie_numerica.append(float(val))
                elif isinstance(val, str):
                    try:
                        # Limpeza básica de string antes de converter
                        val_limpo = val.replace("R$", "").replace(".", "").replace(",", ".").strip()
                        if val_limpo:
                           serie_numerica.append(float(val_limpo))
                    except (ValueError, TypeError):
                        pass # Ignora valores não conversíveis
            
            serie = pd.Series(serie_numerica, dtype=float).dropna()

            if not serie.empty and len(serie) > 1:
                media = serie.mean(); desvio_pad = serie.std()
                if desvio_pad == 0 : # Evitar divisão por zero se todos os valores forem iguais
                    # st.info(f"Todos os valores para '{campo}' são idênticos ({media}), não há desvio para análise de outliers.")
                    continue 

                limite_superior = media + 2 * desvio_pad; limite_inferior = media - 2 * desvio_pad
                
                # Filtrar outliers no DataFrame original usando a série numérica para comparação
                outliers_indices = []
                for i, original_val in enumerate(serie_original):
                    try:
                        # Tenta converter o valor original da mesma forma para comparação
                        val_num_comp = None
                        if isinstance(original_val, (int, float)) and not np.isnan(original_val):
                            val_num_comp = float(original_val)
                        elif isinstance(original_val, str):
                            original_val_limpo = original_val.replace("R$", "").replace(".", "").replace(",", ".").strip()
                            if original_val_limpo:
                                val_num_comp = float(original_val_limpo)
                        
                        if val_num_comp is not None and (val_num_comp > limite_superior or val_num_comp < limite_inferior):
                            outliers_indices.append(df.index[i]) # Assume que o índice do df corresponde
                    except:
                        pass # Ignora erros de conversão na comparação de outliers

                outliers_df = df.loc[outliers_indices]

                for _, linha in outliers_df.iterrows():
                    anomalias_encontradas.append(f"**Anomalia Numérica em `{linha['arquivo_fonte']}`:** Campo '{campo}' com valor `{linha[campo]}` está distante da média ({media:.2f} ± {2*desvio_pad:.2f}). Limites: [{limite_inferior:.2f} - {limite_superior:.2f}]")
            elif len(serie) == 1: 
                anomalias_encontradas.append(f"**Info:** Campo '{campo}' possui apenas um valor numérico (`{serie.iloc[0]}`), não sendo possível análise de desvio.")
    
    campos_categoricos = ["possui_clausula_rescisao_multa", "nome_banco_emissor"]
    for campo in campos_categoricos:
        if campo in df.columns:
            # Tratar valores None ou NaN como uma categoria 'Não Especificado' para contagem
            serie_cat = df[campo].fillna("Não Especificado")
            contagem_valores = serie_cat.value_counts(normalize=True)
            if len(df) > 5: # Só analisa categorias raras se houver um número razoável de contratos
                categorias_raras = contagem_valores[contagem_valores < 0.1] # Menos de 10% de frequência
                for categoria, freq in categorias_raras.items():
                    if categoria == "Não Especificado" and freq > 0.5: # Se 'Não Especificado' for muito comum, não é anomalia
                        continue
                    documentos_com_categoria_rara = df[serie_cat == categoria]['arquivo_fonte'].tolist()
                    anomalias_encontradas.append(f"**Anomalia Categórica:** O valor/categoria '`{categoria}`' para o campo '{campo}' é incomum (presente em {freq*100:.1f}% dos contratos: {', '.join(documentos_com_categoria_rara[:3])}{'...' if len(documentos_com_categoria_rara) > 3 else ''}).")
    
    if not anomalias_encontradas: return ["Nenhuma anomalia significativa detectada com os critérios atuais."]
    return anomalias_encontradas

@st.cache_data(show_spinner="Gerando resumo executivo...")
def gerar_resumo_executivo(arquivo_pdf_bytes, nome_arquivo_original):
    if not arquivo_pdf_bytes or not google_api_key: return "Erro: Arquivo ou chave de API não fornecidos."
    
    texto_completo = ""
    # Tentar extrair texto usando PyMuPDF primeiro, pois é mais robusto para PDFs diversos
    try:
        with fitz.open(stream=arquivo_pdf_bytes, filetype="pdf") as doc_fitz:
            for page in doc_fitz:
                texto_completo += page.get_text() + "\n"
    except Exception as e_fitz:
        st.warning(f"PyMuPDF falhou na extração para resumo de {nome_arquivo_original}: {e_fitz}. Tentando PyPDFLoader.")
        # Fallback para PyPDFLoader se PyMuPDF falhar (embora menos provável com bytes)
        # Para PyPDFLoader, precisamos salvar em um arquivo temporário
        temp_file_path = Path(f"temp_resumo_{nome_arquivo_original}")
        with open(temp_file_path, "wb") as f: f.write(arquivo_pdf_bytes)
        try:
            loader = PyPDFLoader(str(temp_file_path))
            documento_completo_paginas = loader.load()
            texto_completo = "\n\n".join([page.page_content for page in documento_completo_paginas])
        except Exception as e_pypdf:
            st.error(f"PyPDFLoader também falhou para resumo de {nome_arquivo_original}: {e_pypdf}")
            if temp_file_path.exists(): os.remove(temp_file_path)
            return f"Erro: Não foi possível ler o conteúdo do PDF {nome_arquivo_original} para resumo."
        finally:
            if temp_file_path.exists(): os.remove(temp_file_path)

    if not texto_completo.strip():
         # Se ainda não houver texto, tentar Gemini Vision como último recurso
        st.info(f"Texto não extraído por métodos convencionais para resumo de {nome_arquivo_original}. Tentando Gemini Vision...")
        try:
            llm_vision_resumo = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1)
            with fitz.open(stream=arquivo_pdf_bytes, filetype="pdf") as doc_fitz_vision:
                prompt_ocr = "Extraia todo o texto desta página."
                for page_num in range(min(len(doc_fitz_vision), 5)): # Limitar a 5 páginas para resumo com OCR caro
                    page_obj = doc_fitz_vision.load_page(page_num)
                    pix = page_obj.get_pixmap(dpi=200)
                    img_bytes_ocr = pix.tobytes("png")
                    base64_image_ocr = base64.b64encode(img_bytes_ocr).decode('utf-8')
                    human_message_ocr = HumanMessage(content=[
                        {"type": "text", "text": prompt_ocr},
                        {"type": "image_url", "image_url": f"data:image/png;base64,{base64_image_ocr}"}
                    ])
                    with st.spinner(f"Gemini resumindo página {page_num + 1} de {nome_arquivo_original}..."):
                        ai_msg_ocr = llm_vision_resumo.invoke([human_message_ocr])
                    if isinstance(ai_msg_ocr, AIMessage) and ai_msg_ocr.content and isinstance(ai_msg_ocr.content, str):
                        texto_completo += ai_msg_ocr.content + "\n\n"
                    time.sleep(1) 
            if not texto_completo.strip():
                st.error(f"Gemini Vision não conseguiu extrair texto para resumo de {nome_arquivo_original}.")
                return "Erro: Gemini Vision não extraiu texto para resumo."
        except Exception as e_gemini_resumo:
            st.error(f"Erro com Gemini Vision no resumo de {nome_arquivo_original}: {e_gemini_resumo}")
            return "Erro: Falha na extração de texto com Gemini Vision para resumo."


    llm_resumo = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    template_prompt_resumo = PromptTemplate.from_template(
        "Você é um assistente especializado em analisar e resumir documentos jurídicos, como contratos.\n"
        "Com base no texto do contrato fornecido abaixo, crie um resumo executivo em 5 a 7 tópicos concisos (bullet points).\n"
        "Destaque os seguintes aspectos, se presentes: as partes principais envolvidas, o objeto principal do contrato, "
        "prazo de vigência (se houver), principais obrigações financeiras ou condições de pagamento, e as "
        "principais condições ou motivos para rescisão ou cancelamento do contrato.\n"
        "Seja claro e direto.\n\nTEXTO DO CONTRATO:\n{texto_contrato}\n\nRESUMO EXECUTIVO:")
    chain_resumo = LLMChain(llm=llm_resumo, prompt=template_prompt_resumo)
    try: 
        resultado = chain_resumo.invoke({"texto_contrato": texto_completo[:30000]}) # Limitar tamanho do contexto para resumo
        return resultado['text']
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
    try: 
        resultado = chain_riscos.invoke({"nome_arquivo": nome_arquivo_doc, "texto_contrato": texto_completo_doc[:30000]}) # Limitar contexto
        return resultado['text']
    except Exception as e: return f"Erro ao analisar riscos para '{nome_arquivo_doc}': {e}"

@st.cache_data(show_spinner="Extraindo datas e prazos dos contratos...")
def extrair_eventos_dos_contratos(textos_completos_docs: List[dict]) -> List[dict]:
    if not textos_completos_docs or not google_api_key: return []
    llm_eventos = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, request_timeout=120)
    parser = PydanticOutputParser(pydantic_object=ListaDeEventos)
    prompt_eventos_template_str = """Analise o texto do contrato abaixo. Sua tarefa é identificar TODOS os eventos, datas, prazos e períodos importantes mencionados.
Para cada evento encontrado, extraia as seguintes informações:
1.  'descricao_evento': Uma descrição clara e concisa do evento (ex: 'Data de assinatura do contrato', 'Vencimento da primeira parcela', 'Prazo final para entrega do produto', 'Início da vigência', 'Período de carência para alteração de vencimento').
2.  'data_evento_str': A data específica do evento no formato YYYY-MM-DD. Se uma data EXATA não puder ser determinada ou não se aplicar (ex: '10 dias antes do vencimento', 'prazo indeterminado', 'na fatura mensal'), preencha este campo OBRIGATORIAMENTE com la string 'Não Especificado'. NUNCA use null, None ou deixe o campo vazio.
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
    barra_progresso_eventos = st.progress(0, text="Iniciando extração de datas...")
    for i, doc_info in enumerate(textos_completos_docs):
        nome_arquivo, texto_contrato = doc_info["nome"], doc_info["texto"]
        barra_progresso_eventos.progress((i + 1) / len(textos_completos_docs), text=f"Analisando datas em: {nome_arquivo}")
        try:
            # Limitar o tamanho do texto_contrato para evitar erros de token
            texto_contrato_limitado = texto_contrato[:25000] # Ajuste conforme necessário

            resposta_ia_obj = chain_eventos_llm_only.invoke({"texto_contrato": texto_contrato_limitado, "arquivo_fonte": nome_arquivo})
            resposta_ia_str = resposta_ia_obj.content
            try: 
                resultado_parseado = parser.parse(resposta_ia_str)
            except Exception as e_parse:
                st.write(f"Parser Pydantic inicial falhou para {nome_arquivo} (eventos), tentando com OutputFixingParser. Erro: {e_parse}")
                st.code(resposta_ia_str[:1000], language="text") # Mostrar o que a IA retornou
                resultado_parseado = output_fixing_parser.parse(resposta_ia_str)
            
            if resultado_parseado and isinstance(resultado_parseado, ListaDeEventos):
                for evento in resultado_parseado.eventos:
                    data_obj = None
                    if evento.data_evento_str and evento.data_evento_str.lower() not in ["não especificado", "condicional", "vide fatura", "n/a", ""]:
                        try: data_obj = datetime.strptime(evento.data_evento_str, "%Y-%m-%d").date()
                        except ValueError:
                            try: data_obj = datetime.strptime(evento.data_evento_str, "%d/%m/%Y").date()
                            except ValueError: pass # Tentar outros formatos se necessário
                    todos_os_eventos_formatados.append({
                        "Arquivo Fonte": nome_arquivo, "Evento": evento.descricao_evento,
                        "Data Informada": evento.data_evento_str, "Data Objeto": data_obj,
                        "Trecho Relevante": evento.trecho_relevante})
        except Exception as e_main:
            st.warning(f"Erro crítico ao processar datas para '{nome_arquivo}'. Erro: {e_main}")
            todos_os_eventos_formatados.append({
                "Arquivo Fonte": nome_arquivo, "Evento": f"Falha na extração: {str(e_main)[:100]}", 
                "Data Informada": "Erro", "Data Objeto": None, "Trecho Relevante": None})
        time.sleep(1) # Pausa para API
    barra_progresso_eventos.empty()
    if not todos_os_eventos_formatados: st.info("Nenhum evento ou prazo foi extraído dos documentos.")
    else: st.success("Extração de datas e prazos concluída!")
    return todos_os_eventos_formatados


@st.cache_data(show_spinner="Verificando conformidade do documento...")
def verificar_conformidade_documento(texto_doc_referencia, nome_doc_referencia, texto_doc_analisar, nome_doc_analisar):
    if not texto_doc_referencia or not texto_doc_analisar or not google_api_key: return "Erro: Textos dos documentos ou Chave API ausentes."
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
        # Limitar o tamanho dos textos enviados para a API
        texto_doc_referencia_limitado = texto_doc_referencia[:25000]
        texto_doc_analisar_limitado = texto_doc_analisar[:25000]

        resultado = chain_conformidade.invoke({
            "nome_doc_referencia": nome_doc_referencia, "texto_doc_referencia": texto_doc_referencia_limitado,
            "nome_doc_analisar": nome_doc_analisar, "texto_doc_analisar": texto_doc_analisar_limitado})
        return resultado['text']
    except Exception as e: return f"Erro ao gerar análise de conformidade para '{nome_doc_analisar}' vs '{nome_doc_referencia}': {e}"

def formatar_chat_para_markdown(mensagens_chat):
    texto_formatado = "# Histórico da Conversa com Analisador-IA\n\n"
    for mensagem in mensagens_chat:
        if mensagem["role"] == "user": texto_formatado += f"## Você:\n{mensagem['content']}\n\n"
        elif mensagem["role"] == "assistant":
            texto_formatado += f"## IA:\n{mensagem['content']}\n"
            if "sources" in mensagem and mensagem["sources"]:
                texto_formatado += "### Fontes Utilizadas:\n"
                for i, doc_fonte in enumerate(mensagem["sources"]):
                    texto_fonte_original = doc_fonte.page_content; sentenca_chave = mensagem.get("sentenca_chave")
                    # Escapar caracteres Markdown dentro do texto da fonte
                    texto_fonte_md = texto_fonte_original.replace('\n', '  \n') # Manter quebras de linha como no Markdown
                    texto_fonte_md = texto_fonte_md.replace('_', '\\_').replace('*', '\\*').replace('`', '\\`')

                    if sentenca_chave and sentenca_chave in texto_fonte_original: # Comparar com original
                        # Escapar sentenca_chave antes de usar no replace
                        sentenca_chave_escapada_md = sentenca_chave.replace('_', '\\_').replace('*', '\\*').replace('`', '\\`')
                        texto_formatado_fonte = texto_fonte_md.replace(sentenca_chave_escapada_md, f"**{sentenca_chave_escapada_md}**")
                    else: texto_formatado_fonte = texto_fonte_md
                    
                    source_name = doc_fonte.metadata.get('source', 'N/A')
                    page_num = doc_fonte.metadata.get('page', 'N/A')
                    method = doc_fonte.metadata.get('method', '')
                    method_str = f" (Método: {method})" if method else ""

                    texto_formatado += f"- **Fonte {i+1} (Doc: `{source_name}`, Pág: {page_num}{method_str})**:\n  > {texto_formatado_fonte[:500]}...\n\n" # Limitar tamanho da citação
            texto_formatado += "---\n\n"
    return texto_formatado

# --- INICIALIZAÇÃO DO OBJETO DE EMBEDDINGS ---
if google_api_key:
    try:
        embeddings_global = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e_embed:
        st.sidebar.error(f"Erro ao inicializar embeddings: {e_embed}")
        embeddings_global = None
else:
    embeddings_global = None

# --- LAYOUT PRINCIPAL E SIDEBAR ---
st.title("💡 Analisador-IA ProMax")
st.sidebar.image("https://i.imgur.com/aozL2jD.png", width=100) # Exemplo de logo
st.sidebar.header("Gerenciar Documentos")

modo_documento = st.sidebar.radio("Como carregar os documentos?", ("Fazer novo upload de PDFs", "Carregar coleção existente"), key="modo_doc_radio_v3", index=0)
arquivos_pdf_upload_sidebar = None

if modo_documento == "Fazer novo upload de PDFs":
    arquivos_pdf_upload_sidebar = st.sidebar.file_uploader("Selecione um ou mais contratos em PDF", type="pdf", accept_multiple_files=True, key="uploader_sidebar_v3")
    if arquivos_pdf_upload_sidebar:
        if st.sidebar.button("Processar Documentos Carregados", key="btn_proc_upload_sidebar_v3", use_container_width=True):
            if google_api_key and embeddings_global:
                with st.spinner("Processando e indexando documentos... Isso pode levar alguns minutos, especialmente com Gemini Vision."):
                    vs, nomes_arqs = obter_vector_store_de_uploads(arquivos_pdf_upload_sidebar, embeddings_global)
                
                if vs and nomes_arqs: # Checar se o processamento foi bem sucedido
                    st.session_state.vector_store = vs
                    st.session_state.nomes_arquivos = nomes_arqs
                    st.session_state.arquivos_pdf_originais = arquivos_pdf_upload_sidebar # Salva os objetos de arquivo originais
                    st.session_state.colecao_ativa = None
                    st.session_state.messages = []
                    # Limpar dados de abas anteriores
                    for key_to_pop in ['df_dashboard', 'resumo_gerado', 'arquivo_resumido', 
                                       'analise_riscos_resultados', 'eventos_contratuais_df', 
                                       'conformidade_resultados', 'anomalias_resultados']:
                        st.session_state.pop(key_to_pop, None)
                    st.success(f"{len(nomes_arqs)} Documento(s) processado(s)!")
                    st.rerun()
                else:
                    st.error("Falha ao processar documentos. Verifique os logs acima.")
            else: st.sidebar.error("Chave de API ou Embeddings não configurados corretamente.")

elif modo_documento == "Carregar coleção existente":
    colecoes_disponiveis = listar_colecoes_salvas()
    if colecoes_disponiveis:
        colecao_selecionada = st.sidebar.selectbox("Escolha uma coleção:", colecoes_disponiveis, key="select_colecao_sidebar_v3", index=None, placeholder="Selecione uma coleção")
        if colecao_selecionada and st.sidebar.button("Carregar Coleção Selecionada", key="btn_load_colecao_sidebar_v3", use_container_width=True):
            if google_api_key and embeddings_global:
                vs, nomes_arqs = carregar_colecao(colecao_selecionada, embeddings_global)
                if vs and nomes_arqs:
                    st.session_state.vector_store, st.session_state.nomes_arquivos, st.session_state.colecao_ativa = vs, nomes_arqs, colecao_selecionada
                    st.session_state.arquivos_pdf_originais = None # Não há arquivos originais quando carrega coleção
                    st.session_state.messages = []
                    for key_to_pop in ['df_dashboard', 'resumo_gerado', 'arquivo_resumido', 
                                       'analise_riscos_resultados', 'eventos_contratuais_df', 
                                       'conformidade_resultados', 'anomalias_resultados']:
                        st.session_state.pop(key_to_pop, None)
                    st.rerun()
            else: st.sidebar.error("Chave de API ou Embeddings não configurados.")
    else: st.sidebar.info("Nenhuma coleção salva ainda.")

if "vector_store" in st.session_state and st.session_state.vector_store is not None and st.session_state.get("arquivos_pdf_originais"): # Só permite salvar se for de upload novo
    st.sidebar.markdown("---"); st.sidebar.subheader("Salvar Coleção Atual")
    nome_nova_colecao = st.sidebar.text_input("Nome para a nova coleção:", key="input_nome_colecao_sidebar_v3")
    if st.sidebar.button("Salvar Coleção", key="btn_save_colecao_sidebar_v3", use_container_width=True):
        if nome_nova_colecao and st.session_state.nomes_arquivos: salvar_colecao_atual(nome_nova_colecao, st.session_state.vector_store, st.session_state.nomes_arquivos)
        else: st.sidebar.warning("Dê um nome e certifique-se de que há docs carregados.")

if "colecao_ativa" in st.session_state and st.session_state.colecao_ativa: st.sidebar.markdown(f"**Coleção Ativa:** `{st.session_state.colecao_ativa}`")
elif "nomes_arquivos" in st.session_state and st.session_state.nomes_arquivos: st.sidebar.markdown(f"**Arquivos Carregados:** {len(st.session_state.nomes_arquivos)}")

st.sidebar.header("Configurações de Idioma"); idioma_selecionado = st.sidebar.selectbox("Idioma para o CHAT:", ("Português", "Inglês", "Espanhol"), key="idioma_chat_key_sidebar_v3")

# --- INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
if "messages" not in st.session_state: st.session_state.messages = []
# ... (outras inicializações de session_state já estão sendo feitas ao carregar/processar)

# --- LÓGICA DAS ABAS ---
tab_chat, tab_dashboard, tab_resumo, tab_riscos, tab_prazos, tab_conformidade, tab_anomalias = st.tabs([
    "💬 Chat", "📈 Dashboard", "📜 Resumo", "🚩 Riscos", "🗓️ Prazos", "⚖️ Conformidade", "📊 Anomalias"
])
documentos_prontos = google_api_key and embeddings_global and st.session_state.get("vector_store") is not None and st.session_state.get("nomes_arquivos")

if not (google_api_key and embeddings_global):
     st.error("Chave de API do Google ou o modelo de Embeddings não estão configurados. Verifique a barra lateral.")
elif not documentos_prontos:
    st.info("👈 Por favor, carregue e processe documentos PDF ou uma coleção existente na barra lateral para habilitar as funcionalidades.")
else:
    vector_store_global = st.session_state.get("vector_store")
    nomes_arquivos_global = st.session_state.get("nomes_arquivos", [])
    arquivos_pdf_originais_global = st.session_state.get("arquivos_pdf_originais") # Pode ser None se carregou coleção

    with tab_chat:
        st.header("Converse com seus documentos")
        # Mensagem inicial é definida ao carregar/processar docs
        if not st.session_state.messages : 
            st.session_state.messages.append({"role": "assistant", "content": f"Olá! Documentos da coleção '{st.session_state.get('colecao_ativa', 'atual')}' prontos ({len(nomes_arquivos_global)} arquivo(s)). Qual sua pergunta?"})
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("Ver Fontes Utilizadas"):
                        for doc_fonte in message["sources"]:
                            texto_fonte = doc_fonte.page_content; sentenca_chave = message.get("sentenca_chave")
                            # Destaque simples, pode ser melhorado
                            if sentenca_chave and sentenca_chave in texto_fonte: 
                                partes = texto_fonte.split(sentenca_chave, 1)
                                texto_formatado_html = f"{st.markdown(partes[0], unsafe_allow_html=True)}<span style='background-color: #FFFACD; padding: 2px; border-radius: 3px; color: black;'>{st.markdown(sentenca_chave, unsafe_allow_html=True)}</span>{st.markdown(partes[1] if len(partes) > 1 else '', unsafe_allow_html=True)}"
                                st.markdown(texto_formatado_html, unsafe_allow_html=True)
                            else: 
                                st.markdown(texto_fonte)
                            
                            source_name = doc_fonte.metadata.get('source', 'N/A')
                            page_num = doc_fonte.metadata.get('page', 'N/A')
                            method = doc_fonte.metadata.get('method', '')
                            method_str = f" (Método: {method})" if method else ""
                            st.caption(f"Fonte: {source_name} (Pág: {page_num}{method_str})")
                            st.markdown("---")


        if len(st.session_state.messages) > 1 : # Não mostrar botão de exportar para a mensagem inicial da IA
            chat_exportado_md = formatar_chat_para_markdown(st.session_state.messages)
            agora = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(label="📥 Exportar Conversa",data=chat_exportado_md, file_name=f"conversa_contratos_{agora}.md", mime="text/markdown", key="export_chat_btn_tab_v3")
        
        st.markdown("---")
        if prompt := st.chat_input("Faça sua pergunta sobre os contratos...", key="chat_input_v3", disabled=not documentos_prontos):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Pesquisando e pensando..."):
                    llm_chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
                    # Adaptação do prompt para ser mais flexível com a citação
                    template_prompt_chat_str = (
                        "Use os seguintes trechos de contexto para responder à pergunta no final. "
                        "Seja direto e use apenas as informações do contexto fornecido. "
                        "Se o contexto não contiver a resposta, diga que não encontrou a informação nos documentos.\n"
                        "CONTEXTO:\n{context}\n\n"
                        "PERGUNTA: {question}\n\n"
                        "RESPOSTA (no idioma {language}):\n"
                        "[Sua resposta aqui]\n\n"
                        "|||TRECHO MAIS RELEVANTE DO CONTEXTO (cite a sentença ou pequeno parágrafo exato):\n"
                        "[Citação aqui]"
                    )
                    template_prompt_chat = PromptTemplate.from_template(template_prompt_chat_str)

                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm_chat, 
                        chain_type="stuff", 
                        retriever=vector_store_global.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 7, "score_threshold": 0.7} # Ajustar k e threshold
                        ), 
                        return_source_documents=True, 
                        chain_type_kwargs={"prompt": template_prompt_chat.partial(language=idioma_selecionado)}
                    )
                    try:
                        resultado = qa_chain.invoke({"query": prompt}) # Usar invoke para Langchain Expression Language
                        resposta_bruta = resultado["result"]
                        fontes = resultado["source_documents"]
                        
                        resposta_principal = resposta_bruta
                        sentenca_chave = None
                        if '|||TRECHO MAIS RELEVANTE DO CONTEXTO' in resposta_bruta:
                            partes = resposta_bruta.split('|||TRECHO MAIS RELEVANTE DO CONTEXTO', 1)
                            resposta_principal = partes[0].replace("RESPOSTA (no idioma " + idioma_selecionado + "):", "").strip()
                            if len(partes) > 1:
                                sentenca_chave = partes[1].replace("(cite a sentença ou pequeno parágrafo exato):", "").strip()
                        
                        message_placeholder.markdown(resposta_principal)
                        st.session_state.messages.append({"role": "assistant", "content": resposta_principal, "sources": fontes, "sentenca_chave": sentenca_chave})
                    except Exception as e_chat:
                        st.error(f"Erro durante a execução da cadeia de QA: {e_chat}")
                        st.session_state.messages.append({"role": "assistant", "content": "Desculpe, ocorreu um erro ao processar sua pergunta."})
            # st.rerun() # Rerun pode ser problemático aqui, melhor deixar o fluxo seguir

    with tab_dashboard:
        st.header("Análise Comparativa de Dados Contratuais")
        st.markdown("Clique no botão para extrair e comparar os dados chave dos documentos carregados (conforme definido no schema `InfoContrato`).")
        if not (vector_store_global and nomes_arquivos_global):
            st.warning("Carregue documentos ou uma coleção válida para usar o dashboard.")
        else:
            if st.button("🚀 Gerar Dados para Dashboard e Anomalias", key="btn_dashboard_e_anomalias_tab_v3", use_container_width=True):
                dados_extraidos = extrair_dados_dos_contratos(vector_store_global, nomes_arquivos_global)
                if dados_extraidos: 
                    st.session_state.df_dashboard = pd.DataFrame(dados_extraidos)
                    st.success(f"Dados extraídos para {len(st.session_state.df_dashboard)} contratos.")
                else: 
                    st.session_state.df_dashboard = pd.DataFrame()
                    st.warning("Nenhum dado foi extraído para o dashboard. Verifique os logs ou os arquivos.")
                st.session_state.pop('anomalias_resultados', None) # Limpar anomalias antigas
                st.rerun()

            if 'df_dashboard' in st.session_state and st.session_state.df_dashboard is not None:
                if not st.session_state.df_dashboard.empty:
                    st.info("Tabela de dados extraídos dos contratos. Use a barra de rolagem horizontal se necessário.")
                    st.dataframe(st.session_state.df_dashboard, use_container_width=True)
                # Não precisa de 'elif' para warning se vazio, pois já é tratado acima.
            
    with tab_resumo:
        st.header("📜 Resumo Executivo de um Contrato")
        if arquivos_pdf_originais_global: # Resumo funciona melhor com os arquivos originais em mãos
            lista_nomes_arquivos_resumo = [f.name for f in arquivos_pdf_originais_global]
            if lista_nomes_arquivos_resumo:
                arquivo_selecionado_nome_resumo = st.selectbox("Escolha um contrato para resumir:", options=lista_nomes_arquivos_resumo, key="select_resumo_tab_v3", index=None, placeholder="Selecione um arquivo")
                if arquivo_selecionado_nome_resumo and st.button("✍️ Gerar Resumo Executivo", key="btn_resumo_tab_v3", use_container_width=True):
                    arquivo_obj_selecionado = next((arq for arq in arquivos_pdf_originais_global if arq.name == arquivo_selecionado_nome_resumo), None)
                    if arquivo_obj_selecionado:
                        arquivo_bytes = arquivo_obj_selecionado.getvalue() # Obter bytes do UploadedFile
                        resumo = gerar_resumo_executivo(arquivo_bytes, arquivo_obj_selecionado.name)
                        st.session_state.resumo_gerado = resumo
                        st.session_state.arquivo_resumido = arquivo_selecionado_nome_resumo
                        st.rerun()
                    else: st.error("Arquivo selecionado não encontrado (isso não deveria acontecer).")
                
                if st.session_state.get("arquivo_resumido") == arquivo_selecionado_nome_resumo and st.session_state.get("resumo_gerado"):
                    st.subheader(f"Resumo do Contrato: {st.session_state.arquivo_resumido}"); st.markdown(st.session_state.resumo_gerado)
            else: st.info("Nenhum arquivo carregado disponível para resumo nesta sessão.")
        elif nomes_arquivos_global: # Se carregou de coleção
             st.info("A função de resumo individual de arquivos é otimizada para uploads novos. Para coleções, use o chat para pedir resumos.")
        else: st.warning("Carregue documentos para usar a função de resumo.")

    with tab_riscos:
        st.header("🚩 Análise de Cláusulas de Risco")
        st.markdown("Analisa os documentos carregados na sessão atual em busca de cláusulas potencialmente arriscadas.")
        if arquivos_pdf_originais_global:
            if st.button("🔎 Analisar Riscos em Todos os Documentos Carregados", key="btn_analise_riscos_v3", use_container_width=True):
                st.session_state.analise_riscos_resultados = {} # Limpar resultados anteriores
                textos_completos_docs_riscos = []
                # Re-ler os arquivos para garantir que temos o conteúdo completo
                for arquivo_pdf_obj in arquivos_pdf_originais_global:
                    try:
                        arquivo_pdf_obj.seek(0) # Resetar ponteiro
                        pdf_bytes_risco = arquivo_pdf_obj.read()
                        texto_doc_risco = ""
                        with fitz.open(stream=pdf_bytes_risco, filetype="pdf") as doc_fitz_risco:
                            for page_risco in doc_fitz_risco:
                                texto_doc_risco += page_risco.get_text() + "\n"
                        if texto_doc_risco.strip():
                             textos_completos_docs_riscos.append({"nome": arquivo_pdf_obj.name, "texto": texto_doc_risco})
                        else: # Tentar Gemini se PyMuPDF falhar
                            st.info(f"Texto não extraído por PyMuPDF para análise de risco de {arquivo_pdf_obj.name}. Tentando Gemini Vision...")
                            texto_gemini_risco = ""
                            llm_vision_risco = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1)
                            with fitz.open(stream=pdf_bytes_risco, filetype="pdf") as doc_fitz_gemini_risco:
                                for page_num_g_risco in range(len(doc_fitz_gemini_risco)):
                                    page_g_risco = doc_fitz_gemini_risco.load_page(page_num_g_risco)
                                    pix_g_risco = page_g_risco.get_pixmap(dpi=200)
                                    img_bytes_g_risco = pix_g_risco.tobytes("png")
                                    base64_img_g_risco = base64.b64encode(img_bytes_g_risco).decode('utf-8')
                                    msg_g_risco = HumanMessage(content=[{"type": "text", "text": "Extraia o texto desta página."}, {"type": "image_url", "image_url": f"data:image/png;base64,{base64_img_g_risco}"}])
                                    with st.spinner(f"Gemini (riscos) processando pág {page_num_g_risco+1} de {arquivo_pdf_obj.name}..."):
                                        ai_msg_g_risco = llm_vision_risco.invoke([msg_g_risco])
                                    if isinstance(ai_msg_g_risco, AIMessage) and ai_msg_g_risco.content and isinstance(ai_msg_g_risco.content, str):
                                        texto_gemini_risco += ai_msg_g_risco.content + "\n\n"
                                    time.sleep(1)
                            if texto_gemini_risco.strip():
                                textos_completos_docs_riscos.append({"nome": arquivo_pdf_obj.name, "texto": texto_gemini_risco})
                            else:
                                st.warning(f"Não foi possível extrair texto para análise de risco de {arquivo_pdf_obj.name} mesmo com Gemini.")
                    except Exception as e_leitura_risco:
                        st.error(f"Erro ao ler {arquivo_pdf_obj.name} para análise de risco: {e_leitura_risco}")

                resultados_analise_riscos_temp = {}
                if textos_completos_docs_riscos:
                    barra_riscos = st.progress(0, text="Analisando riscos...")
                    for idx, doc_info_risco in enumerate(textos_completos_docs_riscos):
                        barra_riscos.progress((idx + 1) / len(textos_completos_docs_riscos), text=f"Analisando riscos em: {doc_info_risco['nome']}...")
                        resultado_risco_doc = analisar_documento_para_riscos(doc_info_risco["texto"], doc_info_risco["nome"])
                        resultados_analise_riscos_temp[doc_info_risco["nome"]] = resultado_risco_doc
                        time.sleep(1.5) 
                    barra_riscos.empty()
                    st.session_state.analise_riscos_resultados = resultados_analise_riscos_temp
                    st.success("Análise de riscos concluída.")
                else:
                    st.warning("Nenhum texto pôde ser extraído dos documentos para análise de riscos.")
                st.rerun()

            if st.session_state.get("analise_riscos_resultados"):
                st.markdown("---")
                for nome_arquivo_risco, analise_risco in st.session_state.analise_riscos_resultados.items():
                    with st.expander(f"Riscos Identificados em: {nome_arquivo_risco}", expanded=True): st.markdown(analise_risco)
        elif "colecao_ativa" in st.session_state and st.session_state.colecao_ativa: 
            st.warning("A Análise de Riscos detalhada funciona melhor com arquivos recém-carregados, pois requer o conteúdo completo.")
        else: st.info("Faça o upload de documentos para ativar a análise de riscos.")

    with tab_prazos:
        st.header("🗓️ Monitoramento de Prazos e Vencimentos")
        st.markdown("Extrai e organiza datas e prazos importantes dos documentos carregados na sessão atual.")
        if arquivos_pdf_originais_global:
            if st.button("🔍 Analisar Prazos e Datas Importantes", key="btn_analise_prazos_v3", use_container_width=True):
                textos_completos_para_datas_prazos = []
                # Re-ler os arquivos
                for arquivo_pdf_obj_prazo in arquivos_pdf_originais_global:
                    try:
                        arquivo_pdf_obj_prazo.seek(0)
                        pdf_bytes_prazo = arquivo_pdf_obj_prazo.read()
                        texto_doc_prazo = ""
                        with fitz.open(stream=pdf_bytes_prazo, filetype="pdf") as doc_fitz_prazo:
                            for page_prazo in doc_fitz_prazo:
                                texto_doc_prazo += page_prazo.get_text() + "\n"
                        if texto_doc_prazo.strip():
                            textos_completos_para_datas_prazos.append({"nome": arquivo_pdf_obj_prazo.name, "texto": texto_doc_prazo})
                        else: # Tentar Gemini se PyMuPDF falhar
                            st.info(f"Texto não extraído por PyMuPDF para análise de prazos de {arquivo_pdf_obj_prazo.name}. Tentando Gemini Vision...")
                            texto_gemini_prazo = ""
                            llm_vision_prazo = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1)
                            with fitz.open(stream=pdf_bytes_prazo, filetype="pdf") as doc_fitz_gemini_prazo:
                                for page_num_g_prazo in range(len(doc_fitz_gemini_prazo)):
                                    page_g_prazo = doc_fitz_gemini_prazo.load_page(page_num_g_prazo)
                                    pix_g_prazo = page_g_prazo.get_pixmap(dpi=200)
                                    img_bytes_g_prazo = pix_g_prazo.tobytes("png")
                                    base64_img_g_prazo = base64.b64encode(img_bytes_g_prazo).decode('utf-8')
                                    msg_g_prazo = HumanMessage(content=[{"type": "text", "text": "Extraia o texto desta página."}, {"type": "image_url", "image_url": f"data:image/png;base64,{base64_img_g_prazo}"}])
                                    with st.spinner(f"Gemini (prazos) processando pág {page_num_g_prazo+1} de {arquivo_pdf_obj_prazo.name}..."):
                                        ai_msg_g_prazo = llm_vision_prazo.invoke([msg_g_prazo])
                                    if isinstance(ai_msg_g_prazo, AIMessage) and ai_msg_g_prazo.content and isinstance(ai_msg_g_prazo.content, str):
                                        texto_gemini_prazo += ai_msg_g_prazo.content + "\n\n"
                                    time.sleep(1)
                            if texto_gemini_prazo.strip():
                                textos_completos_para_datas_prazos.append({"nome": arquivo_pdf_obj_prazo.name, "texto": texto_gemini_prazo})
                            else:
                                st.warning(f"Não foi possível extrair texto para análise de prazos de {arquivo_pdf_obj_prazo.name} mesmo com Gemini.")
                    except Exception as e_leitura_prazo:
                        st.error(f"Erro ao ler {arquivo_pdf_obj_prazo.name} para análise de prazos: {e_leitura_prazo}")
                
                if textos_completos_para_datas_prazos:
                    eventos_extraidos = extrair_eventos_dos_contratos(textos_completos_para_datas_prazos)
                    if eventos_extraidos:
                        df_eventos = pd.DataFrame(eventos_extraidos)
                        df_eventos['Data Objeto'] = pd.to_datetime(df_eventos['Data Objeto'], errors='coerce')
                        st.session_state.eventos_contratuais_df = df_eventos.sort_values(by="Data Objeto", ascending=True, na_position='last')
                    else: st.session_state.eventos_contratuais_df = pd.DataFrame()
                else:
                    st.warning("Nenhum texto pôde ser extraído dos documentos para análise de prazos.")
                    st.session_state.eventos_contratuais_df = pd.DataFrame()
                st.rerun()

            if 'eventos_contratuais_df' in st.session_state and st.session_state.eventos_contratuais_df is not None:
                df_display_eventos = st.session_state.eventos_contratuais_df.copy()
                if not df_display_eventos.empty:
                    df_display_eventos['Data Formatada'] = pd.NaT # Inicializar coluna
                    if 'Data Objeto' in df_display_eventos.columns and pd.api.types.is_datetime64_any_dtype(df_display_eventos['Data Objeto']):
                         df_display_eventos['Data Formatada'] = df_display_eventos['Data Objeto'].dt.strftime('%d/%m/%Y').fillna('N/A')
                    else: # Fallback se Data Objeto não for datetime
                        df_display_eventos['Data Formatada'] = df_display_eventos.get('Data Informada', pd.Series(['N/A'] * len(df_display_eventos)))

                    st.subheader("Todos os Eventos e Prazos Identificados")
                    colunas_para_exibir_eventos = ['Arquivo Fonte', 'Evento', 'Data Informada', 'Data Formatada', 'Trecho Relevante']
                    colunas_existentes_eventos = [col for col in colunas_para_exibir_eventos if col in df_display_eventos.columns]
                    st.dataframe(df_display_eventos[colunas_existentes_eventos], height=400, use_container_width=True)
                    
                    if 'Data Objeto' in df_display_eventos.columns and pd.api.types.is_datetime64_any_dtype(df_display_eventos['Data Objeto']) and df_display_eventos['Data Objeto'].notna().any():
                        st.subheader("Próximos Eventos (Próximos 90 dias)")
                        hoje_datetime = pd.Timestamp(datetime.now().date()) # Usar pd.Timestamp para comparação correta
                        
                        proximos_eventos = df_display_eventos[
                            (df_display_eventos['Data Objeto'] >= hoje_datetime) &
                            (df_display_eventos['Data Objeto'] <= (hoje_datetime + pd.Timedelta(days=90)))
                        ].copy() # .copy() para evitar SettingWithCopyWarning
                        
                        if not proximos_eventos.empty: 
                            st.table(proximos_eventos[['Arquivo Fonte', 'Evento', 'Data Formatada']])
                        else: st.info("Nenhum evento encontrado para os próximos 90 dias.")
                    else: st.info("Nenhuma data válida encontrada para filtrar próximos eventos ou a coluna 'Data Objeto' está ausente/malformada.")
                elif ("btn_analise_prazos_v3" in st.session_state and st.session_state.btn_analise_prazos_v3):
                     st.warning("A extração de datas não retornou resultados. Verifique os avisos ou os arquivos.")
        elif "colecao_ativa" in st.session_state and st.session_state.colecao_ativa: 
            st.warning("O Monitoramento de Prazos funciona melhor com arquivos recém-carregados, pois requer o conteúdo completo.")
        else: st.info("Faça o upload de documentos para ativar o monitoramento de prazos.")

    with tab_conformidade:
        st.header("⚖️ Verificador de Conformidade Contratual")
        st.markdown("Compare um documento com um documento de referência para identificar desalinhamentos.")
        if arquivos_pdf_originais_global and len(arquivos_pdf_originais_global) >= 1:
            nomes_arquivos_para_selecao_conf = [f.name for f in arquivos_pdf_originais_global]
            col_ref_conf, col_ana_conf = st.columns(2)
            with col_ref_conf:
                doc_referencia_nome_conf = st.selectbox("1. Documento de Referência:", options=nomes_arquivos_para_selecao_conf, key="select_doc_ref_conf_v3", index=None, placeholder="Selecione o doc. de referência")
            
            opcoes_docs_analisar_conf = [n for n in nomes_arquivos_para_selecao_conf if n != doc_referencia_nome_conf] if doc_referencia_nome_conf else nomes_arquivos_para_selecao_conf
            
            if not opcoes_docs_analisar_conf and len(arquivos_pdf_originais_global) > 1 and doc_referencia_nome_conf :
                 st.warning("Selecione um documento de referência diferente para habilitar a análise, ou carregue mais documentos.")
            elif not arquivos_pdf_originais_global or len(arquivos_pdf_originais_global) < 2:
                 st.warning("Carregue pelo menos dois documentos para fazer uma comparação.")

            if opcoes_docs_analisar_conf :
                with col_ana_conf:
                    docs_a_analisar_nomes_conf = st.multiselect("2. Documento(s) a Analisar:", options=opcoes_docs_analisar_conf, key="multiselect_docs_ana_conf_v3", placeholder="Selecione o(s) doc(s) para análise")
                
                if st.button("🔎 Verificar Conformidade", key="btn_ver_conf_v3", use_container_width=True, disabled=not(doc_referencia_nome_conf and docs_a_analisar_nomes_conf)):
                    st.session_state.conformidade_resultados = {} # Limpar
                    doc_referencia_obj_conf = next((arq for arq in arquivos_pdf_originais_global if arq.name == doc_referencia_nome_conf), None)
                    texto_doc_referencia_conf = ""
                    if doc_referencia_obj_conf:
                        try:
                            doc_referencia_obj_conf.seek(0)
                            pdf_bytes_ref = doc_referencia_obj_conf.read()
                            with fitz.open(stream=pdf_bytes_ref, filetype="pdf") as doc_fitz_ref:
                                for page_ref in doc_fitz_ref: texto_doc_referencia_conf += page_ref.get_text() + "\n"
                        except Exception as e_read_ref:
                            st.error(f"Erro ao ler doc de referência {doc_referencia_obj_conf.name}: {e_read_ref}")
                    
                    if not texto_doc_referencia_conf.strip(): st.error(f"Não foi possível ler o conteúdo do documento de referência: {doc_referencia_nome_conf}")
                    else:
                        barra_conf = st.progress(0, text="Analisando conformidade...")
                        for idx_conf, nome_doc_analisar_conf in enumerate(docs_a_analisar_nomes_conf):
                            barra_conf.progress((idx_conf + 1) / len(docs_a_analisar_nomes_conf), text=f"Analisando '{nome_doc_analisar_conf}' vs '{doc_referencia_nome_conf}'...")
                            doc_analisar_obj_conf = next((arq for arq in arquivos_pdf_originais_global if arq.name == nome_doc_analisar_conf), None)
                            if doc_analisar_obj_conf:
                                texto_doc_analisar_conf = ""
                                try:
                                    doc_analisar_obj_conf.seek(0)
                                    pdf_bytes_ana = doc_analisar_obj_conf.read()
                                    with fitz.open(stream=pdf_bytes_ana, filetype="pdf") as doc_fitz_ana:
                                        for page_ana in doc_fitz_ana: texto_doc_analisar_conf += page_ana.get_text() + "\n"
                                except Exception as e_read_ana:
                                     st.error(f"Erro ao ler doc a analisar {doc_analisar_obj_conf.name}: {e_read_ana}")

                                if texto_doc_analisar_conf.strip():
                                    resultado_conformidade_doc = verificar_conformidade_documento(texto_doc_referencia_conf, doc_referencia_nome_conf, texto_doc_analisar_conf, nome_doc_analisar_conf)
                                    st.session_state.conformidade_resultados[f"{nome_doc_analisar_conf}_vs_{doc_referencia_nome_conf}"] = resultado_conformidade_doc
                                    time.sleep(2) 
                                else: st.error(f"Não foi possível ler o conteúdo do documento a analisar: {nome_doc_analisar_conf}")
                            else: st.error(f"Objeto do arquivo '{nome_doc_analisar_conf}' não encontrado (erro interno).")
                        barra_conf.empty()
                        st.success("Análise de conformidade concluída.")
                        st.rerun()

            if st.session_state.get("conformidade_resultados"):
                st.markdown("---")
                for chave_analise_conf, relatorio_conf in st.session_state.conformidade_resultados.items():
                    with st.expander(f"Relatório: {chave_analise_conf.replace('_vs_', ' vs ')}", expanded=True): st.markdown(relatorio_conf)
        elif "colecao_ativa" in st.session_state and st.session_state.colecao_ativa: 
            st.warning("A Verificação de Conformidade funciona melhor com arquivos recém-carregados, pois requer o conteúdo completo.")
        else: st.info("Faça o upload de documentos para ativar a verificação de conformidade.")
    
    with tab_anomalias:
        st.header("📊 Detecção de Anomalias Contratuais")
        st.markdown("Identifica dados que fogem do padrão no conjunto de contratos carregados. "
                    "**Nota:** Esta funcionalidade depende da qualidade e consistência da extração de dados realizada na aba '📈 Dashboard'.")

        df_para_anomalias_tab = st.session_state.get("df_dashboard")

        if df_para_anomalias_tab is None or df_para_anomalias_tab.empty:
            st.warning("Os dados para análise de anomalias ainda não foram gerados. "
                       "Por favor, vá para a aba '📈 Dashboard' e clique em "
                       "'🚀 Gerar Dados para Dashboard e Anomalias' primeiro.")
        else:
            st.info("Analisando os dados extraídos da aba 'Dashboard' em busca de anomalias.")
            if st.button("🚨 Detectar Anomalias Agora", key="btn_detectar_anomalias_v3", use_container_width=True):
                st.session_state.anomalias_resultados = detectar_anomalias_no_dataframe(df_para_anomalias_tab.copy())
                st.rerun()
            
            if st.session_state.get("anomalias_resultados"):
                st.subheader("Resultados da Detecção de Anomalias:")
                if isinstance(st.session_state.anomalias_resultados, list) and len(st.session_state.anomalias_resultados) > 0:
                    for anomalia_item in st.session_state.anomalias_resultados:
                        st.markdown(f"- {anomalia_item}")
                else:
                    st.info("Nenhuma anomalia significativa detectada com os critérios atuais, ou os dados não foram suficientes para a análise.")

