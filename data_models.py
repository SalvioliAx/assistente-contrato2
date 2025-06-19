# data_models.py
"""
Este módulo contém as definições dos modelos de dados Pydantic para
estruturar a informação extraída dos contratos.
"""
from pydantic import BaseModel, Field
from typing import Optional, List

class InfoContrato(BaseModel):
    """Schema para as informações principais de um contrato."""
    arquivo_fonte: str = Field(description="O nome do arquivo de origem do contrato.")
    nome_banco_emissor: Optional[str] = Field(default="Não encontrado", description="O nome do banco ou instituição financeira principal mencionada.")
    valor_principal_numerico: Optional[float] = Field(default=None, description="Valor monetário principal do contrato (empréstimo, limite, etc.).")
    prazo_total_meses: Optional[int] = Field(default=None, description="Prazo de vigência total do contrato em meses.")
    taxa_juros_anual_numerica: Optional[float] = Field(default=None, description="Taxa de juros principal anualizada.")
    possui_clausula_rescisao_multa: Optional[str] = Field(default="Não claro", description="Indica se há menção de multa por rescisão ('Sim', 'Não', 'Não claro').")
    condicao_limite_credito: Optional[str] = Field(default="Não encontrado", description="Resumo da política de definição do limite de crédito.")
    condicao_juros_rotativo: Optional[str] = Field(default="Não encontrado", description="Resumo da regra de aplicação dos juros do crédito rotativo.")
    condicao_anuidade: Optional[str] = Field(default="Não encontrado", description="Resumo da política de cobrança da anuidade.")
    condicao_cancelamento: Optional[str] = Field(default="Não encontrado", description="Resumo das condições para cancelamento do contrato.")

class EventoContratual(BaseModel):
    """Schema para um evento ou prazo contratual com data."""
    descricao_evento: str = Field(description="Uma descrição clara e concisa do evento ou prazo.")
    data_evento_str: Optional[str] = Field(default="Não Especificado", description="A data do evento no formato YYYY-MM-DD.")
    trecho_relevante: Optional[str] = Field(default=None, description="O trecho exato do contrato que menciona este evento/data.")

class ListaDeEventos(BaseModel):
    """Schema para uma lista de eventos extraídos de um arquivo."""
    eventos: List[EventoContratual] = Field(description="Lista de eventos contratuais com suas datas.")
    arquivo_fonte: str = Field(description="O nome do arquivo de origem do contrato de onde estes eventos foram extraídos.")
