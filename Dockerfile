# Dockerfile
# Utiliza uma imagem base oficial do Python
FROM python:3.10-slim

# Instala as dependências de sistema necessárias para compilar bibliotecas como faiss-cpu
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho dentro do contentor
WORKDIR /app

# Copia o ficheiro de dependências primeiro para aproveitar o cache do Docker
COPY requirements.txt ./requirements.txt

# Instala as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia todos os ficheiros do projeto para o diretório de trabalho
COPY . .

# Expõe a porta que o Streamlit irá usar, que o Cloud Run espera
EXPOSE 8080

# O comando para executar a aplicação quando o contentor iniciar
# O $PORT é uma variável de ambiente que o Cloud Run fornece automaticamente
CMD exec streamlit run app.py --server.port=$PORT --server.headless=true
