FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar dependencias y añadir deadsnakes PPA para Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-distutils python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Configurar python3 para que apunte a python3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Asegurarse de que pip3 esté vinculado a Python 3.11
RUN python3.11 -m ensurepip --upgrade
RUN python3.11 -m pip install --upgrade pip

# Copiar los archivos necesarios para instalar dependencias
COPY pyproject.toml setup.py ./

# Instalar las dependencias
RUN python3.11 -m pip install --no-cache-dir . && \
    python3.11 -m pip install --no-cache-dir runpod==1.7.7

# Copiar el resto del código al contenedor
COPY . .

# Verificar que runpod esté instalado
RUN python3.11 -c "import runpod; print(runpod.__version__)"

# Definir el comando para ejecutar tu script
CMD ["python3.11", "handler.py"]