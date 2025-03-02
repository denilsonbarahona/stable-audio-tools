FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias y añadir deadsnakes PPA para Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-distutils python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# (Opcional) Configurar python3 para que apunte a python3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Copiar el pyproject.toml (y opcionalmente el lock file, si existe) al contenedor
COPY pyproject.toml poetry.lock* ./

# Actualizar pip e instalar el paquete usando pyproject.toml
RUN pip3 install --upgrade pip && pip3 install --no-cache-dir .

# Copiar el resto del código al contenedor
COPY . .

CMD ["python3.11", "handler.py"]
