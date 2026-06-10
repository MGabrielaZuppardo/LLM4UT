FROM python:3.9-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y \
    openjdk-11-jdk \
    build-essential \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker-requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r docker-requirements.txt

COPY . .

CMD ["bash"]
