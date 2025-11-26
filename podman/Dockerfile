FROM python:3.12

# Set working directory
WORKDIR /app

# Copy main files
COPY stem_model.py .

RUN apt-get update && \
    apt-get install -y \
        git \
        libgl1 \
        libglu1-mesa \
        libxext6 \
        libxrender1 \
        libsm6 \
        libice6 \
        libxcursor1 \
        libxinerama1

RUN pip install git+https://github.com/StemVibrations/STEM

CMD ["python", "stem_model.py"]
