FROM nvcr.io/nvidia/tensorflow:24.12-tf2-py3

RUN apt-get update -qq && apt-get install -y -qq libopenslide0 && rm -rf /var/lib/apt/lists/*

# Use existing numpy to avoid conflicts
RUN pip install --no-deps openslide-python pillow tqdm huggingface_hub

WORKDIR /app
