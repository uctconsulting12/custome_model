# Use NVIDIA's PyTorch image with CUDA 12.1
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Install OpenCV and system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements first (for Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Expose port and run
EXPOSE 8006

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8006"]


# docker run --gpus all --shm-size=8g -d -p 9005:8006  -v "$USERPROFILE/.aws:/root/.aws"  -v "E:/All_models/custome_model_training/dataset:/app/dataset" -v "E:/All_models/custome_model_training/runs:/app/runs"   --name custome_container   custome_image