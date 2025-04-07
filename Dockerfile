FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install poetry && poetry config virtualenvs.create false && poetry install

# Copy source code
COPY . .

# Default command (can be overridden)
CMD ["python", "train_debug.py", "train-debug"]

# Build: docker build -t protein-trainer .
# Run: docker run --rm protein-trainer --help

