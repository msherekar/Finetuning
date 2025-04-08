# Start from miniconda base image
FROM continuumio/miniconda3

# Set working directory inside container
WORKDIR /app

# Copy your environment file and install Conda env
COPY environment.yml .
RUN conda env create -f environment.yml

# Make conda env your default shell
SHELL ["conda", "run", "-n", "protein-finetune", "/bin/bash", "-c"]

# Copy all source code into container
COPY . .

# Set default command
CMD ["conda", "run", "-n", "protein-finetune", "python", "train_debug.py", "train-debug"]


