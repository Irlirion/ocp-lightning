# Build: sudo docker build -t <project_name> .
# Run: sudo docker run -v $(pwd):/workspace/project --gpus all -it --rm <project_name>


FROM nvidia/cuda:11.2.0-cudnn8-devel

ENV CONDA_ENV_NAME=ocp
ENV PYTHON_VERSION=3.8

# Create a working directory
RUN mkdir /workspaces
WORKDIR /workspaces

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Switch to bash shell
SHELL ["/bin/bash", "-c"]

# Install Miniconda and Python
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda3.sh
RUN /bin/bash miniconda3.sh -b -p /conda \
    && rm miniconda3.sh \
    && echo export PATH=/conda/bin:$PATH >> .bashrc
ENV PATH="/conda/bin:${PATH}"

# Update conda
RUN conda update -n base -c defaults conda

COPY env.yml ./
# Create conda env
RUN conda env create \
    -f env.yml \
    -n ${CONDA_ENV_NAME} \
    python=${PYTHON_VERSION}

# Set ${CONDA_ENV_NAME} to default virutal environment
RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc
