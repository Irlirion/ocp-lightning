# Build: sudo docker build -t <project_name> .
# Run: sudo docker run -v $(pwd):/workspace/project --gpus all -it --rm <project_name>


FROM nvcr.io/partners/gridai/pytorch-lightning:v1.3.8


ENV CONDA_ENV_NAME=base


# Install Miniconda and create main env
RUN conda init bash


# Switch to bash shell
SHELL ["/bin/bash", "-c"]


# Install requirements
COPY requirements.txt ./
RUN source activate ${CONDA_ENV_NAME} \
    && pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt


# Set ${CONDA_ENV_NAME} to default virutal environment
RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc
