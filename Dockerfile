FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

RUN apt-get -y update && apt-get -y upgrade && apt-get install -y git && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install accelerate==1.12.0 pydub==0.25.1 nnAudio==0.3.4 PyYAML==6.0.3 transformers==4.57.3 hydra-core==1.3.2 tensorboard==2.20.0 lightning==2.6.0 pandas==2.3.3 pyarrow==22.0.0 einops==0.8.1 'git+https://github.com/OliBomby/slider.git@gedagedigedagedaoh#egg=slider' torch_tb_profiler==0.4.3 wandb==0.23.1 ninja peft==0.18.1 soxr==1.0.0
RUN MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation

# Modify .bashrc to include the custom prompt
RUN echo 'if [ -f /.dockerenv ]; then export PS1="(docker) $PS1"; fi' >> /root/.bashrc
