FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

RUN apt-get -y update && apt-get -y upgrade && apt-get install -y git && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install accelerate pydub nnAudio PyYAML transformers==4.53.3 hydra-core tensorboard lightning pandas pyarrow einops 'git+https://github.com/OliBomby/slider.git@gedagedigedagedaoh#egg=slider' torch_tb_profiler wandb ninja peft
RUN MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation

# Modify .bashrc to include the custom prompt
RUN echo 'if [ -f /.dockerenv ]; then export PS1="(docker) $PS1"; fi' >> /root/.bashrc
