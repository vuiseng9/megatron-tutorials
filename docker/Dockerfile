FROM nvcr.io/nvidia/pytorch:25.09-py3
SHELL ["/bin/bash", "-lc"]
USER root

RUN apt-get update && \
    apt-get install -y \
        bash-completion \
        git-lfs \
        wget \
        build-essential \
        gdb && \
    rm -rf /var/lib/apt/lists/*


RUN cat << 'EOF' >> ~/.bashrc
if [ -f /etc/bash_completion ]; then
  . /etc/bash_completion
fi
PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\] : \[\033[01;34m\]\w\[\033[01;31m\]$(__git_ps1)\[\033[00m\]\n$ '
EOF

ARG MLM_REPO=https://github.com/NVIDIA/Megatron-LM
ARG MLM_BRANCH=core_r0.14.0

ARG MLM_TUTS=https://github.com/vuiseng9/megatron-tutorials

WORKDIR /workspace
RUN git clone $MLM_TUTS mtuts

WORKDIR /workspace
RUN git clone $MLM_REPO megatron-lm && cd megatron-lm && git checkout $MLM_BRANCH
RUN cd megatron-lm/examples/gpt3 && cp -r /workspace/mtuts/* .

WORKDIR /workspace/megatron-lm/examples/gpt3
RUN make install-dependencies && make prepare-ds-openwebtext-10k

WORKDIR /workspace/megatron-lm/examples/gpt3
CMD ["bash"]