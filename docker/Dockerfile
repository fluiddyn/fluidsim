FROM python:3.11
LABEL Pierre Augier <pierre.augier@univ-grenoble-alpes.fr>
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update

RUN apt-get install -y --no-install-recommends \
  libfftw3-dev libfftw3-mpi-dev \
  openmpi-bin libhdf5-openmpi-dev libopenblas-dev
RUN apt-get install -y --no-install-recommends \
  gfortran clang
# also include utilities for debugging
RUN apt-get install -y --no-install-recommends \
  emacs vim \
  meld less \
  # python3-pip is for Mercurial (hg-git, hg-evolve)
  mercurial python3-pip git
RUN apt-get install -y --no-install-recommends \
  rsync \
  libgl1 xvfb xauth libgl1-mesa-dev \
  libxkbcommon-x11-0 \
  swig sudo graphviz

RUN rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1000 appuser && useradd -m -r -u 1000 -g appuser -s /bin/bash appuser -s /bin/bash && usermod -a -G sudo appuser
RUN echo appuser:appuser | chpasswd
USER appuser
ARG HOME=/home/appuser
RUN mkdir -p $HOME/opt
WORKDIR $HOME/opt
RUN echo $USER $HOME $PWD && whoami

ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN $(hg debuginstall -T '{pythonexe}') -m pip install hg-evolve hg-git --no-cache-dir --user

COPY --chown=appuser:appuser docker/hgrc $HOME/.hgrc
COPY --chown=appuser:appuser docker/pythranrc $HOME/.pythranrc

RUN mkdir -p $HOME/.local/include
RUN mkdir -p $HOME/.local/lib
RUN ln -s /usr/include/fftw* $HOME/.local/include
RUN ln -s /usr/lib/x86_64-linux-gnu/libfftw3* $HOME/.local/lib

ENV LD_LIBRARY_PATH=$HOME/.local/lib
ENV LIBRARY_PATH=$HOME/.local/lib
ENV PATH=$HOME/.local/bin:$PATH
ENV CPATH=$HOME/.local/include:$CPATH

RUN mkdir -p $HOME/.config/matplotlib
RUN echo 'backend      : agg' > $HOME/.config/matplotlib/matplotlibrc

RUN wget https://foss.heptapod.net/fluiddyn/fluidfft/raw/branch/default/doc/install/install_p3dfft.sh -O ./install_p3dfft.sh
RUN chmod +x install_p3dfft.sh
RUN export FCFLAGS="-w -fallow-argument-mismatch -O2" && \
    export FFLAGS="-w -fallow-argument-mismatch -O2" && \
    ./install_p3dfft.sh

RUN wget https://foss.heptapod.net/fluiddyn/fluidfft/raw/branch/default/doc/install/install_pfft.sh -O ./install_pfft.sh
RUN chmod +x install_pfft.sh
RUN ./install_pfft.sh

RUN uv tool install pdm
RUN uv tool install nox
