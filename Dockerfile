#FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM nvidia/cuda:12.8.1-base-ubuntu22.04

# https://docs.docker.com/reference/dockerfile/#shell-and-exec-form
# https://manpages.ubuntu.com/manpages/noble/en/man1/sh.1.html
SHELL ["/bin/sh", "-exc"]

ARG DEBIAN_FRONTEND=noninteractive
ARG python_version=3.9.21

COPY --link --from=ghcr.io/astral-sh/uv:0.7.14 /uv /usr/local/bin/uv

RUN apt-get update --quiet && \
    apt-get upgrade -y && \
    apt-get install --quiet --no-install-recommends -y build-essential git ca-certificates \
    libgl1 libglib2.0-0 libusb-1.0-0-dev libsndfile1 ffmpeg 

# Forcing http 1.1 to fix https://stackoverflow.com/q/59282476
RUN git config --global http.version HTTP/1.1 && \
    uv python install $python_version && \
    rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON="python$python_version" \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app \
    UV_LINK_MODE=copy \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONOPTIMIZE=1 \
    PATH="/app/bin:$PATH"

WORKDIR /project
COPY pyproject.toml uv.lock README.md /project

# Building deps
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project --frozen

COPY ./fairseq_lib /project/fairseq_lib
# Building fairseq_lib
# cd to /project/fairseq_lib and run uv pip install -e .
RUN cd fairseq_lib && \
    uv pip install -e ./ -vvv

#ENV PYTHONPATH="$PYTHONPATH:/project/fairseq_lib"
# Copying the rest of the project files
COPY ./.project-root /project/.project-root
COPY ./src /project/src
COPY ./scripts /project/scripts
COPY ./configs /project/configs


# Default command to keep container alive (so we can exec into it or run commands)
CMD ["sleep", "infinity"]