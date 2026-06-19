# Build stage: compile Rust app
ARG BUILDPLATFORM=linux/amd64
FROM --platform=$BUILDPLATFORM rust:1.82-bullseye AS build

ARG TARGETARCH
ARG TARGETVARIANT
ARG APP_NAME="torch_image_classification"

# Copy libtorch directory and Rust files (source code, Cargo configs)
COPY libtorch /libtorch
COPY rust/native/ /

# Install necessary packages
RUN apt-get update && \
    apt-get install -y \
        ca-certificates \
        libgomp1 \
        binutils \
        cmake \
        build-essential && \
    if [ "$TARGETARCH" = "arm64" ]; then \
         apt-get install -y \
             binutils-aarch64-linux-gnu \
             gcc-aarch64-linux-gnu \
             g++-aarch64-linux-gnu; \
    fi && \
    rm -rf /var/lib/apt/lists/* && \
    ldconfig

# Compile the Rust code to a native binary
RUN case "$TARGETARCH" in \
    "amd64") \
        export RUST_TARGET="x86_64-unknown-linux-gnu" ;; \
    "arm64") \
        dpkg --add-architecture arm64 && \
        export RUST_TARGET="aarch64-unknown-linux-gnu" ;; \
    "arm") \
        export RUST_TARGET="armv7-unknown-linux-gnueabihf" ;; \
    *) echo "Unsupported architecture: $TARGETARCH" && exit 1 ;; \
    esac && \
    export LIBTORCH=/libtorch && \
    rustup target add $RUST_TARGET && \
    cargo build --release --target $RUST_TARGET && \
    mv target/${RUST_TARGET}/release/${APP_NAME} /${APP_NAME}

# Final stage: create runtime image
FROM ubuntu:20.04

ARG TARGETARCH
ARG TARGETVARIANT
ARG APP_NAME="torch_image_classification"

ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libgomp1 \
    binutils \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

# Copy inputs and models
COPY inputs .
COPY models .

# Copy binary built in build stage
COPY --from=build /${APP_NAME} /${APP_NAME}
COPY --from=build /libtorch /libtorch

ENV LD_LIBRARY_PATH=/libtorch/lib
