FROM golang:1.23.4-bullseye

ARG TARGETARCH

# Set environment variables
ENV GOROOT=/usr/local/go
ENV GOPATH=/go
ENV PATH=$PATH:$GOROOT/bin:$GOPATH/bin
ENV GOOS=linux
ENV CGO_ENABLED=1
ENV CGO_CFLAGS="-I/usr/include/perfmon"

# Install dependency packages based on the architecture
RUN apt-get update && \
    if [ "$TARGETARCH" = "amd64" ]; then \
         apt-get install -y gcc libpfm4 libpfm4-dev; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
         apt-get install -y gcc-aarch64-linux-gnu libpfm4 libpfm4-dev; \
         # Create a symlink to libpfm4.so called libpfm.so since the build prcoess expects it
         # This was done automatically for amd64 so it was not necessary there
         ln -s /usr/lib/aarch64-linux-gnu/libpfm4.so /usr/lib/aarch64-linux-gnu/libpfm.so; \
         ln -s /usr/lib/aarch64-linux-gnu/libpfm4.so /usr/lib/libpfm.so; \
         ldconfig ; \
    else \
         echo "Unsupported architecture: $TARGETARCH" && exit 1; \
    fi && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Clone the cadvisor repository into the proper GOPATH
WORKDIR /go/src/github.com/google
RUN git clone https://github.com/google/cadvisor.git \
     && cd cadvisor \
     && git checkout f6e31a3

# Go to the cadvisor directory and prepare for build
WORKDIR /go/src/github.com/google/cadvisor
RUN go mod tidy

# Set up architecture-specific environment variables and build
RUN if [ "$TARGETARCH" = "amd64" ]; then \
         export GOARCH=amd64 && \
         export CC=gcc && \
         export CGO_LDFLAGS="-L/usr/lib/x86_64-linux-gnu"; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
         export GOARCH=arm64 && \
         export CC=aarch64-linux-gnu-gcc && \
         export CGO_LDFLAGS="-L/usr/lib/aarch64-linux-gnu -L/usr/lib"; \
    else \
         echo "Unsupported architecture: $TARGETARCH" && exit 1; \
    fi && \
    GO_FLAGS="-tags=libpfm,netgo" make build

# Copy the build artifacts to an output directory so it can be extracted more 
# conveniently
RUN mkdir -p /output && cp /go/src/github.com/google/cadvisor/_output/cadvisor /output/