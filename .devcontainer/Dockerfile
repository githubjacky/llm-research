# build the lemonade binary
FROM golang:1.22 AS build-lemonade
RUN go install github.com/lemonade-command/lemonade@latest


# setup the devcontainer
FROM ubuntu:24.04

COPY --from=build-lemonade /go/bin/lemonade /usr/local/bin

# python3.12-venv is required by mason.nivm to install python related lsp
# npm is required to install language server
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        build-essential python3.12-venv npm unzip neovim wget
