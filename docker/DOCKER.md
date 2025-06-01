# Docker Usage Guide

This guide explains how to use Docker with this project, focusing on mounting local directories for data persistence.

## Prerequisites

- [Docker](https://www.docker.com/get-started) installed on your system
- Basic understanding of Docker commands

## Basic Usage

To build and run the Docker container:

```bash
# Build the Docker image
docker build -t mingle .

# Run the container
docker run -it mingle
```

## Mounting Local Directories

Docker allows you to mount local directories into containers using the `-v` or `--volume` flag.

### Example: Mounting Directories

Mount local folders to preserve plots, GIFs, and checkpoints between container runs:

```bash
docker run -it \
    -v $(pwd)/plots:/app/plots \
    -v $(pwd)/gifs:/app/gifs \
    -v $(pwd)/checkpoints:/app/checkpoints \
    mingle
```

This command mounts three directories:
- Your local `./plots` directory to `/app/plots` inside the container
- Your local `./gifs` directory to `/app/gifs` inside the container
- Your local `./checkpoints` directory to `/app/checkpoints` inside the container

### Note

- Make sure the local directories exist before mounting them
- Any data written to these paths in the container will persist on your host machine
- Use absolute paths for more reliability: `-v /absolute/path/to/plots:/app/plots`

## Advanced Usage

For development, you might want to mount the entire codebase:

```bash
docker run -it \
    -v $(pwd):/app \
    mingle
```

This allows code changes on your host to be immediately available in the container.
