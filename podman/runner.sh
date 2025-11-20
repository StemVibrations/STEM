podman build -t stem .
podman run --rm -v $(pwd):/app stem
