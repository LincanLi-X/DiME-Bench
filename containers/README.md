# Container image

Build the CPU-only base image from the repository root:

```bash
docker build -f containers/Dockerfile -t dime-bench:1.0.0 .
docker run --rm -v "$PWD/container-results:/results" dime-bench:1.0.0
```

The default command runs the Mock smoke evaluation. Real-model images should
add the required `hf`, `diffusion`, or `metrics` extras and use a CUDA-enabled
PyTorch base image pinned by the experiment owner.
