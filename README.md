# n360ic-benchmark

Official code for the paper **"Neural 360° Image Compression: State-of-the-Art Benchmark and Perspectives"** by Giovanni Gaiardo and Thiago L.T. da Silveira, accepted at LASCAS 2027.


## :construction: Note: Active development
> This repository is currently being populated with code, documentation and experimental baselines.


## Codecs

The evaluated codecs are vendored as git submodules under `codecs/`. After cloning:

```bash
git submodule update --init
```

### OSLO / OSLO-IC

OSLO needs a specific PyTorch + PyTorch-Geometric + CUDA stack, so it ships as a
Docker image. Dependency versions are pinned in `pyproject.toml` (the `oslo`
extra) and locked in `uv.lock`; the image just materialises that frozen
resolution. `torch==2.4.0+cu121` means the host needs an **NVIDIA GPU + driver +
`nvidia-container-toolkit`**.

**1. Get the data + checkpoints**

```bash
git clone https://gitlab.inria.fr/on-the-sphere-learning-for-omnidirectional-images-oslo/oslo_data.git
```

**2. Run** — `run_oslo.sh` builds the image on first use, then runs the codec.
Pass host paths under `oslo_data`; they are mounted read-only at `/data` and
rewritten automatically:

```bash
./run_oslo.sh \
  --oslo-data ./oslo_data \
  --image ./oslo_data/images_equirectangular/01.jpg \
  --checkpoint-file ./oslo_data/checkpoints/factorizedPrior/factorizedPrior_lambda_0.0932_q_7.pth.tar \
  --model oslo \
  --rate-points all
```

- `--model`: `oslo` (SphereFactorizedPrior) or `oslo-ic` (SphereMeanScaleHyperprior).
- `--rate-points`: `all`, or comma-separated indices into `RATE_POINTS` (e.g. `0,4,9`).
- Outputs (HEALPix samples, neighbor structures, reconstructions) land in `./work/oslo`.
- Override the image tag with `N360IC_OSLO_IMAGE` and the work dir with `N360IC_OSLO_WORK`.

The wrapper itself lives in `src/n360ic_benchmark/adapters/oslo.py` and can be
run directly if you manage the environment yourself:

```bash
uv sync --extra oslo
uv run python -m n360ic_benchmark.adapters.oslo --help
```

