import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
OSLO_REPO = _REPO_ROOT / "codecs" / "oslo-360-degree-image-compression"

HEALPIX_RES = 4
PATCH_RES = 2
RATE_POINTS = [
    (0.0005, 1),
    (0.0009, 1),
    (0.0018, 1),
    (0.0035, 2),
    (0.0067, 3),
    (0.0130, 4),
    (0.0250, 5),
    (0.0483, 6),
    (0.0932, 7),
    (0.1800, 8),
]


class OSLOWrapper:
    def __init__(self, workdir: Path, gpu: bool = False, oslo_repo: Path = OSLO_REPO):
        self.workdir = workdir
        self.gpu = gpu
        self.oslo_repo = Path(oslo_repo)
        (self.workdir / "healpix").mkdir(parents=True, exist_ok=True)
        (self.workdir / "neighbor_structure").mkdir(parents=True, exist_ok=True)
        (self.workdir / "out" / "reconstruction").mkdir(parents=True, exist_ok=True)
        (self.workdir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def rate_points(self) -> list[str]:
        return RATE_POINTS

    def ERP_to_HEALPix(self, image: Path):
        try:
            result = subprocess.run(
                [
                    "python3",
                    "./generate_healpix_samples_and_split_to_train_and_test.py",
                    "-i",
                    image.parent.as_posix(),
                    "-e",
                    f"{image.name.split('.')[-1]}",
                    "-r",
                    f"{HEALPIX_RES}",
                    "-s",
                    "-o",
                    (self.workdir / "healpix").as_posix(),
                    "-t",
                    str(0.0),
                    "-v",
                    str(0.0),
                ],
                capture_output=True,
                check=True,
                text=True,
                cwd=self.oslo_repo,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Failed with {e.returncode}")
            print(e.stderr)

    def build_neighbor_struct(self):
        try:
            result = subprocess.run(
                [
                    "python3",
                    "./generate_and_save_healpix_oslo_struct.py",
                    "-hr",
                    f"{HEALPIX_RES}",
                    "-pr",
                    f"{PATCH_RES}",
                    "-o",
                    (self.workdir / "neighbor_structure").as_posix(),
                ],
                capture_output=True,
                check=True,
                text=True,
                cwd=self.oslo_repo,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Failed with {e.returncode}")
            print(e.stderr)

    def run(self, rate_point, checkpoint_file):
        lmbda, q = rate_point
        try:
            result = subprocess.run(
                [
                    "python3",
                    "./main_sphere_compression.py",
                    "-m",
                    "SphereFactorizedPrior",
                    "-nd",
                    (self.workdir / "neighbor_structure").as_posix(),
                    "-o",
                    (self.workdir / "test").as_posix(),
                    "--lambda",
                    str(lmbda),
                    "-q",
                    str(q),
                    "--checkpoint-file",
                    checkpoint_file,
                    "--foldername-valtest",
                    (self.workdir / "out" / "reconstruction").as_posix(),
                    "-t",
                    (self.workdir / "healpix" / "test.txt").as_posix(),
                    "-hr",
                    str(HEALPIX_RES),
                    "-prvt",
                    str(PATCH_RES),
                ],
                capture_output=True,
                check=True,
                text=True,
                cwd=self.oslo_repo,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Failed with {e.returncode}")
            print(e.stderr)


class OSLOICWrapper(OSLOWrapper):
    def run(self, rate_point, checkpoint_file):
        lmbda, q = rate_point
        try:
            result = subprocess.run(
                [
                    "python3",
                    "./main_sphere_compression.py",
                    "-m",
                    "SphereMeanScaleHyperprior",
                    "-nd",
                    (self.workdir / "neighbor_structure").as_posix(),
                    "-o",
                    (self.workdir / "test").as_posix(),
                    "--lambda",
                    str(lmbda),
                    "-q",
                    str(q),
                    "--checkpoint-file",
                    checkpoint_file,
                    "--foldername-valtest",
                    (self.workdir / "out" / "reconstruction").as_posix(),
                    "-t",
                    (self.workdir / "healpix" / "test.txt").as_posix(),
                    "-hr",
                    str(HEALPIX_RES),
                    "-prvt",
                    str(PATCH_RES),
                    "--context-model",
                    "autoregressive",
                    "--attention",
                    "--nonlinearity",
                    "RB",
                    "--unpool-func",
                    "SDPAConvTranspose",
                ],
                capture_output=True,
                check=True,
                text=True,
                cwd=self.oslo_repo,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Failed with {e.returncode}")
            print(e.stderr)


_MODELS = {
    "oslo": OSLOWrapper,
    "oslo-ic": OSLOICWrapper,
}


def _select_rate_points(spec: str) -> list:
    """Resolve a --rate-points spec into a list of (lambda, q) tuples.

    "all"       -> every point in RATE_POINTS
    "0,2,9"     -> RATE_POINTS[0], [2], [9] (by index)
    """
    if spec == "all":
        return RATE_POINTS
    try:
        idxs = [int(x) for x in spec.split(",") if x.strip() != ""]
    except ValueError:
        raise SystemExit(f"Invalid --rate-points spec: {spec!r}")
    try:
        return [RATE_POINTS[i] for i in idxs]
    except IndexError:
        raise SystemExit(
            f"--rate-points index out of range (have {len(RATE_POINTS)} points): {spec!r}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="n360ic_benchmark.adapters.oslo",
        description="Run the OSLO / OSLO-IC 360 image-compression codec over a set of rate points.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(_MODELS),
        default="oslo",
        help="Which OSLO variant to run (default: oslo).",
    )
    parser.add_argument(
        "--oslo-data",
        type=Path,
        required=True,
        help="Directory holding input ERP images and pretrained checkpoints "
        "(the oslo_data repo). Mounted into the container at runtime.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input ERP image to compress (inside --oslo-data).",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        required=True,
        help="Scratch/output directory for HEALPix samples, neighbor structures "
        "and reconstructions.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        required=True,
        help="Path to the pretrained checkpoint (.pth.tar) for the chosen rate point.",
    )
    parser.add_argument(
        "--rate-points",
        default="all",
        help="'all', or comma-separated indices into RATE_POINTS (e.g. '0,4,9'). "
        "Default: all.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Run the underlying OSLO scripts on GPU (CUDA).",
    )
    parser.add_argument(
        "--oslo-repo",
        type=Path,
        default=OSLO_REPO,
        help=f"Path to the OSLO submodule checkout (default: {OSLO_REPO}).",
    )
    args = parser.parse_args(argv)

    if not args.oslo_repo.exists():
        raise SystemExit(
            f"OSLO submodule not found at {args.oslo_repo}. "
            "Did you run `git submodule update --init`?"
        )

    wrapper_cls = _MODELS[args.model]
    wrapper = wrapper_cls(workdir=args.workdir, gpu=args.gpu, oslo_repo=args.oslo_repo)

    rate_points = _select_rate_points(args.rate_points)

    print(f"[oslo] model={args.model} repo={args.oslo_repo}")
    print(f"[oslo] ERP -> HEALPix: {args.image}")
    print(wrapper.ERP_to_HEALPix(args.image) or "", end="")

    print("[oslo] building neighbor structure")
    print(wrapper.build_neighbor_struct() or "", end="")

    for rp in rate_points:
        print(f"[oslo] run rate_point={rp} checkpoint={args.checkpoint_file}")
        print(wrapper.run(rp, args.checkpoint_file) or "", end="")

    return 0


if __name__ == "__main__":
    sys.exit(main())

    def run(self, rate_point, checkpoint_file):
        lmbda, q = rate_point
        try:
            result = subprocess.run(
                [
                    "python3",
                    "./main_sphere_compression.py",
                    "-m",
                    "SphereMeanScaleHyperprior",
                    "-nd",
                    (self.workdir / "neighbor_structure").as_posix(),
                    "-o",
                    (self.workdir / "test").as_posix(),
                    "--lambda",
                    str(lmbda),
                    "-q",
                    str(q),
                    "--checkpoint-file",
                    checkpoint_file,
                    "--foldername-valtest",
                    (self.workdir / "out" / "reconstruction").as_posix(),
                    "-t",
                    (self.workdir / "healpix" / "test.txt").as_posix(),
                    "-hr",
                    str(HEALPIX_RES),
                    "-prvt",
                    str(PATCH_RES),
                    "--context-model",
                    "autoregressive",
                    "--attention",
                    "--nonlinearity",
                    "RB",
                    "--unpool-func",
                    "SDPAConvTranspose",
                ],
                capture_output=True,
                check=True,
                text=True,
                cwd=self.oslo_repo,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Failed with {e.returncode}")
            print(e.stderr)
