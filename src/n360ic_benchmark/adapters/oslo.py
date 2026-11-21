import subprocess
from pathlib import Path

HEALPIX_RES = 8
PATCH_RES = 6


class OSLOWrapper:
    name = "oslo"

    def rate_points(self) -> list[str]:
        pass

    def _ERP_to_HEALPix(self, image: Path):
        try:
            result = subprocess.run(
                [
                    "python3",
                    "./generate_healpix_samples_and_split_to_train_and_test.py",
                    "-i",
                    f"{image.parent}",
                    "-e",
                    f"{image.suffix}",
                    "-r",
                    f"{HEALPIX_RES}",
                    "-s",
                    "-o",
                    f"{image.parent / 'HEALPix'}",
                    "--no-set-separation",
                ],
                capture_output=True,
                check=True,
                text=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Failed with {e.returncode}")
            print(e.stderr)

    def _build_neighbor_struct(self):
        try:
            result = subprocess.run(
                [
                    "python3",
                    "./generate_and_save_healpix_oslo_struct.py",
                    "-hr",
                    f"{HEALPIX_RES}",
                    "-pr",
                    f"{PATCH_RES}",
                ],
                capture_output=True,
                check=True,
                text=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Failed with {e.returncode}")
            print(e.stderr)

    def run(self, image, rate_point, workdir):
        try:
            result = subprocess.run(
                [
                    "python3",
                    "./main_sphere_compression.py",
                    "--gpu",
                    "-m",
                    "SphereFactorizedPrior",
                    "-nd",
                    f"{workdir / 'neighbor_structure'}",
                ],
                capture_output=True,
                check=True,
                text=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Failed with {e.returncode}")
            print(e.stderr)
