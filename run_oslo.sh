#!/usr/bin/env bash

set -euo pipefail

IMAGE_TAG="${N360IC_OSLO_IMAGE:-n360ic/oslo:latest}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OSLO_DATA=""
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --oslo-data)
      OSLO_DATA="$2"; PASSTHROUGH+=("$1" "/data"); shift 2 ;;
    --oslo-data=*)
      OSLO_DATA="${1#*=}"; PASSTHROUGH+=("--oslo-data" "/data"); shift ;;
    *)
      PASSTHROUGH+=("$1"); shift ;;
  esac
done

if [[ -z "${OSLO_DATA}" ]]; then
  echo "error: --oslo-data <dir> is required (host path to the oslo_data repo)." >&2
  exit 2
fi
if [[ ! -d "${OSLO_DATA}" ]]; then
  echo "error: --oslo-data path does not exist: ${OSLO_DATA}" >&2
  exit 2
fi
OSLO_DATA="$(cd "${OSLO_DATA}" && pwd)"  # absolute

WORK_DIR="${N360IC_OSLO_WORK:-${REPO_ROOT}/work/oslo}"
mkdir -p "${WORK_DIR}"

if ! podman image exists "${IMAGE_TAG}"; then
  echo ">> building ${IMAGE_TAG} (first run only)..."
  podman build -f "${REPO_ROOT}/docker/oslo/Dockerfile" -t "${IMAGE_TAG}" "${REPO_ROOT}"
fi

echo ">> running OSLO (${IMAGE_TAG})"
# GPU access via CDI: `--device nvidia.com/gpu=all` (requires the NVIDIA Container
# Toolkit CDI spec generated on the host: `nvidia-ctk cdi generate`).
# :ro,Z / :Z relabel the mounts for SELinux hosts (no-op elsewhere).
exec podman run --rm -it \
  --device nvidia.com/gpu=all \
  -v "${OSLO_DATA}:/data:ro,Z" \
  -v "${WORK_DIR}:/work:Z" \
  "${IMAGE_TAG}" \
  --workdir /work \
  "${PASSTHROUGH[@]}"
