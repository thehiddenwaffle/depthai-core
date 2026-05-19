#!/bin/bash
set -euo pipefail

: "${AK_URL:?AK_URL must be set}"
: "${AK_TOKEN:?AK_TOKEN must be set}"
: "${ZOO_HELPER_PLATFORM:?ZOO_HELPER_PLATFORM must be set}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/setup-ak.sh"

AK_REPO="luxonis-depthai-helper-binaries"

# Set paths
export PATH_PREFIX="zoo_helper/$ZOO_HELPER_PLATFORM"
export ZOO_HELPER_BINARY_LOCAL_PATH="build/zoo_helper"

# Get git hash
git config --global --add safe.directory "$(pwd)"
export ZOO_HELPER_GIT_HASH
ZOO_HELPER_GIT_HASH="$(git rev-parse HEAD)"

UPLOAD_PATH="$PATH_PREFIX/$ZOO_HELPER_GIT_HASH/zoo_helper"

echo "----------------------------------------"
echo "AK_REPO: $AK_REPO"
echo "PATH_PREFIX: $PATH_PREFIX"
echo "ZOO_HELPER_BINARY_LOCAL_PATH: $ZOO_HELPER_BINARY_LOCAL_PATH"
echo "ZOO_HELPER_GIT_HASH: $ZOO_HELPER_GIT_HASH"
echo "zoo_helper binary size: $(du -sh "$ZOO_HELPER_BINARY_LOCAL_PATH")"
echo "zoo_helper upload path: $AK_REPO:$UPLOAD_PATH"
echo "----------------------------------------"

setup_ak_cli

ak artifact push --path "$UPLOAD_PATH" "$AK_REPO" "$ZOO_HELPER_BINARY_LOCAL_PATH"
