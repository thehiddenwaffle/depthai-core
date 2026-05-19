#!/bin/bash
set -euo pipefail

: "${AK_URL:?AK_URL must be set}"
: "${AK_TOKEN:?AK_TOKEN must be set}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../ci/setup-ak.sh"

# Set repository based on mode
case "$1" in
    --snapshot)
        AK_REPO="luxonis-python-snapshot-local"
        ;;
    --release)
        AK_REPO="luxonis-python-release-local"
        ;;
    *)
        echo "Error: Unknown option $1"
        echo "Usage: $0 [--snapshot | --release]"
        exit 1
        ;;
esac

setup_ak_cli

ak artifact push "$AK_REPO" 'wheelhouse/audited/*'
