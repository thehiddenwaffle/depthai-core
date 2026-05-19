#!/bin/bash
set -euo pipefail

setup_ak_cli() {
    : "${AK_URL:?AK_URL must be set}"
    : "${AK_TOKEN:?AK_TOKEN must be set}"

    local ak_base_url
    ak_base_url="${AK_URL%/}"

    if ! command -v ak >/dev/null 2>&1; then
        curl -fsSL https://raw.githubusercontent.com/artifact-keeper/artifact-keeper-cli/main/install.sh | sh
    fi

    export AK_CONFIG_DIR="${AK_CONFIG_DIR:-$(mktemp -d)}"
    export AK_INSTANCE="${AK_INSTANCE:-ci}"
    export AK_NO_INPUT="${AK_NO_INPUT:-true}"

    if ! ak instance list --format quiet | grep -Fxq "$AK_INSTANCE"; then
        ak instance add "$AK_INSTANCE" "$ak_base_url"
    fi
}
