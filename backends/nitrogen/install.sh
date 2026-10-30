#!/bin/bash
set -e

backend_dir=$(dirname "$0")
if [ -d "$backend_dir/common" ]; then
    source "$backend_dir/common/libbackend.sh"
else
    source "$backend_dir/../common/libbackend.sh"
fi

installRequirements

# NitroGen's own dependencies cover its Windows game client too, so we install only the package.
uv pip install --no-deps "nitrogen @ https://github.com/MineDojo/NitroGen/archive/32608444660950ffda95e1e57c79632ad65bea10.tar.gz"
