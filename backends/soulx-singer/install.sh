#!/bin/bash
set -e

backend_dir=$(dirname "$0")
if [ -d "$backend_dir/common" ]; then
    source "$backend_dir/common/libbackend.sh"
else
    source "$backend_dir/../common/libbackend.sh"
fi

installRequirements

# SoulX-Singer is not a package, so we keep its source next to the backend at a pinned commit.
rm -rf "$backend_dir/SoulX-Singer"
mkdir -p "$backend_dir/SoulX-Singer"
curl -fsSL https://github.com/Soul-AILab/SoulX-Singer/archive/81aeb3ae772c70093c3de74dc23c92d983801ae4.tar.gz | tar -xz --strip-components=1 -C "$backend_dir/SoulX-Singer"
# The voice converter loads Whisper by its hub name, so we point it at the copy the gallery installs next to the model.
encoder="$backend_dir/SoulX-Singer/soulxsinger/models/modules/whisper_encoder.py"
sed -i -e 's#"openai/whisper-base"#os.environ.get("SOULX_WHISPER", "openai/whisper-base")#' -e 's#^import torch$#import os\nimport torch#' "$encoder"
grep -q 'os.environ.get("SOULX_WHISPER"' "$encoder"

python -c "import nltk; [nltk.download(p, download_dir='$backend_dir/nltk_data', quiet=True) for p in ('averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger', 'cmudict')]"
