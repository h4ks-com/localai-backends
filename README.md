# localai-backends

Custom [LocalAI](https://localai.io) backends, published as images on ghcr, and model configs that use them.

## Use

Add both galleries to LocalAI, then install a backend or a model from them:

```sh
LOCALAI_BACKEND_GALLERIES='[{"name":"h4ks","url":"github:h4ks-com/localai-backends/index.yaml@main"}]'
LOCALAI_GALLERIES='[{"name":"h4ks","url":"github:h4ks-com/localai-backends/models/index.yaml@main"}]'
local-ai backends install ltx2
local-ai models install h4ks@ltx-2.5
```

Model entries pin their weight files by revision and sha256, so LocalAI skips the files already on disk.

## Add or update a backend

1. Put the backend in `backends/<name>/`, laid out like LocalAI's own Python backends (`backend.py`, `install.sh`, `run.sh`, requirements files). A backend that extends one of LocalAI's names it in an `upstream` file and holds only the files it adds or replaces.
2. Add its entry to `index.yaml`, or bump its `version` and the matching end of its image tag.
3. Push to `main`. CI builds every image in `index.yaml` that ghcr does not have yet.
4. `local-ai backends upgrade <name>` installs the new version wherever the gallery is configured.

## Develop

```sh
uv sync
prek install
```
