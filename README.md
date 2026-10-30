# localai-backends

Custom [LocalAI](https://localai.io) backends, published as images on ghcr and listed in a backend gallery.

## Use

Add the gallery to LocalAI and install a backend from it:

```sh
LOCALAI_BACKEND_GALLERIES='[{"name":"h4ks","url":"github:h4ks-com/localai-backends/index.yaml@main"}]'
local-ai backends install h4ks@ltx2
```

## Add or update a backend

1. Put the backend in `backends/<name>/`, laid out like LocalAI's own Python backends (`backend.py`, `install.sh`, `run.sh`, requirements files).
2. Add its entry to `index.yaml`, or bump its `version` and the matching end of its image tag.
3. Push to `main`. CI builds every image in `index.yaml` that ghcr does not have yet.
4. `local-ai backends upgrade <name>` installs the new version wherever the gallery is configured.

## Develop

```sh
uv sync
prek install
```
