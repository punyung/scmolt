# Model weights

Weights are distributed as GitHub Release assets under `v0.1.0`; they are not
tracked by Git or Git LFS. See `release-index.json` for exact filenames, byte
sizes, conditions, epochs, seed, and SHA256 digests.

Each asset is a model-only PyTorch payload loaded with
`torch.load(..., weights_only=True)`. Optimizer and replay state are absent.
Use `scmolt inspect` before `scmolt strict-load`.
