# Contributing

Please open an issue before changing the pretrained-model contract. Any result
claim or checkpoint change must preserve the frozen cells, features, masks,
preprocessing, seed/epoch declaration, metric directions, and truth-sealing
boundary, or clearly introduce a new versioned contract.

For code changes:

```bash
python -m pip install -e '.[test]'
pytest
python scripts/audit_repository.py .
```

Do not commit weights, input matrices, cell identifiers, labels, truth,
predictions, latents, optimizer state, replay state, cluster paths, or secrets.
