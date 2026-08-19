# Architecture

scMoLT Cell C wraps the unmodified VAE and batch discriminator supplied by
`scmidas==0.1.13`. It adds an RNA-canonical transport graph over diagonal
Gaussian biological posteriors while keeping the native technical posterior,
batch expert, and decoders.

For this two-modality BMMC release, only `ATAC -> RNA` and `RNA -> ATAC`
transport branches exist. No ADT tensor, branch, or loss is synthesized.

The C3 forward path is:

1. Encode every observed modality with the native scMIDAS encoder.
2. Split each 34-dimensional posterior into biological `c` (32) and technical
   `u` (2).
3. Translate only the biological Gaussian parameters into RNA coordinates.
4. Fuse translated biological experts with a prior-inclusive product of
   experts; fuse native technical experts separately.
5. Decode the resulting 34-dimensional latent with the native decoder.

Training adds cycle consistency, identity, cross-decoding, domain
discrimination, missingness-conditioned batch confusion, paired RNA-anchor
consistency, and leakage-safe feature re-masking. Replay contains detached
Gaussian parameters only and is intentionally excluded from model state.

The exact code in `src/scmolt/trackc/` includes `gaussian.py`. This matters
because an earlier internal portable bundle omitted that module and was only
loadable while the larger source tree remained on `PYTHONPATH`; this repository
does not rely on that hidden dependency.
