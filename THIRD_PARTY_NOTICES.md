# Third-party notices

## MIDAS / scMIDAS

scMoLT builds on the modality encoders, decoders, batch encoder and decoder, batch
discriminator, loss functions, data loading and minibatch sampler of scMIDAS 0.1.13.
scMIDAS is installed from PyPI as a dependency (`scmidas==0.1.13`) and imported at run
time; its source code is not copied into this repository.
`src/scmolt/frozenbridge65/gpu_resident.py` reimplements, for GPU-resident training
data, the ATAC binarization and the minibatch schedule of the scMIDAS
`MultiBatchSampler`, and checks them against the installed scMIDAS.

- Project: MIDAS / scMIDAS, <https://github.com/labomics/midas>
- Version used: 0.1.13 (<https://pypi.org/project/scmidas/0.1.13/>)
- License: MIT License (<https://github.com/labomics/midas/blob/main/LICENSE>)
- Reference: He, Z. et al. Mosaic integration and knowledge transfer of single-cell
  multimodal data with MIDAS. *Nature Biotechnology* 42, 1594–1605 (2024).
  <https://doi.org/10.1038/s41587-023-02040-y>

The MIT license of scMIDAS, as distributed with scmidas 0.1.13:

```text
Copyright (c) 2018 The Python Packaging Authority

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Other dependencies (PyTorch, Lightning, NumPy, SciPy, pandas, AnnData, h5py and
others) are installed separately and keep their own licenses.
