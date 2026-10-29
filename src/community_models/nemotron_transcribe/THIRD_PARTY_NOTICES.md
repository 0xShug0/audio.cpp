# Nemotron Transcribe port notices

The alignment implementation and timestamp unit tables were adapted from NVIDIA/NeMo-Speech.cpp at revision 66d50d91b75a406becd95fbaf39d2c9c13e95d7c. NVIDIA-authored code retains its Apache-2.0 headers.

### SciPy, NumPy and PyTorch numerics

The speech-LLM word aligner (`alignment.cpp`) reproduces
these routines so that its alignments equal NeMo's Python aligner exactly:

- SciPy's rectangular linear sum assignment
  (`scipy/optimize/rectangular_lsap/rectangular_lsap.cpp`, the shortest augmenting
  path algorithm after D. F. Crouse; author P. M. Larsen):
  [`scipy/scipy`](https://github.com/scipy/scipy); Copyright (c) 2001-2002
  Enthought, Inc. 2003, SciPy Developers; BSD 3-Clause License
- NumPy's float32 logarithm kernel (`simd_log_FLOAT`, including its polynomial
  coefficients) and its pairwise summation:
  [`numpy/numpy`](https://github.com/numpy/numpy); Copyright (c) 2005-2025, NumPy
  Developers; BSD 3-Clause License
- PyTorch's float32 `linspace` arithmetic:
  [`pytorch/pytorch`](https://github.com/pytorch/pytorch); Copyright (c) 2016-
  Facebook, Inc (Adam Paszke) and the other contributors listed in its `LICENSE`;
  BSD 3-Clause License

The BSD 3-Clause license text is reproduced below.

### Unicode Character Database

The generated `timestamp_unit_tables.inc` table contains data derived from
the Unicode 15.1 Character Database. Its source generator is
`scripts/gen_timestamp_unit_tables.py` in the pinned NVIDIA repository.

Copyright © 1991-2026 Unicode, Inc. License: Unicode License v3; the license text
is reproduced below.


## BSD 3-Clause License

The following text applies to the BSD-licensed material identified above, with
the copyright notices listed for each.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


## Unicode License v3

UNICODE LICENSE V3

COPYRIGHT AND PERMISSION NOTICE

Copyright © 1991-2026 Unicode, Inc.

NOTICE TO USER: Carefully read the following legal agreement. BY
DOWNLOADING, INSTALLING, COPYING OR OTHERWISE USING DATA FILES, AND/OR
SOFTWARE, YOU UNEQUIVOCALLY ACCEPT, AND AGREE TO BE BOUND BY, ALL OF THE
TERMS AND CONDITIONS OF THIS AGREEMENT. IF YOU DO NOT AGREE, DO NOT
DOWNLOAD, INSTALL, COPY, DISTRIBUTE OR USE THE DATA FILES OR SOFTWARE.

Permission is hereby granted, free of charge, to any person obtaining a
copy of data files and any associated documentation (the "Data Files") or
software and any associated documentation (the "Software") to deal in the
Data Files or Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, and/or sell
copies of the Data Files or Software, and to permit persons to whom the
Data Files or Software are furnished to do so, provided that either (a)
this copyright and permission notice appear with all copies of the Data
Files or Software, or (b) this copyright and permission notice appear in
associated Documentation.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
THIRD PARTY RIGHTS.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE
BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES,
OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE DATA
FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall
not be used in advertising or otherwise to promote the sale, use or other
dealings in these Data Files or Software without prior written
authorization of the copyright holder.
