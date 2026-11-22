# Dettato port notices

Dettato code and model weights are developed by KobiMusic and licensed under
Apache-2.0: https://github.com/kobimusic/dettato

Dettato
Copyright 2026 KobiMusic

This product includes software and model weights developed by KobiMusic
(https://github.com/kobimusic/dettato, https://huggingface.co/kobimusic).

The grouping decoder follows the official Dettato host algorithms. Its
nearest-neighbour-chain linkage, stable linkage labeling and distance-cut
traversal follow SciPy's `scipy/cluster/_hierarchy.pyx` (v1.17.0).
Rectangular assignment follows SciPy's rectangular_lsap, using the existing
Nemotron port as its source without changing Nemotron.

## SciPy BSD 3-Clause license

Copyright (c) 2001-2002 Enthought, Inc.
Copyright (c) 2003, SciPy Developers.

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
