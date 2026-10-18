# Sortformer v2.1 converter notices

## handy-computer/transcribe.cpp mapping

The tensor mapping used by `convert_sortformer_v2_1.py` is adapted from
`handy-computer/transcribe.cpp`, commit
`e2f82cb6702315a1194f3bf1a6fee67cd2678447`, under the MIT License.

Copyright (c) 2026 The transcribe.cpp authors

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

## NVIDIA NeMo-Speech.cpp AOSC reference

The v2.1 host-side AOSC state and channel-birth behavior are adapted from
`NVIDIA/NeMo-Speech.cpp`, commit `ffa38cb2408f1e832a36d46fef5e3e1e80d07e6c`,
under the Apache License 2.0. The upstream copyright and license headers are
retained in the adapted source.

## NVIDIA NeMo-Speech.cpp streaming scheduler reference

The v2.1 stream-window scheduling lifecycle is adapted from
`NVIDIA/NeMo-Speech.cpp`, commit `ffa38cb2408f1e832a36d46fef5e3e1e80d07e6c`,
under the Apache License 2.0. The audio.cpp implementation translates the
window geometry and final-tail behavior into its own runtime types; it does not
copy the NeMo public API or `ParameterParser`.

## NVIDIA model weights

The checkpoint is from
`nvidia/diar_streaming_sortformer_4spk-v2.1`, revision
`fafaab5faa1617a0ca52d38dd3dc4bd636800d3d`, and is governed by the NVIDIA
Open Model License:

https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/

The converted weights are local-use artifacts until redistribution is
approved separately.
