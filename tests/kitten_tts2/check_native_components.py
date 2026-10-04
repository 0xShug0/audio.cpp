#!/usr/bin/env python3
"""Compare native components with upstream PyTorch (validation only, never runtime).

Speaker inputs are captured from a native clone request. Decoder comparison
replaces both upstream Gaussian draws with zeros and returns the mel boundary,
avoiding differences between PyTorch CPU RNG and the native S3 RNG.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch


def compare(name, actual, expected, maximum):
    a, b = np.asarray(actual).reshape(-1), np.asarray(expected).reshape(-1)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    error = np.abs(a-b)
    result = dict(max_abs=float(error.max()), mean_abs=float(error.mean()),
                  cosine=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))))
    print(name, json.dumps(result), flush=True)
    assert np.isfinite(a).all() and result['max_abs'] < maximum, (name, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--clone-trace', type=Path)
    p.add_argument('--decoder-output', type=Path)
    p.add_argument('--codes', type=Path)
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    results = {}
    if args.clone_trace:
        spec = importlib.util.spec_from_file_location('kitten_speaker_reference',
            args.upstream/'kittenml/kittentts2/speaker_embedding.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.load_speaker_embedding_model(args.model/'speaker/model.safetensors')
        waveform = np.fromfile(args.clone_trace/'reference_16k.f32', dtype=np.float32)
        raw = model(torch.from_numpy(waveform).reshape(1,1,-1)).squeeze(0).numpy()
        reference = raw / np.linalg.norm(raw)
        actual = np.fromfile(args.clone_trace/'speaker_embedding.f32', dtype=np.float32)
        results['speaker'] = compare('speaker', actual, reference, 0.00005)
        # Project the same embedding to isolate projection arithmetic from encoder error.
        from safetensors import safe_open
        with safe_open(args.model/'lm/model.safetensors', framework='pt') as f:
            state = {k: f.get_tensor(k) for k in f.keys() if k.startswith('spk_proj.')}
        linear = torch.nn.functional.linear(torch.from_numpy(actual).to(torch.bfloat16),
            state['spk_proj.0.weight'], state['spk_proj.0.bias'])
        projected = torch.nn.functional.layer_norm(linear, (2048,),
            state['spk_proj.1.weight'], state['spk_proj.1.bias'], 1e-5).float().numpy()
        results['projection'] = compare('projection',
            np.fromfile(args.clone_trace/'speaker_projection.f32', dtype=np.float32), projected, 0.032)
    if args.decoder_output:
        model = torch.jit.load(str(args.model/'cpp/default/decoder.pt'), map_location='cpu')
        graph = model.graph.copy()
        mel = None
        for node in list(graph.nodes()):
            if node.kind() in ('aten::randn', 'aten::randn_like'):
                replacement = graph.create(node.kind().replace('randn', 'zeros'), list(node.inputs()), 1)
                replacement.output().setType(node.output().type())
                replacement.insertBefore(node)
                node.output().replaceAllUsesWith(replacement.output())
                node.destroy()
            elif node.kind() == 'aten::_convolution':
                weight = list(node.inputs())[1].toIValue()
                if weight is not None and list(weight.shape) == [512,80,3]:
                    assert mel is None, 'ambiguous mel boundary in decoder export'
                    mel = list(node.inputs())[0]
        assert mel is not None, 'could not locate upstream mel boundary'
        graph.eraseOutput(0)
        graph.registerOutput(mel)
        encoder = next(value for node in graph.nodes() for value in node.outputs() if value.debugName() == 'h.1')
        graph.registerOutput(encoder)
        outputs = list(graph.outputs())
        combined = graph.create('prim::TupleConstruct', outputs)
        combined.output().setType(torch._C.TupleType([x.type() for x in outputs]))
        graph.appendNode(combined)
        graph.eraseOutput(1)
        graph.eraseOutput(0)
        graph.registerOutput(combined.output())
        torch._C._jit_pass_dce(graph)
        function = torch._C._create_function_from_graph('kitten_zero_noise_mel', graph)
        voice = json.loads((args.model/'cpp/default/voices.json').read_text(encoding='utf-8'))['Bruno']
        codes = json.loads(args.codes.read_text(encoding='utf-8')) + [4299]*3
        with torch.jit.optimized_execution(False):
            expected, encoder = function(model, torch.tensor([codes]), torch.tensor(voice['prompt_token']),
                torch.tensor(voice['prompt_feat']), torch.tensor(voice['embedding']))
        results['encoder'] = compare('encoder', np.fromfile(args.decoder_output/'encoder.f32', dtype=np.float32), encoder.numpy(), 0.0001)
        expected = expected.numpy()
        native_mel = np.fromfile(args.decoder_output/'zero_noise_mel.f32', dtype=np.float32).reshape(expected.shape)
        results['mel'] = compare('mel', native_mel, expected, 0.005)
        # Isolate HiFT with identical native mel and zero phase/noise draws.
        graph = model.graph.copy()
        mel_input = graph.addInput('native_mel')
        mel_input.setType(torch._C.TensorType.get())
        f0_value = None
        for node in list(graph.nodes()):
            if node.kind() in ('aten::randn', 'aten::randn_like', 'aten::rand'):
                kind = 'aten::zeros_like' if node.kind().endswith('_like') else 'aten::zeros'
                replacement = graph.create(kind, list(node.inputs()), 1)
                replacement.output().setType(node.output().type())
                replacement.insertBefore(node)
                node.output().replaceAllUsesWith(replacement.output())
                node.destroy()
            else:
                for value in node.outputs():
                    if value.debugName() == 'feat0.1': value.replaceAllUsesWith(mel_input)
                    if value.debugName() == 'f0.1': f0_value = value
        assert f0_value is not None
        combined = graph.create('prim::TupleConstruct', [list(graph.outputs())[0], f0_value])
        combined.output().setType(torch._C.TupleType([torch._C.TensorType.get()]*2))
        graph.appendNode(combined)
        graph.eraseOutput(0)
        graph.registerOutput(combined.output())
        torch._C._jit_pass_dce(graph)
        function = torch._C._create_function_from_graph('kitten_zero_noise_vocoder', graph)
        with torch.jit.optimized_execution(False):
            wave, pitch = function(model, torch.tensor([codes]), torch.tensor(voice['prompt_token']),
                torch.tensor(voice['prompt_feat']), torch.tensor(voice['embedding']), torch.from_numpy(native_mel))
        results['f0'] = compare('f0', np.fromfile(args.decoder_output/'f0.f32', dtype=np.float32), pitch.numpy(), 0.01)
        # The wrapper applies a 960-sample fade; the generic vocoder returns raw audio.
        results['vocoder'] = compare('vocoder', np.fromfile(args.decoder_output/'vocoder.f32', dtype=np.float32)[960:], wave.numpy().reshape(-1)[960:], 0.01)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
