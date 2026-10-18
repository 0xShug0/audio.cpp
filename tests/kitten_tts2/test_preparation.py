"""Asset-reader regression tests; standard library only, no model weights."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

MODULE_PATH = Path(__file__).resolve().parents[2] / 'tools/community_models/prepare_kitten_tts2_gguf.py'
SPEC = importlib.util.spec_from_file_location('kitten_prepare', MODULE_PATH)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class VoiceArrayTests(unittest.TestCase):
    def read(self, descriptor, shape, payload, dtype, version=1):
        header = repr({'descr': descriptor, 'fortran_order': False, 'shape': shape}).encode('ascii')
        width = 2 if version == 1 else 4
        data = b'\x93NUMPY' + bytes([version, 0]) + len(header).to_bytes(width, 'little') + header + payload
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'voice.npz'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('value.npy', data)
            return PREPARE.read_voice_array(path, 'value', dtype)

    def test_upstream_numeric_types_and_endianness(self):
        for version in (1, 2, 3):
            self.assertEqual(self.read('<f4', (2,), struct.pack('<2f', .5, -1), 'f4', version), [.5, -1])
            self.assertEqual(self.read('>i4', (2,), struct.pack('>2i', 4299, 6560), 'i4', version), [4299, 6560])

    def test_rejects_object_or_wrong_shape(self):
        for descriptor, shape in (('|O', (1,)), ('<f4', (1, 1)), ('<i4', (1,))):
            with self.assertRaises(ValueError):
                self.read(descriptor, shape, b'\0' * 4, 'f4')

    def test_rejects_truncated_and_nonfinite_values(self):
        with self.assertRaises(ValueError):
            self.read('<f4', (2,), struct.pack('<f', 1), 'f4')
        with self.assertRaises(ValueError):
            self.read('<f4', (1,), struct.pack('<f', float('nan')), 'f4')

    def test_incomplete_prepared_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            voices = root / 'voices.json'
            voices.write_text(json.dumps({'Bruno': {}}))
            args = SimpleNamespace(prepared_voices=voices, voices_output=None)
            with self.assertRaisesRegex(ValueError, 'German'):
                PREPARE.prepare_voices(args, root, root / 'unused.safetensors', root, {'Bruno', 'German'})


if __name__ == '__main__':
    unittest.main()
