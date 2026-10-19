# Weight Adapters

`engine/framework/assets/weight_adapter.h` separates adapter serialization,
model target bindings, and load-time weight materialization.

1. `read_weight_adapter()` imports factors and effective scales into a
   `WeightAdapter`. Native A/B, PEFT, ComfyUI up/down, and matrix LoKr share
   the same representation after import. PEFT alpha/rank and rsLoRA scaling
   are resolved here, not in model code.
2. Models declare `WeightAdapterBindings` using their tensor names. Bindings
   can split a fused external projection into complete model tensors by output
   row slices. Underscore aliases are registered centrally and collisions fail.
3. `bind_weight_adapter()` validates all targets and shapes before a weight
   store is changed. The model selects its merge arithmetic explicitly.
4. `make_weight_adapter_source()` overlays the plan on immutable base weights.
   Normal weight loading materializes each adapted tensor on the host and uses
   the existing dtype conversion/upload API. Inference graphs do not change.

Scaling follows the file convention: native A/B uses direct user strength,
PEFT reads its config, and ComfyUI uses per-tensor alpha when present. A PEFT
config beside a native or ComfyUI file is not applied to that file. Resolve
the metadata coefficient before multiplying user strength, so a direct native
scale does not acquire extra rounding through a multiply/divide by rank.

| Merge policy | Arithmetic | Existing consumer |
| --- | --- | --- |
| `RankWiseF32` | Add each scaled rank contribution to the F32 base in rank order | VibeVoice |
| `RoundedBF16Delta` | Accumulate B*A in F32, round scaled delta to BF16, add base and round to BF16 | YuE2 |
| `DeltaF32` | Accumulate B*A in F32, scale once, then add to base | ACE-Step |

LoKr reconstructs its optional low-rank factors and applies their Kronecker
product without allocating a full dense delta. Supported factors are matrices;
Tucker/conv, DoRA, QALoRA, and transposed PEFT updates are rejected explicitly.
Full replacements are unscaled and take precedence over updates. Whether a zero
user strength disables replacements is a model policy; YuE2 retains that behavior.

Upload caching is optional. It retains one dtype per adapted tensor, replacing
it rather than accumulating variants. Base storage is never overwritten, reads
never apply an update twice, and the cache belongs to the adapted source.

Add a naming convention to the importer, an adapter algorithm to the normalized
update type/materializer, or an architecture mapping to the model bindings.
Do not add ComfyUI/PEFT conditionals to inference code. New formats must have
import/shape/scaling tests; new algorithms also need numerical reference tests.
