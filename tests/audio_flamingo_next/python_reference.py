"""Run the official Transformers implementation on a supplied audio file."""

import argparse
import json
from pathlib import Path
import time

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoProcessor, MusicFlamingoForConditionalGeneration
from transformers.audio_utils import load_audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--instruct", default="Transcribe the input speech.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--audio-stage-parity", action="store_true")
    parser.add_argument("--fp32-parity", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = MusicFlamingoForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32 if args.fp32_parity else torch.bfloat16,
        device_map="cpu" if args.fp32_parity else "cuda",
    ).eval()
    conversation = [[{"role": "user", "content": [
        {"type": "text", "text": args.instruct},
        {"type": "audio", "audio": load_audio(
            str(Path(args.audio).resolve()), sampling_rate=processor.feature_extractor.sampling_rate)},
    ]}]]
    records = []
    if args.fp32_parity:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode():
            batch = processor.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=True,
                return_dict=True,
            ).to("cuda")
            model.audio_tower.cuda()
            model.multi_modal_projector.cuda()
            model.pos_emb.cuda()
            audio = model.get_audio_features(
                input_features=batch["input_features"].float(),
                input_features_mask=batch["input_features_mask"],
                input_ids=batch["input_ids"],
            ).pooler_output
            model.audio_tower.cpu()
            model.multi_modal_projector.cpu()
            torch.cuda.empty_cache()
            model.language_model.cuda()
            embeddings = model.language_model.get_input_embeddings()(batch["input_ids"])
            embeddings[batch["input_ids"] == model.config.audio_token_id] = audio
            generated = model.language_model.generate(
                input_ids=batch["input_ids"], inputs_embeds=embeddings,
                attention_mask=batch["attention_mask"], max_new_tokens=args.max_tokens,
                do_sample=False, repetition_penalty=args.repetition_penalty,
                return_dict_in_generate=True, output_scores=True,
            )
            torch.save([score.cpu() for score in generated.scores], args.out_dir / "scores.pt")
            completion = generated.sequences[:, batch["input_ids"].shape[1]:]
            text = processor.batch_decode(completion, skip_special_tokens=True)[0]
            (args.out_dir / "transcript.txt").write_text(text + "\n")
            (args.out_dir / "tokens.json").write_text(json.dumps(completion.tolist()))
            print(text, flush=True)
        return
    if args.audio_stage_parity:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model.audio_tower.float()
        model.multi_modal_projector.float()
        with torch.inference_mode():
            batch = processor.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=True,
                return_dict=True,
            ).to(model.device)
            output = model.get_audio_features(
                input_features=batch["input_features"].float(),
                input_features_mask=batch["input_features_mask"],
                input_ids=batch["input_ids"],
            )
            save_file({
                "features": batch["input_features"].cpu(),
                "mask": batch["input_features_mask"].float().cpu(),
                "input_ids": batch["input_ids"].float().cpu(),
                "encoded": output.last_hidden_state.cpu(),
                "projected": output.pooler_output.cpu(),
            }, str(args.out_dir / "audio_stages.safetensors"))
        print("Saved controlled audio-stage parity tensors", flush=True)
        return
    with torch.inference_mode():
        for iteration in range(args.warmup + args.repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            batch = processor.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=True,
                return_dict=True,
            ).to(model.device)
            batch["input_features"] = batch["input_features"].to(model.dtype)
            generated = model.generate(
                **batch, max_new_tokens=args.max_tokens,
                do_sample=False, repetition_penalty=args.repetition_penalty,
            )
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000
            completion = generated[:, batch["input_ids"].shape[1]:]
            text = processor.batch_decode(completion, skip_special_tokens=True)[0]
            record = {
                "session.wall_ms": elapsed_ms,
                "warmup": iteration < args.warmup,
                "input_ids": batch["input_ids"][0].tolist(),
                "output_ids": completion[0].tolist(),
                "text": text,
            }
            print(json.dumps(record), flush=True)
            records.append(record)
    (args.out_dir / "transcript.txt").write_text(text + "\n")
    (args.out_dir / "results.json").write_text(json.dumps({
        "torch": torch.__version__, "transformers": transformers.__version__,
        "timing_scope": "decoded audio to generated token IDs; includes frontend and generation",
        "settings": vars(args) | {"out_dir": str(args.out_dir)},
        "results": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
