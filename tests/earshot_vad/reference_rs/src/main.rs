//! Dump upstream Earshot per-frame scores and default segments as JSON.
//!
//! Usage: earshot_reference <16 kHz mono wav> [--bench <repeats>]

use std::time::Instant;

use earshot::{Detector, FRAME_SIZE, SegmenterOptions, segments};

fn read_wav(path: &str) -> Vec<f32> {
	let mut reader = hound::WavReader::open(path).expect("open wav");
	let spec = reader.spec();
	assert_eq!(spec.sample_rate, 16_000, "reference expects 16 kHz audio");
	assert_eq!(spec.channels, 1, "reference expects mono audio");
	match spec.sample_format {
		hound::SampleFormat::Int => {
			let scale = (1u32 << (spec.bits_per_sample - 1)) as f32;
			reader.samples::<i32>().map(|s| s.expect("sample") as f32 / scale).collect()
		}
		hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.expect("sample")).collect()
	}
}

fn main() {
	let args: Vec<String> = std::env::args().collect();
	let path = args.get(1).expect("usage: earshot_reference <wav> [--bench N]");
	let repeats: usize = match args.get(2).map(String::as_str) {
		Some("--bench") => args.get(3).and_then(|v| v.parse().ok()).unwrap_or(10),
		_ => 0
	};
	let audio = read_wav(path);

	let mut detector = Detector::default();
	let scores: Vec<f32> = audio.chunks_exact(FRAME_SIZE).map(|frame| detector.predict(frame).raw).collect();
	let segs: Vec<(usize, usize)> = segments(audio.as_slice(), &SegmenterOptions::default())
		.map(|s| (s.start_sample(), s.end_sample()))
		.collect();

	let mut bench_ms = 0.0f64;
	if repeats > 0 {
		let start = Instant::now();
		for _ in 0..repeats {
			let mut detector = Detector::default();
			let mut sink = 0.0f32;
			for frame in audio.chunks_exact(FRAME_SIZE) {
				sink += detector.predict(frame).raw;
			}
			std::hint::black_box(sink);
		}
		bench_ms = start.elapsed().as_secs_f64() * 1000.0 / repeats as f64;
	}

	let scores_json: Vec<String> = scores.iter().map(|s| format!("{:.9e}", s)).collect();
	let segs_json: Vec<String> = segs.iter().map(|(a, b)| format!("[{a},{b}]")).collect();
	println!(
		"{{\"sample_rate\":16000,\"frame_size\":{},\"num_samples\":{},\"scores\":[{}],\"segments\":[{}],\"bench_ms\":{:.6}}}",
		FRAME_SIZE,
		audio.len(),
		scores_json.join(","),
		segs_json.join(","),
		bench_ms
	);
}
