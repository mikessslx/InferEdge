use image;
use std::env;
use std::fs::File;
use std::io::Read;
use std::time::Instant;
use tch::{CModule, Kind, Tensor};

fn main() {
    let start = Instant::now();
    let runtime_args = parse_runtime_args();

    let model = CModule::load(&runtime_args.model_path).unwrap_or_else(|e| {
        panic!("Failed to load model {}: {:?}", runtime_args.model_path, e);
    });
    let load_model_elapsed = start.elapsed();

    let (input, input_loaded_seconds, input_resized_seconds) =
        image_to_tensor(&runtime_args.image_path, 224, 224, start);
    let load_input_elapsed = start.elapsed();

    for trial in 1..=runtime_args.trials {
        let trial_start = Instant::now();
        let output = model
            .forward_ts(&[input.shallow_clone()])
            .unwrap_or_else(|e| {
                panic!("Failed to execute inference: {:?}", e);
            });
        let inference_duration = trial_start.elapsed();
        let inference_elapsed = start.elapsed();
        let trial_workload_duration = trial_start.elapsed();
        let total_elapsed = start.elapsed();

        println!("InferEdge trial {} start", trial);
        if trial == 1 {
            print_time_metrics(
                load_model_elapsed.as_secs_f64(),
                input_loaded_seconds,
                input_resized_seconds,
                load_input_elapsed.as_secs_f64(),
                inference_elapsed.as_secs_f64(),
                total_elapsed.as_secs_f64(),
            );
        } else {
            print_time_metrics(
                0.0,
                0.0,
                0.0,
                0.0,
                inference_duration.as_secs_f64(),
                trial_workload_duration.as_secs_f64(),
            );
        }

        print_top5(&output);
        println!("InferEdge trial {} end", trial);
    }
}

struct RuntimeArgs {
    model_path: String,
    image_path: String,
    trials: usize,
}

fn parse_runtime_args() -> RuntimeArgs {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model.pt> <image> [--trials N]", args[0]);
        std::process::exit(2);
    }

    let mut trials = 1;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--trials" => {
                if i + 1 >= args.len() {
                    panic!("Missing value after --trials");
                }
                trials = args[i + 1]
                    .parse::<usize>()
                    .unwrap_or_else(|_| panic!("Invalid --trials value: {}", args[i + 1]));
                i += 2;
            }
            option => panic!("Unknown option: {}", option),
        }
    }

    if trials == 0 {
        panic!("--trials must be at least 1");
    }

    RuntimeArgs {
        model_path: args[1].clone(),
        image_path: args[2].clone(),
        trials,
    }
}

fn print_time_metrics(
    model_loaded_seconds: f64,
    input_loaded_seconds: f64,
    input_resized_seconds: f64,
    input_ready_seconds: f64,
    inference_executed_seconds: f64,
    workload_seconds: f64,
) {
    println!(
        "Time until model loaded in seconds: {:.6}",
        model_loaded_seconds
    );
    println!(
        "Time until input loaded in seconds: {:.6}",
        input_loaded_seconds
    );
    println!(
        "Time until input resized in seconds: {:.6}",
        input_resized_seconds
    );
    println!(
        "Time until input ready in seconds: {:.6}",
        input_ready_seconds
    );
    println!(
        "Time until inference executed in seconds: {:.6}",
        inference_executed_seconds
    );
    println!(
        "Total workload duration in seconds: {:.6}",
        workload_seconds
    );
}

fn image_to_tensor(path: &str, height: u32, width: u32, start: Instant) -> (Tensor, f64, f64) {
    let mut file_img = File::open(path).unwrap();
    let mut img_buf = Vec::new();
    file_img.read_to_end(&mut img_buf).unwrap();
    let img = image::load_from_memory(&img_buf).unwrap().to_rgb8();

    let load_image_elapsed = start.elapsed();

    let resized =
        image::imageops::resize(&img, height, width, image::imageops::FilterType::Triangle);
    let resize_elapsed = start.elapsed();

    let mut flat_img = Vec::with_capacity((height * width * 3) as usize);
    for rgb in resized.pixels() {
        flat_img.push((rgb[0] as f32 / 255. - 0.485) / 0.229);
        flat_img.push((rgb[1] as f32 / 255. - 0.456) / 0.224);
        flat_img.push((rgb[2] as f32 / 255. - 0.406) / 0.225);
    }

    let tensor = Tensor::of_slice(&flat_img)
        .view([height as i64, width as i64, 3])
        .permute(&[2, 0, 1])
        .unsqueeze(0);

    (
        tensor,
        load_image_elapsed.as_secs_f64(),
        resize_elapsed.as_secs_f64(),
    )
}

fn print_top5(output: &Tensor) {
    let probabilities = output.softmax(-1, Kind::Float);
    let (scores, indices) = probabilities.topk(5, -1, true, true);
    let scores = scores.view([-1]);
    let indices = indices.view([-1]);

    for rank in 0..5 {
        let class_id = indices.int64_value(&[rank]);
        let score = scores.double_value(&[rank]);
        println!("   {}.) [{}]({:.4})", rank + 1, class_id, score);
    }
}
