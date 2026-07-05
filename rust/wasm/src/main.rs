use image;
use std::env;
use std::fs::File;
use std::io::Read;
use std::time::Instant;
use wasi_nn;
mod imagenet_classes;

pub fn main() {
    let start = Instant::now();
    let runtime_args = parse_runtime_args();

    let model_bin_name: &str = &runtime_args.model_bin_name;
    let image_name: &str = &runtime_args.image_name;

    let graph = wasi_nn::GraphBuilder::new(
        wasi_nn::GraphEncoding::Pytorch,
        wasi_nn::ExecutionTarget::CPU,
    )
    .build_from_files([model_bin_name])
    .unwrap_or_else(|e| panic!("Failed to load model: {:?}", e));

    let mut context = graph.init_execution_context().unwrap();
    let load_model_elapsed = start.elapsed();
    println!(
        "Time until model loaded in seconds: {:.2}",
        load_model_elapsed.as_secs_f64()
    );

    // Load a tensor that precisely matches the graph input tensor (see
    let (tensor_data, input_loaded_seconds, input_resized_seconds) =
        image_to_tensor(image_name.to_string(), 224, 224, start);
    context
        .set_input(0, wasi_nn::TensorType::F32, &[1, 3, 224, 224], &tensor_data)
        .unwrap();
    let load_input_elapsed = start.elapsed();
    println!(
        "Time until input ready in seconds: {:.2}",
        load_input_elapsed.as_secs_f64()
    );

    for trial in 1..=runtime_args.trials {
        let trial_start = Instant::now();
        context.compute().unwrap();
        let inference_duration = trial_start.elapsed();
        let inference_elapsed = start.elapsed();

        // Retrieve the output.
        let mut output_buffer = vec![0f32; 1000];
        context.get_output(0, &mut output_buffer).unwrap();
        let results = sort_results(&output_buffer);
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
        print_top_results(&results);
        println!("InferEdge trial {} end", trial);
    }
}

struct RuntimeArgs {
    model_bin_name: String,
    image_name: String,
    trials: usize,
}

fn parse_runtime_args() -> RuntimeArgs {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        panic!("Usage: interpreted <model> <image> [--trials N]");
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
        model_bin_name: args[1].clone(),
        image_name: args[2].clone(),
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

fn print_top_results(results: &[InferenceResult]) {
    for i in 0..5 {
        println!(
            "   {}.) [{}]({:.4}){}",
            i + 1,
            results[i].0,
            results[i].1,
            imagenet_classes::IMAGENET_CLASSES[results[i].0]
        );
    }
}

// Sort the buffer of probabilities. The graph places the match probability for each class at the
// index for that class (e.g. the probability of class 42 is placed at buffer[42]). Here we convert
// to a wrapping InferenceResult and sort the results.
fn sort_results(buffer: &[f32]) -> Vec<InferenceResult> {
    let mut results: Vec<InferenceResult> = buffer
        .iter()
        .enumerate()
        .map(|(c, p)| InferenceResult(c, *p))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results
}

// Take the image located at 'path', open it, resize it to height x width, and then converts
// the pixel precision to FP32. The resulting BGR pixel vector is then returned.
fn image_to_tensor(path: String, height: u32, width: u32, start: Instant) -> (Vec<u8>, f64, f64) {
    let mut file_img = File::open(path).unwrap();
    let mut img_buf = Vec::new();
    file_img.read_to_end(&mut img_buf).unwrap();
    let img = image::load_from_memory(&img_buf).unwrap().to_rgb8();

    let load_image_elapsed = start.elapsed();

    let resized =
        image::imageops::resize(&img, height, width, ::image::imageops::FilterType::Triangle);
    let resize_elapsed = start.elapsed();

    let mut flat_img: Vec<f32> = Vec::new();
    for rgb in resized.pixels() {
        flat_img.push((rgb[0] as f32 / 255. - 0.485) / 0.229);
        flat_img.push((rgb[1] as f32 / 255. - 0.456) / 0.224);
        flat_img.push((rgb[2] as f32 / 255. - 0.406) / 0.225);
    }
    let bytes_required = flat_img.len() * 4;
    let mut u8_f32_arr: Vec<u8> = vec![0; bytes_required];

    for c in 0..3 {
        for i in 0..(flat_img.len() / 3) {
            // Read the number as a f32 and break it into u8 bytes
            let u8_f32: f32 = flat_img[i * 3 + c] as f32;
            let u8_bytes = u8_f32.to_ne_bytes();

            for j in 0..4 {
                u8_f32_arr[((flat_img.len() / 3 * c + i) * 4) + j] = u8_bytes[j];
            }
        }
    }

    return (
        u8_f32_arr,
        load_image_elapsed.as_secs_f64(),
        resize_elapsed.as_secs_f64(),
    );
}

// A wrapper for class ID and match probabilities.
#[derive(Debug, PartialEq)]
struct InferenceResult(usize, f32);
