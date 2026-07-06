use image;
use std::env;
use std::fs::File;
use std::io::Read;
use std::time::Instant;
use tch::{CModule, Kind, Tensor};

fn main() {
    let start = Instant::now();
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <model.pt> <image>", args[0]);
        std::process::exit(2);
    }

    let model_path = &args[1];
    let image_path = &args[2];

    let model = CModule::load(model_path).unwrap_or_else(|e| {
        panic!("Failed to load model {}: {:?}", model_path, e);
    });
    let load_model_elapsed = start.elapsed();
    println!(
        "Time until model loaded in seconds: {:.2}",
        load_model_elapsed.as_secs_f64()
    );

    let input = image_to_tensor(image_path, 224, 224, start);
    let load_input_elapsed = start.elapsed();
    println!(
        "Time until input ready in seconds: {:.2}",
        load_input_elapsed.as_secs_f64()
    );

    let output = model.forward_ts(&[input]).unwrap_or_else(|e| {
        panic!("Failed to execute inference: {:?}", e);
    });
    let inference_elapsed = start.elapsed();
    println!(
        "Time until inference executed in seconds: {:.2}",
        inference_elapsed.as_secs_f64()
    );

    print_top5(&output);

    let elapsed = start.elapsed();
    println!(
        "Total workload duration in seconds: {:.2}",
        elapsed.as_secs_f64()
    );
}

fn image_to_tensor(path: &str, height: u32, width: u32, start: Instant) -> Tensor {
    let mut file_img = File::open(path).unwrap();
    let mut img_buf = Vec::new();
    file_img.read_to_end(&mut img_buf).unwrap();
    let img = image::load_from_memory(&img_buf).unwrap().to_rgb8();

    let load_image_elapsed = start.elapsed();
    println!(
        "Time until input loaded in seconds: {:.2}",
        load_image_elapsed.as_secs_f64()
    );

    let resized =
        image::imageops::resize(&img, height, width, image::imageops::FilterType::Triangle);
    let resize_elapsed = start.elapsed();
    println!(
        "Time until input resized in seconds: {:.2}",
        resize_elapsed.as_secs_f64()
    );

    let mut flat_img = Vec::with_capacity((height * width * 3) as usize);
    for rgb in resized.pixels() {
        flat_img.push((rgb[0] as f32 / 255. - 0.485) / 0.229);
        flat_img.push((rgb[1] as f32 / 255. - 0.456) / 0.224);
        flat_img.push((rgb[2] as f32 / 255. - 0.406) / 0.225);
    }

    Tensor::of_slice(&flat_img)
        .view([height as i64, width as i64, 3])
        .permute(&[2, 0, 1])
        .unsqueeze(0)
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
