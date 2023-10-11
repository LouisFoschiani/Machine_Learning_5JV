use image::{GenericImageView, DynamicImage};
use std::fs;
use std::path::Path;

fn main() {
    let input_directory = "D:/GitHub/Machine_Learning_5JV/images/volleyball ball/";
    let output_directory = "D:/GitHub/Machine_Learning_5JV/images/volleyball ball/";

    if !Path::new(output_directory).exists() {
        fs::create_dir(output_directory).expect("Failed to create the output directory.");
    }
    let image_extensions = vec!["png", "bmp", "gif", "tiff", "jpeg", "webp"];

    for entry in fs::read_dir(input_directory).expect("Failed to read directory") {
        if let Ok(entry) = entry {
            let path = entry.path();
            if let Some(extension) = path.extension() {
                let extension_str = extension.to_string_lossy().to_lowercase();
                if !image_extensions.contains(&&*extension_str) {
                    continue; // Ignore les fichiers avec d'autres extensions.
                }
            } else {
                continue; // Ignore les fichiers sans extension.
            }
            let img = image::open(&path);
            match img {
                Ok(img) => {
                    let output_path = Path::new(output_directory)
                        .join(path.file_stem().unwrap())
                        .with_extension("jpg");
                    img.save(output_path).expect("Failed to save image as JPEG");
                }
                Err(err) => {
                    println!("Failed to open image {}: {}", path.display(), err);
                }
            }
        }
    }

    println!("Image conversion completed.");
}

