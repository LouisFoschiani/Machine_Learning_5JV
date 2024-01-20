use std::fs;
use std::path::Path;

pub(crate) fn main() {
    let mut file_type = "";
    let repertoire_source = "C:\\Users\\Louis\\";

    if !Path::new(repertoire_source).exists() {
        eprintln!("Le répertoire source n'existe pas.");
        return;
    }

    let mut index = 0;

    for entry in fs::read_dir(repertoire_source).expect("Impossible de lire le répertoire") {
        if let Ok(entry) = entry {
            let path = entry.path();
            if let Some(extension) = path.extension() {
                if let Some(extension_str) = extension.to_str() {
                    if is_image_extension(extension_str) {
                        if extension_str.to_lowercase() == "mov" {
                            file_type = "video_";
                        }
                        else{
                            file_type = "image_";
                        }
                        let new_name = format!("{}{}.{}", file_type, index, extension_str);

                        index += 1;
                        let new_path = path.with_file_name(new_name);
                        if let Err(err) = fs::rename(&path, &new_path) {
                            eprintln!("Échec de la rénomination de {:?} : {}", path, err);
                        } else {
                            println!("Renommé {:?} en {:?}", path, new_path);
                            println!("Index {:?}", index);
                        }
                    }
                }
            }
        }
    }
    index = 0;
}

fn is_image_extension(extension: &str) -> bool {
    let image_extensions = vec!["jpg", "jpeg", "png", "bmp", "gif", "tiff", "heic", "mov"];
    image_extensions.contains(&extension.to_lowercase().as_str())
}
