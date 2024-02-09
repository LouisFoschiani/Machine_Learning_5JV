use std::fs;
use std::path::Path;

fn main() {
    let repertoire_source = "D:/GitHub/Machine_Learning_5JV/images/volleyball ball/";

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
                        let new_name = format!("volleyball_{}.{}", index, extension_str);
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
    let image_extensions = vec!["jpg", "jpeg", "png", "bmp", "gif", "tiff"];
    image_extensions.contains(&extension.to_lowercase().as_str())
}
