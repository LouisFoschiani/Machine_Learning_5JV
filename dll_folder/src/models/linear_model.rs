use std::collections::HashMap;
#[warn(non_snake_case)]

use image::{self, GenericImageView};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path};
use plotters::prelude::*;


fn plot_errors(train_errors: &Vec<f32>, test_errors: &Vec<f32>, index: i32, target: String, nonTarget: String) -> Result<(), Box<dyn std::error::Error>> {

    let name = format!("index-{}.png", index);
    let title = format!("Training and Test Errors Over Iteration: {} / {}", target, nonTarget);
    let root = BitMapBackend::new(&name, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 20).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..train_errors.len(), 0f32..1f32)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        train_errors.iter().enumerate().map(|(i, &err)| (i, err)),
        &RED,
    ))?.label("Training Error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart.draw_series(LineSeries::new(
        test_errors.iter().enumerate().map(|(i, &err)| (i, err)),
        &BLUE,
    ))?.label("Test Error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart.configure_series_labels().border_style(&BLACK).draw()?;
    Ok(())
}


fn load_image_data(base_path: &Path, target_category: &str, non_target_categories: &Vec<String>) -> io::Result<(Vec<Vec<f32>>, Vec<i32>)> {
    let mut features = Vec::new();
    let mut labels = Vec::new();

    // Load target category images
    let target_path = base_path.join(target_category);
    for entry in fs::read_dir(target_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            let image_features = process_image(&path)?;
            features.push(image_features);
            labels.push(1);
        }
    }

    // Load non-target category images

    for non_target_category in non_target_categories.iter() {

        let non_target_path = base_path.join(non_target_category);
        for entry in fs::read_dir(non_target_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                let image_features = process_image(&path)?;
                features.push(image_features);
                labels.push(-1);
            }
        }

    }



    Ok((features, labels))
}

fn process_image(path: &Path) -> io::Result<Vec<f32>> {
    let img = image::open(path).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    let features = img.pixels()
        .flat_map(|(_, _, pixel)| pixel.0.to_vec())
        .map(|value| value as f32 / 255.0)
        .collect();
    Ok(features)
}

fn init_model_weights(cols_x_len: usize, rows_w_len: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let between = Uniform::from(-1.0..1.0);
    let mut weights = vec![0.0; rows_w_len];
    for weight in weights.iter_mut().take(cols_x_len + 1) {
        *weight = between.sample(&mut rng);
    }
    weights
}

fn predict_linear_model_classification(model_weights: &[f32], inputs: &[f32]) -> i32 {
    let mut res = 0.0;
    for (&weight, &input) in model_weights.iter().skip(1).zip(inputs.iter()) {
        res += weight * input;
    }
    let total_sum = model_weights[0] + res;
    if total_sum >= 0.0 {
        1
    } else {
        -1
    }
}

fn train_linear_model(x: &[Vec<f32>], y: &[i32], w: &mut [f32], rows_x_len: i32, rows_w_len: i32, iter: i32) {
    let learning_rate = 0.001;

    for _ in 0..iter {
        let k = rand::thread_rng().gen_range(0..rows_x_len) as usize;
        let gxk = predict_linear_model_classification(w, &x[k]) as f32;
        let yk = y[k] as f32;
        let diff = yk - gxk;

        w[0] = w[0] + learning_rate * diff;
        for j in 1..rows_w_len as usize {
            w[j] = w[j] + learning_rate * diff * x[k][j - 1];
        }
    }
}


fn save_model_linear(model_weight: &[f32], file_path: &Path, efficiency: f32) -> Result<(), io::Error> {
    let mut file = File::create(file_path)?;
    writeln!(file, "-- Efficiency --\n{}", efficiency)?;
    writeln!(file, "-- Weights --")?;
    for &weight in model_weight {
        writeln!(file, "{{{}}}", weight)?;
    }
    Ok(())
}

fn load_model_weights(file_path: &Path, colX: i32, rowW: i32) -> io::Result<(Vec<f32>, f32)> {

    let mut weights = Vec::new();
    let mut efficiency = 0.0;

    if fs::metadata(file_path).is_ok() {

        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut read_efficiency = false;

        for line in reader.lines() {
            let line = line?;
            if line.starts_with("-- Efficiency --") {
                read_efficiency = true;
                continue;
            }

            if read_efficiency {
                efficiency = line.parse::<f32>().unwrap_or_else(|_| 0.0);
                read_efficiency = false;
                continue;
            }

            if line.starts_with("{") && line.ends_with("}") {
                let weight_str = &line[1..line.len()-1];
                match weight_str.trim().parse::<f32>() {
                    Ok(weight) => weights.push(weight),
                    Err(_) => continue,
                }
            }
        }
    } else {
        weights = init_model_weights(colX as usize, rowW as usize)
    }



    Ok((weights, efficiency))
}

fn set_var(x: &Vec<Vec<f32>>) -> (i32, i32, i32) {
    let rows_x_len = x.len();
    let cols_x_len = if !x.is_empty() { x[0].len() } else { 0 };
    let rows_w_len = cols_x_len + 1;
    (rows_x_len as i32, cols_x_len as i32, rows_w_len as i32)
}

pub fn run_linear_model(mode: &str, category:usize) -> io::Result<()> {

    let CHECK;

    if(mode == "train") {CHECK = false;}
    else{CHECK = true;};

    let predict_category = category;

    let iterations = 150;

    let mut weights_file_path_List: Vec<String> = Vec::new();
    weights_file_path_List.push("linear_model_weights_0.txt".to_string());
    weights_file_path_List.push("linear_model_weights_1.txt".to_string());
    weights_file_path_List.push("linear_model_weights_2.txt".to_string());

    let base_training_path = Path::new("..\\images_32\\Training");
    let base_test_path = Path::new("..\\images_32\\Test");

    let mut target_List: Vec<String> = Vec::new();
    target_List.push("Tomato".to_string());
    target_List.push("Orange".to_string());
    target_List.push("Aubergine".to_string());

    let mut non_target_List: Vec<Vec<String>> = Vec::new();
    non_target_List.push(Vec::new());
    non_target_List.push(Vec::new());
    non_target_List.push(Vec::new());

    non_target_List[0].push("Orange".to_string());
    non_target_List[0].push("Aubergine".to_string());
    non_target_List[1].push("Tomato".to_string());
    non_target_List[1].push("Aubergine".to_string());
    non_target_List[2].push("Orange".to_string());
    non_target_List[2].push("Tomato".to_string());



    if CHECK == true{

        let mut result: Vec<String> = Vec::new();

        let image_path = Path::new("..\\images_32\\CHECK\\Aubergine\\aubergine1.jpg");



        let (train_features, _) = load_image_data(base_training_path, &target_List[predict_category], &non_target_List[predict_category])?;

        let (_, colsXLen, rowsWLen) = set_var(&train_features);
        let (w, _) = load_model_weights(Path::new(&weights_file_path_List[predict_category]), colsXLen, rowsWLen)?;

        // Traitement de l'image à tester
        let image_features = process_image(&image_path)?;

        // Faire une prédiction
        let prediction = predict_linear_model_classification(&w, &image_features);


        // Afficher le résultat
        if prediction == 1 {
            result.push(target_List[predict_category].to_string());
        } else {
            result.push("Image inconue".to_string());
        }


        let mut resultCount = HashMap::new();

        // Comptage des occurrences de chaque élément
        for element in &result {
            *resultCount.entry(element).or_insert(0) += 1;
        }

        let mut max_element = None;
        let mut max_count = 0;
        for (element, count) in resultCount {
            if count > max_count {
                max_count = count;
                max_element = Some(element);
            }
        }

        match max_element {
            Some(element) => println!("'{}'", element),
            None => println!("La liste est vide."),
        }


    }else{

        let mut train_errors: Vec<Vec<f32>> = Vec::new();
        let mut test_errors: Vec<Vec<f32>> = Vec::new();

        for i in 0..weights_file_path_List.len() {
            train_errors.push(Vec::new());
            test_errors.push(Vec::new());
        }



        for iter in 0..iterations {


            println!("------- ITERATION {} -------", iter);
            println!("\n");

            // TRAINING
            for i in 0..weights_file_path_List.len() {

                let (train_features, train_labels) = load_image_data(base_training_path, &target_List[i], &non_target_List[i])?;

                let (rowsXLen, colsXLen, rowsWLen) = set_var(&train_features);

                let (mut w, max_percent) = load_model_weights(Path::new(&weights_file_path_List[i]), colsXLen, rowsWLen)?;

                train_linear_model(&train_features, &train_labels, &mut w, rowsXLen, rowsWLen, 1000);

                let mut final_result = 0;

                for i in 0..rowsXLen {
                    let result = predict_linear_model_classification(&w, &train_features[i as usize]);
                    if result == train_labels[i as usize] {
                        final_result += 1
                    }
                }
                println!("Training Result : {} / {} = {}%", final_result, train_labels.len(), final_result as f32 / train_labels.len() as f32 * 100.0);


                // TESTING
                final_result = 0;
                let (train_features, train_labels) = load_image_data(base_test_path, &target_List[i], &non_target_List[i])?;
                let (rowsXLen, colsXLen, rowsWLen) = set_var(&train_features);

                let mut final_result = 0;

                for i in 0..rowsXLen {
                    let result = predict_linear_model_classification(&w, &train_features[i as usize]);
                    if result == train_labels[i as usize] {
                        final_result += 1
                    }
                }

                println!("Test Result : {} / {} = {}%", final_result, train_labels.len(), final_result as f32 / train_labels.len() as f32 * 100.0);
                let success = final_result as f32 / train_labels.len() as f32 * 100.0;
                test_errors[i].push(1.0 - success/100.0);

                if success > max_percent{
                    println!("Update weights : {} > {}", success, max_percent);
                    save_model_linear(&w, Path::new(&weights_file_path_List[i]), success).expect("Erreur lors de l'enregistrement des poids");
                }

                println!("\n");



                final_result = 0;
                let (train_features, train_labels) = load_image_data(base_training_path, &target_List[i], &non_target_List[i])?;
                let (rowsXLen, colsXLen, rowsWLen) = set_var(&train_features);

                let mut final_result = 0;

                for i in 0..rowsXLen {
                    let result = predict_linear_model_classification(&w, &train_features[i as usize]);
                    if result == train_labels[i as usize] {
                        final_result += 1
                    }
                }
                let success = final_result as f32 / train_labels.len() as f32 * 100.0;
                train_errors[i].push(1.0 - success/100.0);


            }

        }

        for i in 0..weights_file_path_List.len() {
            plot_errors(&train_errors[i], &test_errors[i], i as i32, target_List[i].to_string(), format!("({0} | {1})", non_target_List[i][0].to_string(), non_target_List[i][1].to_string())).expect("Erreur lors de la création du graphique");
        }


    }


    Ok(())
}
