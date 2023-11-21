#![allow(non_snake_case)]
mod linear_model;

fn main() {
    println!("Démarrage du programme...");
    match linear_model::run() {
        Ok(()) => println!("Modèle linéaire exécuté avec succès."),
        Err(e) => println!("Erreur lors de l'exécution du modèle linéaire : {}", e),
    }
    println!("Programme terminé.");
}