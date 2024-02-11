from flask import Flask, jsonify, request
from flask_cors import CORS
import ctypes
import subprocess
import json
import os
from werkzeug.utils import secure_filename
from PIL import Image


app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Charger la DLL
ma_lib = ctypes.CDLL('C:\\Users\\Louis\\Documents\\GitHub\\Machine_Learning_5JV\\dll_folder\\target\\release\\dll_folder.dll')


command = """
cd .. && cargo build --release && cd src && python app.py
"""

@app.route('/reload_dll', methods=['POST'])
def reload_dll():
   subprocess.run(['C:\\Users\\Louis\\Documents\\GitHub\\Machine_Learning_5JV\\dll_folder\\restart_app.bat'], shell=True)
   return jsonify({"message": "DLL reload triggered successfully"}), 200


@app.route('/run_algo', methods=['POST'])
def run_algo():

    data = request.json
    with open('config.json', 'w') as config_file:
        json.dump(data, config_file)

    with open('prediction.json', 'r') as file:
        prediction_data = json.load(file)

    resultat = ma_lib.run_algo()

    # Retourner le résultat en JSON
    return jsonify(prediction_data)


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        folder_path = 'C:\\Users\\Louis\\Documents\\GitHub\\Machine_Learning_5JV\\dll_folder\\src\\image_predict'
        clear_directory(folder_path)
        save_path = os.path.join(folder_path, filename)
        file.save(save_path)

        # Récupérer la taille d'image sélectionnée
        image_size = request.form.get('size', '16') # Utiliser '16' comme valeur par défaut
        is32 = True if image_size == '32' else False
    
        # Appeler la fonction de redimensionnement avec la taille correcte
        resize_images_in_directory(folder_path, is32)
        rename_files_in_directory(folder_path)

        return jsonify({'message': 'Image successfully uploaded and processed'})

def clear_directory(directory_path):

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            print(f'Supprimé: {file_path}')
        except Exception as e:
            print(f'Échec de la suppression de {file_path}. Raison: {e}')

def resize_images_in_directory(directory_path, is32):
    # Vérifier si le chemin du dossier existe
    if not os.path.exists(directory_path):
        print(f"Le dossier spécifié n'existe pas: {directory_path}")
        return

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            try:
                # Ouvrir l'image
                with Image.open(file_path) as img:
                    # Redimensionner l'image
                    img_resized = img.resize((32 if is32 else 16, 32 if is32 else 16))

                    # Construire le nouveau chemin de fichier avec l'extension .png
                    new_file_path = os.path.splitext(file_path)[0] + ".png"

                    # Sauvegarder l'image redimensionnée au format PNG (écrase l'ancienne image si elle a le même nom)
                    img_resized.save(new_file_path, format="PNG")

                    print(f"Image redimensionnée et sauvegardée au format PNG: {new_file_path}")
            except IOError as e:
                print(f"Erreur lors du traitement de l'image {filename}: {e}")


def rename_files_in_directory(directory_path):
    # Vérifier si le chemin du dossier existe
    if not os.path.exists(directory_path):
        print(f"Le dossier spécifié n'existe pas: {directory_path}")
        return

    for filename in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, filename)):
            # Séparer le nom de fichier et son extension
            file_name, file_extension = os.path.splitext(filename)

            # Générer un nouveau nom de fichier aléatoire
            new_file_name = ''.join("Image_Test_Prediction")

            # Construire le chemin complet des anciens et nouveaux fichiers
            old_file = os.path.join(directory_path, filename)
            new_file = os.path.join(directory_path, new_file_name + file_extension)

            # Renommer le fichier
            os.rename(old_file, new_file)
            print(f"Renommé: {filename} en {new_file_name}{file_extension}")


if __name__ == '__main__':

    app.run(debug=True)
    directory_path = 'C:\\Users\\Louis\\Documents\\GitHub\\Machine_Learning_5JV\\dll_folder\\src\\image_predict'
    resize_images_in_directory(directory_path, True)
    rename_files_in_directory(directory_path)