from flask import Flask, jsonify, request
from flask_cors import CORS
import ctypes
import subprocess
import json
import os

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

if __name__ == '__main__':
    app.run(debug=True)