from flask import Flask, jsonify, request
from flask_cors import CORS
import ctypes
import subprocess
import json

app = Flask(__name__)
cors = CORS(app, resources={r"/run_algo": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

# Charger la DLL
ma_lib = ctypes.CDLL('C:\\wamp64\\www\\machine_learning_5JV\\dll_folder\\target\\release\\dll_folder.dll')

command = """
cd .. &&
cargo build --release &&
cd src &&
python app.py
"""

@app.route('/run_algo', methods=['POST'])
def run_algo():

    data = request.json
    with open('config.json', 'w') as config_file:
        json.dump(data, config_file)

    subprocess.run(command, shell=True, check=True)

    resultat = ma_lib.run_algo()

    # Retourner le résultat en JSON
    return jsonify(resultat=resultat)

if __name__ == '__main__':
    app.run(debug=True)