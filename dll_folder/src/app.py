from flask import Flask, jsonify, request
from flask_cors import CORS
import ctypes
import subprocess
import json
import os

app = Flask(__name__)
cors = CORS(app, resources={r"/run_algo": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

# Charger la DLL
ma_lib = ctypes.CDLL('C:\\wamp64\\www\\machine_learning_5JV\\dll_folder\\target\\release\\dll_folder.dll')

def kill_process_on_port(port):
    try:
        # Trouver le PID du processus écoutant sur le port donné
        result = subprocess.check_output(f"netstat -aon | findstr :{port}", shell=True)
        line = result.decode().strip().split('\n')[0]
        pid = line.split()[-1]
        # Tuer le processus par son PID
        os.system(f"taskkill /F /PID {pid}")
        print(f"Process on port {port} killed.")
    except Exception as e:
        print(f"Could not kill process on port {port}: {e}")

# Exemple d'utilisation : tuer le processus sur le port 5000 avant de continuer


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

    #kill_process_on_port(5000)
    subprocess.run(command, shell=True, check=True)

    resultat = ma_lib.run_algo()

    # Retourner le résultat en JSON
    return jsonify(resultat=resultat)

if __name__ == '__main__':
    app.run(debug=True)