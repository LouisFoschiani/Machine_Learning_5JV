from flask import Flask, jsonify
from flask_cors import CORS
import ctypes

app = Flask(__name__)
CORS(app)

# Charger la DLL
ma_lib = ctypes.CDLL('C:\\Users\\Louis\\Documents\\GitHub\\machine_Learning_5JV\\dll_folder\\target\\debug\\dll_folder.dll')

@app.route('/calculer', methods=['GET'])
def calculer():
    # Définir le type de retour de la fonction
    ma_lib.calculer.restype = ctypes.c_int

    # Appeler la fonction
    resultat = ma_lib.calculer()

    # Retourner le résultat en JSON
    return jsonify(resultat=resultat)

if __name__ == '__main__':
    app.run(debug=True)
