from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
import os
from methods.Quantum import Quantum_Unit
import yaml
from methods.ML import ML_unit
from methods.DataGen import Data_Generator

app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_aqui'

# Configuración de directorios
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static', 'images')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Variables globales
filename_global = None
model_global = False
MLU = None

@app.route('/', methods=['GET'])
def index():
    graph_path = os.path.join(STATIC_FOLDER, 'graph.png')
    graph_exists = os.path.exists(graph_path)
    timestamp = datetime.now().timestamp() if graph_exists else 0
    return render_template('index.html', graph_exists=graph_exists, timestamp=timestamp)

@app.route('/upload', methods=['POST'])
def upload():
    global filename_global
    file = request.files.get('file')
    if not file or file.filename == '':
        flash('No se seleccionó ningún archivo', 'error')
        return redirect(url_for('index'))

    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        filename_global = filepath
        flash('Archivo subido correctamente', 'success')
    except Exception as e:
        flash(f'Error al subir el archivo: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/train', methods=['POST'])
def train():
    global model_global, MLU, filename_global
    if not filename_global or not os.path.exists(filename_global):
        flash('Primero debe subir un archivo válido', 'error')
        return redirect(url_for('index'))

    try:
        with open(filename_global, 'r') as f:
            config = yaml.safe_load(f)

        elements = config['elements']
        distance_range = config['dis']
        basis = config['bas']

        data = Data_Generator(elements, distance_range, basis)
        deltas, gstates = data.get_data()

        qu = Quantum_Unit(gstates, deltas, 8)
        qu.create_entangler(8)
        qu.all_states()

        x = qu.x
        y = qu.y
        d = np.linspace(distance_range[0], distance_range[1], 200)

        MLU = ML_unit(x, y, d)
        MLU.train()

        model_global = True
        flash('Modelo entrenado correctamente', 'success')
    except Exception as e:
        flash(f'Error al entrenar el modelo: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/evaluate', methods=['POST'])
def evaluate():
    global MLU, model_global
    if not model_global or MLU is None:
        flash('Primero debe entrenar el modelo', 'error')
        return redirect(url_for('index'))

    try:
        graph_path = os.path.join(STATIC_FOLDER, 'graph.png')
        MLU.evaluate()
        MLU.visualize_vs_distance(graph_path)
        flash('Evaluación completada correctamente', 'success')
    except Exception as e:
        flash(f'Error al evaluar el modelo: {str(e)}', 'error')

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
