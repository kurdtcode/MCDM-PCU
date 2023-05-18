import csv
import numpy as np
import pandas as pd
import os
import json
import tempfile
from flask import Flask, render_template, request
from pymcdm import methods as mcdm_methods
from pymcdm.methods import PROMETHEE_II
from pymcdm import normalizations as norm

app = Flask(__name__)

def csv_to_matrix(file):
    with open(file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)
        data = list(csv_reader)
    matrix = np.array(data, dtype=float)
    return matrix

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
            data = request.get_json()  # Mengambil data dalam format JSON

            weights = data.get('weights', [])
            criteria = data.get('criteria', [])

            # Ubah tipe data weights dan criteria menjadi float dan int
            weights = [float(weight) for weight in weights]
            criteria = [int(criterion) for criterion in criteria]

            # Lakukan normalisasi jika diperlukan
            weights = norm(weights)

            file = request.files['csv_file']
   
            if file:
                # Simpan file ke tempat temporary, soale nek pake request.file bakal error (minta fileStorage)
                temp_csv_filepath = os.path.join(tempfile.gettempdir(), file.filename)
                file.save(temp_csv_filepath)

                # Simpan data dalam bentuk dictionary
                data = {
                    'weights': weights,
                    'criteria': criteria
                }
                
                # Ubah data menjadi JSON string
                json_data = json.dumps(data)

                # Simpan JSON string ke tempat temporary yang berbeda
                with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                    temp_json_filepath = temp_file.name
                    temp_file.write(json_data)


                if(request.form['type'] == "vikor"):
                    print("vikor")
                    vikor_results = to_vikor(temp_csv_filepath, weights, criteria)
                    # promethee_results = to_promethee(temp_filepath)

                    # Yang temporary tadi hapus
                    os.remove(temp_csv_filepath)

                    # hasil di print ke html
                    return render_template('result.html', results=vikor_results, name="Vikor")
                elif(request.form["type"] == "topsis"):
                    print("prom")
                    topsis_results = to_topsis(temp_csv_filepath, weights, criteria)

                    # Yang temporary tadi hapus
                    os.remove(temp_csv_filepath)

                    # hasil di print ke html
                    return render_template('result.html', results=topsis_results, name="topsis")
                
                elif(request.form["type"] == "promethee"):
                    print("asw")
                    promethee_results = to_promethee(temp_csv_filepath, weights, criteria)

                    # Yang temporary tadi hapus
                    os.remove(temp_csv_filepath)

                    # hasil di print ke html
                    return render_template('result.html', results=promethee_results, name="promethee")
                
                # Hapus file temporary setelah selesai
                os.remove(temp_csv_filepath)
                os.remove(temp_json_filepath)
                
            # render tempat upload
            return render_template('upload.html')

def to_vikor(file, weights, criteria):
    matrix = csv_to_matrix(file)
    weight = np.array(weights)
    criteria = np.array(criteria)
    vikor_methods = {
        'VIKOR': mcdm_methods.VIKOR(),
        'MINMAX': mcdm_methods.VIKOR(norm.minmax_normalization),
        'MAX': mcdm_methods.VIKOR(norm.max_normalization),
        'SUM': mcdm_methods.VIKOR(norm.sum_normalization)
    }
    vikor_results = {}
    for name, function in vikor_methods.items():
        vikor_results[name] = function(matrix, weight, criteria)
    return vikor_results


def to_topsis(file, weights, criteria):
    matrix = csv_to_matrix(file)
    weight = np.array(weights)
    criteria = np.array(criteria)
    topsis_methods = {
        'minmax': mcdm_methods.TOPSIS(norm.minmax_normalization),
        'max': mcdm_methods.TOPSIS(norm.max_normalization),
        'sum': mcdm_methods.TOPSIS(norm.sum_normalization),
        'vector': mcdm_methods.TOPSIS(norm.vector_normalization),
    }
    topsis_results = {}
    for name, function in topsis_methods.items():
        topsis_results[name] = function(matrix, weight, criteria)
    return topsis_results

def to_promethee(file, weights, criteria):
    matrix = csv_to_matrix(file)
    weight = np.array(weights)
    criteria = np.array(criteria)
    preference_functions = ['usual', 'vshape', 'ushape', 'level', 'vshape_2']
    promethee_methods = {
        f'{pref}': mcdm_methods.PROMETHEE_II(preference_function=pref)
        for pref in preference_functions
    }
    p = np.random.rand(matrix.shape[1]) / 2
    q = np.random.rand(matrix.shape[1]) / 2 + 0.5
    promethee_results = {}
    for name, function in promethee_methods.items():
        promethee_results[name] = function(matrix, weight, criteria, p=p, q=q)
    return promethee_results

if __name__ == '__main__':
    app.run(debug=True)
