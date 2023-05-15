import csv
import numpy as np
import pandas as pd
import os
from flask import Flask, render_template, request
from pymcdm import methods as mcdm_methods
from pymcdm import normalizations as norm
import tempfile
from tabulate import tabulate

app = Flask(__name__)

def csv_to_matrix(file):
    with open(file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)
        data = list(csv_reader)
    matrix = np.array(data, dtype=float)
    return matrix

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['csv_file']
        if file:
            # Simpan file ke tempat temporary, soale nek pake request.file bakal error (minta fileStorage)
            temp_filepath = os.path.join(tempfile.gettempdir(), file.filename)
            file.save(temp_filepath)
            
            # dibuat model vikor
            results = to_vikor(temp_filepath)
            
            # Yang temporary tadi hapus
            os.remove(temp_filepath)
            
            # hasil di print ke html
            return render_template('result.html', results=results)
    
    # render tempat upload
    return render_template('upload.html')

def to_vikor(file):
    matrix = csv_to_matrix(file)
    weight = np.array([0.4, 0.2, 0.05, 0.35])
    criteria = np.array([1, -1, -1, -1])
    vikor = mcdm_methods.VIKOR()
    vikor_methods = {
        'VIKOR': mcdm_methods.VIKOR(),
        'minmax': mcdm_methods.VIKOR(norm.minmax_normalization),
        'max': mcdm_methods.VIKOR(norm.max_normalization),
        'sum': mcdm_methods.VIKOR(norm.sum_normalization)
    }
    results = {}
    for name, function in vikor_methods.items():
        results[name] = function(matrix, weight, criteria)
    return results

if __name__ == '__main__':
    app.run()
