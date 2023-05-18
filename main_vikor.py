import csv
import numpy as np
import pandas as pd
import os
from flask import Flask, render_template, request
from pymcdm import methods as mcdm_methods
from pymcdm.methods import PROMETHEE_II
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
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['csv_file']
        if file:
            # Simpan file ke tempat temporary, soale nek pake request.file bakal error (minta fileStorage)
            temp_filepath = os.path.join(tempfile.gettempdir(), file.filename)
            file.save(temp_filepath)

            if(request.form['type'] == "vikor"):
                print("vikor")
                vikor_results = to_vikor(temp_filepath)
                # promethee_results = to_promethee(temp_filepath)

                # Yang temporary tadi hapus
                os.remove(temp_filepath)

                # hasil di print ke html
                return render_template('result.html', vikor_results=vikor_results)
            elif(request.form["type"] == "promethee"):
                print("prom")
                promethee_results = to_promethee(temp_filepath)
                # promethee_results = to_promethee(temp_filepath)

                # Yang temporary tadi hapus
                os.remove(temp_filepath)

                # hasil di print ke html
                return render_template('result.html', vikor_results=promethee_results)

            # dibuat model vikor

    # render tempat upload
    return render_template('upload.html')


def to_vikor(file):
    matrix = csv_to_matrix(file)
    weight = np.array([0.4, 0.2, 0.05, 0.35])
    criteria = np.array([1, -1, -1, -1])
    vikor = mcdm_methods.VIKOR()
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


def to_promethee(file):
    matrix = csv_to_matrix(file)
    criteria = np.array([1, -1, -1, -1])
    promethee_methods = {
        'PROMETHEE II': mcdm_methods.promethee.PROMETHEE_II()
    }
    promethee_results = {}
    for name, method in promethee_methods.items():
        rankings = method.compute(matrix, criteria)
        promethee_results[name] = rankings
    return promethee_results


if __name__ == '__main__':
    app.run()
