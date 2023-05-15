import csv
import numpy as np
from flask import Flask, render_template, request
from pymcdm.methods import vikor, promethee

app = Flask(__name__)

# membaca file CSV
def read_csv(file_path):
    with open(file_path, newline='') as csv_file:
        reader = csv.reader(csv_file)
        data = []
        for row in reader:
            data.append(list(map(float, row)))
    return np.array(data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/vikor', methods=['POST'])
def vikor_result():
    # membaca file CSV
    file = request.files['csv_file']
    data = read_csv(file)

    # normalisasi matriks keputusan
    sums = data.sum(axis=0)
    norm_data = data / sums

    # implementasi Vikor
    w = np.array([0.3, 0.4, 0.3]) # bobot kriteria
    S, R = vikor(norm_data, w)
    return render_template('vikor_result.html', S=S, R=R)

@app.route('/promethee', methods=['POST'])
def promethee_result():
    # membaca file CSV
    file = request.files['csv_file']
    data = read_csv(file)

    # normalisasi matriks keputusan
    sums = data.sum(axis=0)
    norm_data = data / sums

    # implementasi Promethee
    criteria = ['C1', 'C2', 'C3'] # nama kriteria
    weights = [0.3, 0.4, 0.3] # bobot kriteria
    criteria_directions = ['max', 'max', 'max'] # arah preferensi
    flow = 'dominance' # tipe aliran preferensi
    P, C = promethee(norm_data, weights, criteria_directions, flow)
    return render_template('promethee_result.html', P=P, C=C)

if __name__ == '__main__':
    app.run(debug=True)
