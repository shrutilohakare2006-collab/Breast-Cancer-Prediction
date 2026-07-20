from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model/breastCancer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    clump_thickness = float(request.form['Clump_Thickness'])
    uniformity_cell_size = float(request.form['Uniformity_Cell_Size'])
    uniformity_cell_shape = float(request.form['Uniformity_Cell_Shape'])
    marginal_adhesion = float(request.form['Marginal_Adhesion'])
    single_epithelial_cell_size = float(request.form['Single_Epithelial_Cell_Size'])
    bare_nuclei = float(request.form['Bare_Nuclei'])
    bland_chromatin = float(request.form['Bland_Chromatin'])
    normal_nucleoli = float(request.form['Normal_Nucleoli'])
    mitoses = float(request.form['Mitoses'])

    features = np.array([[
        clump_thickness,
        uniformity_cell_size,
        uniformity_cell_shape,
        marginal_adhesion,
        single_epithelial_cell_size,
        bare_nuclei,
        bland_chromatin,
        normal_nucleoli,
        mitoses
    ]])

    prediction = model.predict(features)

    if prediction[0] == 2:
        result = "Benign (Non-Cancerous)"
    else:
        result = "Malignant (Cancerous)"

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)