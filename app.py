import numpy as np
import pickle
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained Logistic Regression model
with open('modelLOR.pkl', 'rb') as f:
    model = pickle.load(f)

# HTML Template with a vibrant, colorful, and responsive UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Health Risk & Diabetes Prediction Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --card-bg: rgba(255, 255, 255, 0.95);
        }
        body {
            font-family: 'Poppins', sans-serif;
            background: var(--primary-gradient);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .container {
            width: 100%;
            max-width: 800px;
            background: var(--card-bg);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
        }
        h2 {
            text-align: center;
            color: #4a3f89;
            margin-bottom: 10px;
            font-weight: 600;
        }
        p.subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .grid-form {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        .form-group.full-width {
            grid-column: span 2;
        }
        label {
            font-size: 13px;
            font-weight: 600;
            color: #444;
            margin-bottom: 8px;
        }
        input, select {
            padding: 12px 15px;
            border: 2px solid #e1e1e1;
            border-radius: 10px;
            font-size: 14px;
            transition: all 0.3s ease;
            outline: none;
            background: #fafafa;
        }
        input:focus, select:focus {
            border-color: #667eea;
            background: #fff;
            box-shadow: 0 0 8px rgba(102, 126, 234, 0.3);
        }
        .btn-submit {
            grid-column: span 2;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 14px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            margin-top: 10px;
        }
        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(245, 87, 108, 0.4);
        }
        .result-card {
            margin-top: 25px;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 18px;
            font-weight: 600;
            animation: fadeIn 0.5s ease-in-out;
        }
        .result-positive {
            background: #ffdde1;
            color: #d90429;
            border: 2px solid #ef233c;
        }
        .result-negative {
            background: #e2fbc7;
            color: #2b9348;
            border: 2px solid #55a630;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @media(max-width: 600px) {
            .grid-form { grid-template-columns: 1fr; }
            .form-group.full-width { grid-column: span 1; }
            .btn-submit { grid-column: span 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Health Assessment Portal</h2>
        <p class="subtitle">Fill out the parameters below to evaluate your health risk profile instantly.</p>
        
        <form method="POST" class="grid-form">
            <div class="form-group">
                <label>Age</label>
                <input type="number" step="any" name="age" required placeholder="e.g. 45">
            </div>
            <div class="form-group">
                <label>Gender</label>
                <select name="gender">
                    <option value="1">Male</option>
                    <option value="0">Female</option>
                </select>
            </div>
            <div class="form-group">
                <label>City Code / Identifier</label>
                <input type="number" step="any" name="city" required placeholder="e.g. 1">
            </div>
            <div class="form-group">
                <label>BMI</label>
                <input type="number" step="any" name="bmi" required placeholder="e.g. 24.5">
            </div>
            <div class="form-group">
                <label>Family History of Diabetes</label>
                <select name="family_history_diabetes">
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>
            <div class="form-group">
                <label>Physical Activity Level</label>
                <input type="number" step="any" name="physical_activity_level" required placeholder="e.g. 2">
            </div>
            <div class="form-group">
                <label>Diet Type</label>
                <input type="number" step="any" name="diet_type" required placeholder="e.g. 1">
            </div>
            <div class="form-group">
                <label>Smoking Status</label>
                <select name="smoking_status">
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>
            <div class="form-group">
                <label>Alcohol Consumption</label>
                <select name="alcohol_consumption">
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>
            <div class="form-group">
                <label>Hours Sleep Per Night</label>
                <input type="number" step="any" name="hours_sleep_per_night" required placeholder="e.g. 7">
            </div>
            <div class="form-group">
                <label>Stress Level</label>
                <input type="number" step="any" name="stress_level" required placeholder="e.g. 3">
            </div>
            <div class="form-group">
                <label>Fasting Blood Sugar</label>
                <input type="number" step="any" name="fasting_blood_sugar" required placeholder="e.g. 95">
            </div>
            <div class="form-group">
                <label>HbA1c Level</label>
                <input type="number" step="any" name="hba1c_level" required placeholder="e.g. 5.6">
            </div>
            <div class="form-group">
                <label>Blood Pressure (Systolic)</label>
                <input type="number" step="any" name="blood_pressure_systolic" required placeholder="e.g. 120">
            </div>
            <div class="form-group">
                <label>Blood Pressure (Diastolic)</label>
                <input type="number" step="any" name="blood_pressure_diastolic" required placeholder="e.g. 80">
            </div>
            <div class="form-group">
                <label>Waist Circumference (cm)</label>
                <input type="number" step="any" name="waist_circumference_cm" required placeholder="e.g. 85">
            </div>
            <div class="form-group full-width">
                <label>Income Bracket</label>
                <input type="number" step="any" name="income_bracket" required placeholder="e.g. 2">
            </div>
            
            <button type="submit" class="btn-submit">Run Prediction</button>
        </form>

        {% if prediction_text is not none %}
            <div class="result-card {% if prediction_text == 1 %}result-positive{% else %}result-negative{% endif %}">
                {% if prediction_text == 1 %}
                    ⚠️ Prediction: High Risk / Positive for Condition Detected.
                {% else %}
                    ✅ Prediction: Low Risk / Negative for Condition.
                {% endif %}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    if request.method == 'POST':
        try:
            # Extract features matching the exact order trained in your model
            feature_names = [
                'age', 'gender', 'city', 'bmi', 'family_history_diabetes', 
                'physical_activity_level', 'diet_type', 'smoking_status', 
                'alcohol_consumption', 'hours_sleep_per_night', 'stress_level', 
                'fasting_blood_sugar', 'hba1c_level', 'blood_pressure_systolic', 
                'blood_pressure_diastolic', 'waist_circumference_cm', 'income_bracket'
            ]
            
            # Map input fields to float array
            features = [float(request.form[f]) for f in feature_names]
            final_features = np.array([features])
            
            # Predict using the loaded model
            prediction = model.predict(final_features)
            prediction_text = int(prediction[0])
        except Exception as e:
            prediction_text = f"Error in processing input: {str-e}"

    return render_template_string(HTML_TEMPLATE, prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
