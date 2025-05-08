from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the saved models
with open('units_model.pkl', 'rb') as f:
    units_model = pickle.load(f)

with open('bill_model.pkl', 'rb') as f:
    bill_model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        past_units = float(request.form['past_units'])
        past_bill = float(request.form['past_bill'])
        house_size = request.form['house_size']
        num_people = int(request.form['num_people'])
        heavy_appliances = request.form['heavy_appliances']
        weather = request.form['weather']

        # Prepare data
        data = pd.DataFrame({
            'Past_Units': [past_units],
            'Past_Bill': [past_bill],
            'House_Size': [house_size],
            'Number_of_People': [num_people],
            'Heavy_Appliances': [heavy_appliances],
            'Weather': [weather]
        })

        # Encode
        data['House_Size'] = data['House_Size'].map({'small': 0, 'medium': 1, 'large': 2})
        data['Weather'] = data['Weather'].map({'cold': 0, 'moderate': 1, 'hot': 2})
        data['Heavy_Appliances'] = data['Heavy_Appliances'].map({'few': 0, 'many': 1})
        appliances_used = data['Heavy_Appliances'][0]

        # Predict
        future_units = units_model.predict(data)[0]
        future_bill = bill_model.predict(data)[0]

        # Efficiency and differences
        efficiency_score = round(future_units / num_people, 2)
        delta_units = round(future_units - past_units, 2)
        delta_bill = round(future_bill - past_bill, 2)
        target_units = round(future_units * 0.8, 2)
        target_bill = round(future_bill * 0.8, 2)

        # Tips and Recommendation
        recommendation = ""
        tips = []

        if delta_units > 100:
            recommendation = "High increase in usage detected!"
            tips.append("Review appliance usage habits.")
            tips.append("Consider using appliances during off-peak hours.")
        elif delta_units < -50:
            recommendation = "Energy consumption is decreasing. Good job!"
        else:
            recommendation = "Usage is stable. Try to reduce further."

        tips += [
            "Use energy-efficient appliances.",
            "Turn off devices when not in use."
        ]

        if appliances_used == 1:
            tips.append("Reduce simultaneous use of heavy appliances.")

        if hasattr(units_model, 'feature_importances_'):
            importances = units_model.feature_importances_
            features = data.columns
            feature_importance = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:3]
        else:
            feature_importance = []

        return render_template('index.html',
                               future_units=round(future_units, 2),
                               future_bill=round(future_bill, 2),
                               efficiency_score=efficiency_score,
                               target_units=target_units,
                               target_bill=target_bill,
                               delta_units=delta_units,
                               delta_bill=delta_bill,
                               recommendation=recommendation,
                               tips=tips,
                               feature_importance=feature_importance)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
