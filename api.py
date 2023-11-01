from flask import Flask, request, render_template
import pickle
import joblib


app = Flask(__name__)

# Load the trained model
model = joblib.load('dtr.pkl')
with open('dtr.pkl', 'rb') as file:
    model = pickle.load(file)
with open('state.pkl', 'rb') as f:
    state_dict = pickle.load(f)
with open('crop.pkl', 'rb') as f:
    crop_dict = pickle.load(f)
with open('district.pkl', 'rb') as f:
    district_dict = pickle.load(f)
with open('season.pkl', 'rb') as f:
    season_dict = pickle.load(f)
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        try:
            Crop_Year = int(request.form['Crop_Year'])
            State_Name = state_dict[request.form['slct1']]
            District_Name = district_dict[request.form['slct2']]
            Season_Type = season_dict[request.form['slct3']]
            Crop_Name = crop_dict[request.form['slct4']]
            Area = float(request.form['Area'])
            print('hello')

            # Prepare the input features as a 2D array
            input_data = [[Crop_Year, State_Name, District_Name, Season_Type, Crop_Name, Area]]

            # Assuming 'model' is a regressor for prediction
            predicted_production = model.predict(input_data)

            # Render the result on the index.html template
            print(predicted_production[0], predicted_production)
            return render_template('infor-yield.html', predicted_production=predicted_production[0])

        except Exception as e:
            error_message = "Error occurred: {}".format(str(e))
            return render_template('infor-yield.html', error_message=error_message)

    # Handle the case where the request method is not POST
    return render_template('infor-yield.html')

if __name__ == '__main__':
    app.run(debug=True)
