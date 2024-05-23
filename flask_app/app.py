from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import sklearn
from sklearn.linear_model import LogisticRegression
import pickle

app = Flask(__name__)
CORS(app)

print('The scikit-learn version is {}.'.format(sklearn.__version__))

# Load the trained model
randomforest_model = LogisticRegression()
model_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swiss_super_league.pkl")
with open(model_filename, 'rb') as f:
    randomforest_model = pickle.load(f)

matches_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preprocessed_matches.csv'),
                            sep=',', encoding='utf-8')
home_teams = matches_df['HTeam'].unique()
away_teams = matches_df['ATeam'].unique()
all_teams = sorted(set(home_teams) | set(away_teams))

# Define the relevant features (make sure these match your training data)
features = [
   'home_possession', 'away_possession', 'home_shots_on_target', 'away_shots_on_target',
       'home_saves', 'away_saves', 'avg_betting_odds_home',
       'avg_betting_odds_draw', 'avg_betting_odds_away'
]

@app.route('/')
def index():
    return "<p>Hello World!</p>"

@app.route('/api/teams', methods=['GET'])
def get_teams():
    return jsonify(all_teams)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    home_team = data['home_team']
    away_team = data['away_team']
    
    # Gather the last 5 games' statistics for the selected teams
    home_team_last5 = matches_df[matches_df['HTeam'] == home_team].tail(5)
    away_team_last5 = matches_df[matches_df['ATeam'] == away_team].tail(5)
    
    # Calculate the mean of the last 5 games' statistics
    home_stats = home_team_last5[[
        'home_possession', 'home_shots_on_target', 'avg_betting_odds_home', 'home_saves'
    ]].mean()
    
    away_stats = away_team_last5[[
        'away_possession', 'away_shots_on_target', 'away_saves', 'avg_betting_odds_away', 'avg_betting_odds_draw'
    ]].mean()
    
    # Combine the statistics into a single input for prediction
    input_features = pd.DataFrame([[
        home_stats['home_possession'], away_stats['away_possession'],
        home_stats['home_shots_on_target'], away_stats['away_shots_on_target'],
        home_stats['home_saves'], away_stats['away_saves'],
        home_stats['avg_betting_odds_home'], away_stats['avg_betting_odds_draw'], away_stats['avg_betting_odds_away']
    ]], columns=features)

    # Check for feature mismatch
    missing_features = set(features) - set(input_features.columns)
    if missing_features:
        return jsonify({'error': f'Missing features: {missing_features}'}), 400

    # Make prediction
    prediction = randomforest_model.predict(input_features)[0]
    
    # Map prediction to result
    result_mapping = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    result = result_mapping.get(int(prediction), "Unknown result")
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
