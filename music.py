import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import joblib
import plotly.express as px

# Load data
df = pd.read_csv('spotify_track_data.csv')

# Preprocess data
df['genres'] = df['genres'].apply(lambda x: ', '.join(eval(x)))
df = df.rename(columns={'duration (ms)': 'duration_ms'})

# Feature selection
features = ['valence', 'energy', 'danceability', 'tempo', 'duration_ms', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness']
X = df[features]
y = df['mood']

# Standardize and normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Handle imbalance data by applying SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Initialize model
model = XGBClassifier(eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print('XGBoost Baseline Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Hyperparameter tuning
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}

# Perform GridSearch
grid_search_xgb = GridSearchCV(estimator=XGBClassifier(eval_metric='mlogloss', random_state=42),
                               param_grid=param_grid_xgb, cv=3, n_jobs=-1, verbose=2)

# Fit grid search
grid_search_xgb.fit(X_train, y_train)

# Get the best estimator (for the model)
best_model_estimator = grid_search_xgb.best_estimator_

# Evaluate the best estimator for the model
y_pred = best_model_estimator.predict(X_test)

print('XGBoost Accuracy after Grid Search:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save the best model and scaler
joblib.dump(best_model_estimator, 'best_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Define web-app
# Load the saved scaler and best model
scaler = joblib.load('scaler.joblib')
best_model = joblib.load('best_model.joblib')

# Function to encode mood
def encode_mood(mood_str):
    return le.transform([mood_str])[0]

# Function to decode mood
def decode_mood(mood_encoded):
    return le.inverse_transform([mood_encoded])[0]

# Initialize the dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1('Mood-Based Music Recommendation System'),
    dcc.Dropdown(
        id='mood-dropdown',
        options=[{'label': label, 'value': encode_mood(label)} for label in le.classes_],
        placeholder='Select a mood',
    ),
    html.Div(id='recommendations')
])

# Define the callback
@app.callback(
    Output('recommendations', 'children'),
    [Input('mood-dropdown', 'value')]
)
def update_recommendations(mood_encoded):
    print(f"Selected encoded mood: {mood_encoded}")
    if mood_encoded is None:
        return 'Select a mood to get recommendations.'
    
    decoded_mood = decode_mood(mood_encoded)
    print(f"Decoded mood: {decoded_mood}")
    print(f"Mood values in df: {df['mood'].unique()}")

    # Filter the data based on the selected mood
    mood_tracks = df[df['mood'] == decoded_mood]

    print(f"Filtered tracks: {len(mood_tracks)}")
    
    if mood_tracks.empty:
        return 'No tracks available for the selected mood.'
    
    recommended_tracks = mood_tracks.sample(min(5, len(mood_tracks)))

    # Display the recommended tracks
    return html.Ul([html.Li(f"{row['name']} by {row['artist']}") for _, row in recommended_tracks.iterrows()])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)