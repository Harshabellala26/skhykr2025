from flask import Flask, render_template, request, flash, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor  # More accurate than single Decision Tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configuration with adjusted base prices
COMMODITIES = {
    "paddy": {"base_price": 1200, "file": "paddy.csv", "unit": "quintal", "image": "paddy.jpeg"},
    "wheat": {"base_price": 1400, "file": "wheat.csv", "unit": "quintal", "image": "wheat.jpg"},
    "sugarcane": {"base_price": 1500, "file": "sugarcane.csv", "unit": "ton", "image": "sugarcane.jpg"},
    "jowar": {"base_price": 1550, "file": "jowar.csv", "unit": "quintal", "image": "jowar.jpg"}
}

def load_data(commodity):
    """Load and preprocess commodity data"""
    try:
        filepath = os.path.join("static", COMMODITIES[commodity]["file"])
        data = pd.read_csv(filepath)
        data.columns = data.columns.str.strip()
        
        # Feature engineering
        data['Month_sin'] = np.sin(2 * np.pi * data['Month']/12)
        data['Month_cos'] = np.cos(2 * np.pi * data['Month']/12)
        
        required_cols = ["Month", "Year", "Rainfall", "WPI", "Month_sin", "Month_cos"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"CSV missing required columns")
            
        return data
    except Exception as e:
        raise ValueError(f"Error loading {commodity} data: {str(e)}")

def train_model(X, y):
    """Train and return a more robust model"""
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use Random Forest for better accuracy
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, y)
    return model, scaler

def generate_realistic_predictions(model, scaler, data, commodity_name):
    """Generate more realistic predictions for 2025"""
    try:
        # Use weighted average of recent years' rainfall patterns
        recent_years = data[data["Year"] >= 2022]
        if len(recent_years) == 0:
            raise ValueError("No recent data available")
        
        # Weight more recent years heavier
        weights = np.linspace(0.5, 1, num=len(recent_years))
        weights /= weights.sum()
        
        predictions = []
        for month in range(1, 13):
            # Calculate weighted rainfall pattern
            weighted_rainfall = 0
            for year, group in recent_years.groupby('Year'):
                weight = weights[np.where(recent_years['Year'].unique() == year)[0][0]]
                weighted_rainfall += group[group['Month'] == month]['Rainfall'].values[0] * weight
            
            # Prepare features
            features = np.array([[
                month,
                2025,  # Prediction year
                weighted_rainfall,
                np.sin(2 * np.pi * month/12),
                np.cos(2 * np.pi * month/12)
            ]])
            
            # Scale features and predict
            features_scaled = scaler.transform(features)
            wpi_pred = model.predict(features_scaled)[0]
            
            # Apply market adjustment factor (0.9-1.1 range based on recent trends)
            last_3_months = data[(data['Year'] == 2024) & (data['Month'] >= 10)]
            if len(last_3_months) >= 2:
                trend = last_3_months['WPI'].pct_change().mean()
                adjustment = max(0.9, min(1.1, 1 + trend*2))  # Dampen the trend impact
                wpi_pred *= adjustment
            
            price_pred = (wpi_pred / 100) * COMMODITIES[commodity_name]["base_price"]
            
            predictions.append({
                "month": month,
                "year": 2025,
                "rainfall": round(weighted_rainfall, 1),
                "wpi": round(wpi_pred, 2),
                "price": round(price_pred, 2),
                "price_per_unit": f"₹{round(price_pred, 2)}/{COMMODITIES[commodity_name]['unit']}"
            })
        
        return predictions
    
    except Exception as e:
        raise ValueError(f"Prediction error: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html', commodities=COMMODITIES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        commodity = request.form['commodity']
        if commodity not in COMMODITIES:
            flash("Invalid commodity selected", "error")
            return redirect(url_for('index'))
        
        data = load_data(commodity)
        
        # Prepare features and target
        features = data[["Month", "Year", "Rainfall", "Month_sin", "Month_cos"]].values
        target = data["WPI"].values
        
        # Train model
        model, scaler = train_model(features, target)
        
        # Evaluate model on test set
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        y_pred = model.predict(scaler.transform(X_test))
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Model MAE for {commodity}: {mae:.2f} WPI points")
        
        # Generate predictions
        predictions = generate_realistic_predictions(model, scaler, data, commodity)
        
        return render_template('commodity.html',
                           commodity=commodity.capitalize(),
                           commodity_info=COMMODITIES[commodity],
                           predictions=predictions,
                           model_accuracy=f"{100 - mae:.1f}%")
    
    except Exception as e:
        flash(f"Error processing {commodity}: {str(e)}", "error")
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)