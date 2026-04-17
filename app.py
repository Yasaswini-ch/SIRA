import os
import json
import joblib
import datetime
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from utils import calculate_restock_quantity, generate_alert, format_prediction_response

# ----- APP SETUP -----
app = Flask(__name__)

# Constants
MODEL_PATH = os.path.join("models", "demand_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
ENCODERS_PATH = os.path.join("models", "encoders.pkl")
CHARTS_DIR = os.path.join("static", "charts")

# Globals mapping
model = None
scaler = None
encoders = None

def load_ml_assets():
    global model, scaler, encoders
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODERS_PATH):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            encoders = joblib.load(ENCODERS_PATH)
            print(f"[{datetime.datetime.now()}] [INFO] Machine Learning assets successfully booted.")
        else:
            print(f"[{datetime.datetime.now()}] [WARNING] Could not locate ML models in models/. Please run train.py first.")
    except Exception as e:
        print(f"[{datetime.datetime.now()}] [ERROR] Fault loading ML assets: {e}")

# Call immediately at startup (not per-request)
load_ml_assets()

# ----- MIDDLEWARE -----
@app.after_request
def add_cors_headers(response):
    """Adds CORS headers for local GUI development."""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# ----- ERROR HANDLERS -----
@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad Request", "message": str(e)}), 400

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred during processing."}), 500

# ----- CORE URL ROUTES -----
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/charts/<path:filename>')
def serve_chart(filename):
    """Serves local generated charts dynamically."""
    return send_from_directory(CHARTS_DIR, filename)

@app.route('/health')
def health():
    return jsonify({
        "status": "ok", 
        "model": "loaded" if model else "uninitialized"
    }), 200

# ----- HELPER PIPELINE -----
def _get_sensible_default(field_name):
    """Resolves sensible default substitutions for broken request data."""
    defaults = {
        'Item_Weight': 12.0,
        'Item_Fat_Content': 'Regular',
        'Item_Visibility': 0.05,
        'Item_Type': 'Snack Foods',
        'Item_MRP': 140.0,
        'Outlet_Establishment_Year': 1999,
        'Outlet_Size': 'Medium',
        'Outlet_Location_Type': 'Tier 2',
        'Outlet_Type': 'Supermarket Type1',
        'current_stock': 20
    }
    return defaults.get(field_name, 0)

# ----- API ENDPOINTS -----
@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Predictive model currently offline. Run training pipeline."}), 503
        
    data = request.json
    if not data:
        return jsonify({"error": "Invalid or missing JSON payload."}), 400
        
    print(f"[{datetime.datetime.now()}] [PREDICT REQUEST] Received: {data}")
    
    warnings = []
    row = {}
    
    # 1. Fill MISSING fields intelligently
    required_fields = ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 
                      'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size', 
                      'Outlet_Location_Type', 'Outlet_Type', 'current_stock']
                      
    for field in required_fields:
        if field not in data or data[field] is None or str(data[field]).strip() == "":
            val = _get_sensible_default(field)
            row[field] = val
            warnings.append(f"Missing required parameter '{field}', substituted intelligently with default: {val}")
        else:
            row[field] = data[field]
            
    try:
        # Import and route to prediction orchestrator explicitly
        from predict import run_prediction_single
        prediction_val = float(run_prediction_single(row, model, scaler, encoders))
    except Exception as e:
        print(f"[{datetime.datetime.now()}] [ERROR] Prediction inference failed: {e}")
        # Standard fallback if predict pipeline has issues locally
        prediction_val = 1500.0
        warnings.append("Prediction Engine offline/errored. Utilizing fallback metrics.")

    current_stock = int(row.get('current_stock', 0))
    
    # 2. Leverage the Smart Decision Logic Engine
    restock_info = calculate_restock_quantity(prediction_val, current_stock)
    alert_info = generate_alert(current_stock, restock_info['reorder_point'], prediction_val)
    
    # 3. Securely format API response JSON
    response = format_prediction_response(
        item_id=data.get('Item_Identifier', 'UI-DEMO-001'),
        outlet_id=data.get('Outlet_Identifier', 'OUT001'),
        predicted_demand=prediction_val,
        current_stock=current_stock,
        restock_info=restock_info,
        alert_info=alert_info
    )
    
    if warnings:
        response['diagnostic_warnings'] = warnings

    return jsonify(response)

@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    if not model:
        return jsonify({"error": "Batch engine offline. Model missing."}), 503
        
    if 'file' not in request.files:
        return jsonify({"error": "No CSV file block detected in multipart form."}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400
        
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Unsupported Media Type. Only .csv is permitted."}), 415
        
    try:
        print(f"[{datetime.datetime.now()}] [BATCH UPLOAD] Processing: {file.filename}")
        df = pd.read_csv(file)
        
        # Dispatch to robust batch orchestrator
        from predict import predict_batch, generate_batch_report
        from utils import generate_charts
        results_list, warnings = predict_batch(df, model, scaler, encoders)
        
        # Freshly regenerate dashboard charts based on the new batch data
        batch_df = pd.DataFrame(results_list)
        # Merge back categorical columns from original df for better plotting context
        for col in ['Item_Type', 'Outlet_Type', 'Item_MRP']:
            if col in df.columns:
                batch_df[col] = df[col]
                
        generate_charts(batch_df, model=model, feature_names=model.feature_names_in_)
        
        batch_report = generate_batch_report(results_list)
        
        return jsonify({
            "status": "success",
            "filename_processed": file.filename,
            "diagnostic_warnings": warnings,
            "report_metrics": batch_report,
            "preview_data": results_list[:25]
        })
    except Exception as e:
        print(f"[{datetime.datetime.now()}] [ERROR] Batch pipeline failed: {e}")
        return jsonify({"error": "Failed to process CSV batch.", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
