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
if os.environ.get("VERCEL"):
    CHARTS_DIR = os.path.join("/tmp", "charts")
else:
    CHARTS_DIR = os.path.join("static", "charts")

# Globals mapping
model = None
scaler = None
encoders = None
LAST_BATCH_RESULTS = []
LAST_BATCH_REPORT = {}

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
    metadata = {}
    metadata_path = os.path.join("models", "model_metadata.json")
    try:
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        
    return render_template('index.html', model_metadata=metadata)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predictor')
def predictor():
    return render_template('predictor.html')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/simulator')
def simulator():
    return render_template('simulator.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/charts/<path:filename>')
def serve_chart(filename):
    """Serves local generated charts dynamically."""
    return send_from_directory(CHARTS_DIR, filename)

@app.route('/download-report')
def download_report():
    """Generates and streams the professional PDF inventory report."""
    from utils import generate_pdf_report
    from flask import make_response
    
    # Use global state from last batch if exists, else provide empty indicators
    results = LAST_BATCH_RESULTS or []
    report = LAST_BATCH_REPORT or {}
    
    pdf_buffer = generate_pdf_report(results, report)
    
    filename = f"restock_report_{datetime.datetime.now().strftime('%Y-%m-%d')}.pdf"
    
    response = make_response(pdf_buffer.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename={filename}'
    return response

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

    # Attach Trend + sparkline for the Predictor Hub charts
    try:
        from predict import generate_synthetic_history, detect_trend
        history = generate_synthetic_history(prediction_val, row.get('Item_Type', 'Snack Foods'))
        trend = detect_trend(history)
        response['Trend'] = trend
    except Exception as te:
        print(f"[WARN] Trend generation failed: {te}")
        response['Trend'] = {
            "direction": "stable",
            "sparkline_data": [int(prediction_val)] * 8,
            "seasonal_flag": False
        }

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
        
        # NOTE: Server-side chart generation (matplotlib) is disabled to stay within Vercel's 500MB limit.
        # Visual analytics are now handled client-side via Chart.js in the dashboard.
        # generate_charts(batch_df, model=model, feature_names=model.feature_names_in_, charts_dir=CHARTS_DIR)
        
        batch_report = generate_batch_report(results_list)
        
        # Persist for PDF download
        global LAST_BATCH_RESULTS, LAST_BATCH_REPORT
        LAST_BATCH_RESULTS = results_list
        LAST_BATCH_REPORT = batch_report
        
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

@app.route('/api/loss-report', methods=['GET'])
def get_loss_report():
    if not model:
        return jsonify({"error": "Engine offline. Model missing."}), 503
        
    try:
        # Generate a simulated loss report based on sample batch data
        df = pd.read_csv("sample_items.csv")
        from predict import predict_batch
        from utils import calculate_restock_quantity, calculate_waste_loss
        
        results_list, _ = predict_batch(df, model, scaler, encoders)
        
        products = []
        for idx, row in df.iterrows():
            res = results_list[idx]
            restock_info = calculate_restock_quantity(res["Predicted_Demand"], res["Current_Stock"])
            
            products.append({
                "item_id": res["Item_Identifier"],
                "item_mrp": float(row.get("Item_MRP", 0)),
                "current_stock": res["Current_Stock"],
                "predicted_demand": res["Predicted_Demand"],
                "reorder_point": restock_info["reorder_point"],
                "item_type": str(row.get("Item_Type", "default")),
                "alert_level": res["Alert_Level"]
            })
            
        report = calculate_waste_loss(products)
        return jsonify(report)
    except Exception as e:
        print(f"[{datetime.datetime.now()}] [ERROR] Loss report failed: {e}")
        return jsonify({"error": "Failed to generate loss report.", "details": str(e)}), 500

@app.route('/api/dashboard-stats', methods=['GET'])
def get_dashboard_stats():
    """Provides real-time aggregated metrics and model insights for the dashboard charts."""
    if not model:
        return jsonify({"error": "Engine offline"}), 503
        
    try:
        # Load model metadata for feature importance
        metadata_path = os.path.join("models", "model_metadata.json")
        features = []
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
                features = meta.get('top_features', [])

        # Process sample data for category demand
        df = pd.read_csv("sample_items.csv")
        from predict import predict_batch
        res, _ = predict_batch(df, model, scaler, encoders)
        res_df = pd.DataFrame(res)
        
        cat_demand = res_df.groupby('Item_Type')['Predicted_Demand'].mean().round(2).to_dict()
        alert_dist = res_df['Alert_Level'].value_counts().to_dict()
        
        # Prepare for distribution chart
        bins = [0, 500, 1000, 1500, 2000, 2500, 5000]
        labels = ['0-500', '500-1k', '1k-1.5k', '1.5k-2k', '2k-2.5k', '2.5k+']
        sales_dist = pd.cut(res_df['Predicted_Demand'], bins=bins, labels=labels).value_counts().sort_index().to_dict()

        return jsonify({
            "features": features,
            "category_demand": cat_demand,
            "alert_distribution": alert_dist,
            "sales_distribution": sales_dist
        })
    except Exception as e:
        print(f"Error generating dashboard stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/heatmap-data', methods=['GET'])
def get_heatmap_data():
    """Returns pivot matrices for the restock intensity and loss heatmaps on the dashboard."""
    if not model:
        return jsonify({"error": "Engine offline"}), 503

    try:
        df = pd.read_csv("sample_items.csv")
        from predict import predict_batch
        from utils import calculate_restock_quantity, calculate_waste_loss

        results_list, _ = predict_batch(df, model, scaler, encoders)
        res_df = pd.DataFrame(results_list)

        # Merge original columns needed for pivoting
        res_df['Outlet_Type'] = df['Outlet_Type'].values
        res_df['Item_Type']   = df['Item_Type'].values
        res_df['Item_MRP']    = df['Item_MRP'].values

        # 1. Restock Intensity Pivot (Outlet_Type × Item_Type → avg Restock_Qty)
        restock_pivot = (
            res_df.groupby(['Outlet_Type', 'Item_Type'])['Restock_Qty']
            .mean()
            .fillna(0)
            .reset_index()
        )

        # 2. Loss Intensity Pivot (Outlet_Type × Item_Type → total loss ₹)
        products = []
        for _, row in res_df.iterrows():
            ri = calculate_restock_quantity(row['Predicted_Demand'], row['Current_Stock'])
            products.append({
                'item_id':          row['Item_Identifier'],
                'item_mrp':         float(row['Item_MRP']),
                'current_stock':    row['Current_Stock'],
                'predicted_demand': row['Predicted_Demand'],
                'reorder_point':    ri['reorder_point'],
                'item_type':        row['Item_Type'],
                'alert_level':      row['Alert_Level'],
                'outlet_type':      row['Outlet_Type'],
            })

        # Compute per-item losses and re-attach outlet/category info
        SPOILAGE_RATES = {
            "Dairy": 0.12, "Breads": 0.18, "Fruits and Vegetables": 0.25,
            "Meat": 0.20, "Frozen Foods": 0.05, "Canned": 0.02,
            "Household": 0.01, "default": 0.05
        }
        PROFIT_MARGIN = 0.15
        loss_rows = []
        for p in products:
            overstock_units = max(0, p['current_stock'] - p['reorder_point'])
            rate = SPOILAGE_RATES.get(p['item_type'], SPOILAGE_RATES['default'])
            overstock_loss = overstock_units * p['item_mrp'] * rate
            stockout_units = max(0, p['predicted_demand'] - p['current_stock'])
            stockout_loss  = stockout_units * p['item_mrp'] * PROFIT_MARGIN
            loss_rows.append({
                'Outlet_Type': p['outlet_type'],
                'Item_Type':   p['item_type'],
                'total_loss':  round(overstock_loss + stockout_loss, 2)
            })
        loss_df = pd.DataFrame(loss_rows)
        loss_pivot = (
            loss_df.groupby(['Outlet_Type', 'Item_Type'])['total_loss']
            .sum()
            .fillna(0)
            .reset_index()
        )

        return jsonify({
            "restock": restock_pivot.to_dict(orient='records'),
            "loss":    loss_pivot.to_dict(orient='records'),
        })
    except Exception as e:
        print(f"[ERROR] Heatmap data generation failed: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
