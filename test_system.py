import os
import joblib
import pandas as pd
import numpy as np
import requests
from utils import calculate_restock_quantity, generate_alert

# Configuration
BASE_URL = "http://localhost:5000"
MODEL_PATH = "models/demand_model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODERS_PATH = "models/encoders.pkl"

def test_local_logic():
    print("\n" + "="*80)
    print("TEST 1: LOCAL DECISION LOGIC & ARTIFACT LOADING")
    print("="*80)
    
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        print("[PASS] Artifacts loaded successfully.")
    except Exception as e:
        print(f"[FAIL] Artifact loading failed: {e}")
        return

    # 5 Manual Test Items
    test_items = [
        {
            "name": "Standard Snack (Mid-Stock)",
            "data": {
                'Item_Weight': 12.0, 'Item_Fat_Content': 'Regular', 'Item_Visibility': 0.05,
                'Item_Type': 'Snack Foods', 'Item_MRP': 140.0, 'Outlet_Establishment_Year': 1999,
                'Outlet_Size': 'Medium', 'Outlet_Location_Type': 'Tier 2', 'Outlet_Type': 'Supermarket Type1',
                'current_stock': 20
            }
        },
        {
            "name": "Luxury Dairy (Zero Stock - Edge Case)",
            "data": {
                'Item_Weight': 15.0, 'Item_Fat_Content': 'Low Fat', 'Item_Visibility': 0.01,
                'Item_Type': 'Dairy', 'Item_MRP': 250.0, 'Outlet_Establishment_Year': 2005,
                'Outlet_Size': 'High', 'Outlet_Location_Type': 'Tier 1', 'Outlet_Type': 'Supermarket Type3',
                'current_stock': 0
            }
        },
        {
            "name": "Budget Household (High Stock - Edge Case)",
            "data": {
                'Item_Weight': 10.0, 'Item_Fat_Content': 'Regular', 'Item_Visibility': 0.08,
                'Item_Type': 'Household', 'Item_MRP': 50.0, 'Outlet_Establishment_Year': 1985,
                'Outlet_Size': 'Small', 'Outlet_Location_Type': 'Tier 3', 'Outlet_Type': 'Grocery Store',
                'current_stock': 500
            }
        },
        {
            "name": "Unknown Category (Graceful Handle)",
            "data": {
                'Item_Type': 'Digital Software', # Unknown to training set
                'Item_MRP': 100.0, 'current_stock': 15
            }
        },
        {
            "name": "Premium Frozen (Low Stock)",
            "data": {
                'Item_Weight': 8.0, 'Item_Fat_Content': 'Regular', 'Item_Visibility': 0.12,
                'Item_Type': 'Frozen Foods', 'Item_MRP': 190.0, 'Outlet_Establishment_Year': 2009,
                'Outlet_Size': 'Medium', 'Outlet_Location_Type': 'Tier 2', 'Outlet_Type': 'Supermarket Type2',
                'current_stock': 5
            }
        }
    ]

    from predict import run_prediction_single
    
    results = []
    print(f"{'Test Item':<35} | {'Demand':<8} | {'Order':<8} | {'Alert'}")
    print("-" * 75)
    
    for item in test_items:
        try:
            pred_demand = run_prediction_single(item['data'], model, scaler, encoders)
            curr_stock = item['data'].get('current_stock', 20)
            
            restock = calculate_restock_quantity(pred_demand, curr_stock)
            alert = generate_alert(curr_stock, restock['reorder_point'], pred_demand)
            
            # Verifications
            assert pred_demand > 0, "Predicted demand should be positive"
            assert restock['suggested_order_qty'] >= 0, "Restock quantity cannot be negative"
            assert alert['level'] in ['CRITICAL', 'WARNING', 'LOW', 'SAFE'], "Invalid alert level"
            
            print(f"{item['name']:<35} | {pred_demand:<8.2f} | {restock['suggested_order_qty']:<8.2f} | {alert['level']}")
            
            # Edge Case Specific Verifications
            if "Zero Stock" in item['name']:
                assert alert['level'] == 'CRITICAL', "Zero stock should trigger CRITICAL"
            if "High Stock" in item['name']:
                assert alert['level'] == 'SAFE', "High stock should be SAFE"

        except Exception as e:
            print(f"{item['name']:<35} | [ERROR] {e}")

def test_flask_api():
    print("\n" + "="*80)
    print("TEST 2: FLASK API END-TO-END (/predict)")
    print("="*80)
    
    test_payload = {
        "Item_Type": "Snack Foods",
        "Item_MRP": 150.0,
        "Outlet_Type": "Supermarket Type1",
        "current_stock": 5
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=test_payload, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("[PASS] API returned 200 OK.")
            print(f"Response Summary: Demand={data['forecast']['total_monthly_demand']}, Qty={data['restock_directive']['suggested_order_qty']}, Level={data['alert']['level']}")
            
            # Check structure
            keys = ['inventory_context', 'forecast', 'restock_directive', 'alert']
            if all(k in data for k in keys):
                print("[PASS] Response JSON structure is valid.")
            else:
                print("[FAIL] Missing keys in JSON response.")
        else:
            print(f"[FAIL] API returned status {response.status_code}: {response.text}")
    except requests.exceptions.ConnectionError:
        print("[SKIPPED] Flask server not running at http://localhost:5000. Start app.py to test API.")
    except Exception as e:
        print(f"[FAIL] API test encountered error: {e}")

if __name__ == "__main__":
    test_local_logic()
    test_flask_api()
