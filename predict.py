import os
import pandas as pd
import numpy as np
from utils import calculate_restock_quantity, generate_alert

# Sensible defaults matching app.py fallback logic
DEFAULTS = {
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

def validate_and_fill_batch(df):
    """Gracefully interpolates missing columns during CSV inferencing."""
    warnings = []
    required_cols = list(DEFAULTS.keys())
    
    for col in required_cols:
        if col not in df.columns:
            warnings.append(f"Missing mandatory column '{col}'. Hydrated universally with default value: {DEFAULTS[col]}.")
            df[col] = DEFAULTS[col]
        else:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                warnings.append(f"Auto-filled {missing_count} null row(s) inside column '{col}' with median/mode safe baseline.")
                df[col] = df[col].fillna(DEFAULTS[col])
                
    return df, warnings

def preprocess_inference_data(df, scaler, encoders):
    """Clones the precise data modeling pipeline transformations dynamically for batch/single inference."""
    pdf = df.copy()
    
    # 1. Standardize text strings mapping
    if 'Item_Fat_Content' in pdf.columns:
        pdf['Item_Fat_Content'] = pdf['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})
        le = encoders['Item_Fat_Content']
        known_classes = set(le.classes_)
        # Out-of-bounds categorical inference safe catching
        pdf['Item_Fat_Content'] = pdf['Item_Fat_Content'].apply(lambda x: x if x in known_classes else le.classes_[0])
        pdf['Item_Fat_Content'] = le.transform(pdf['Item_Fat_Content'])
        
    # 2. Re-trigger Feature Engineering logic exactly
    if 'Outlet_Establishment_Year' in pdf.columns:
        pdf['Outlet_Age'] = 2013 - pdf['Outlet_Establishment_Year']
    
    if 'Item_Visibility' in pdf.columns:
        pdf['Item_Visibility_MeanRatio'] = pdf['Item_Visibility'] / 0.05
        
    if 'Item_MRP' in pdf.columns and 'Item_Weight' in pdf.columns:
        pdf['Price_Per_Unit_Weight'] = pdf['Item_MRP'] / pdf['Item_Weight']
        
    if 'Item_MRP' in pdf.columns:
        # Re-apply binning rules. Static boundaries are safer for unseen inferential distribution shifts
        bins = [0, 90, 140, 190, 2000]
        labels = ['Budget', 'Mid', 'Premium', 'Luxury']
        pdf['MRP_Tier'] = pd.cut(pdf['Item_MRP'], bins=bins, labels=labels, include_lowest=True)

    # 3. Synchronize One-Hot encodings matrix dynamically
    cat_cols = ['Item_Type', 'Outlet_Type', 'Outlet_Location_Type', 'Outlet_Size', 'MRP_Tier']
    
    ohe = encoders['ohe']
    encoded_cats = ohe.transform(pdf[cat_cols])
    encoded_cat_df = pd.DataFrame(encoded_cats, columns=ohe.get_feature_names_out(cat_cols))
    
    pdf = pdf.drop(columns=cat_cols)
    pdf = pd.concat([pdf.reset_index(drop=True), encoded_cat_df.reset_index(drop=True)], axis=1)

    for col in ['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year', 'current_stock']:
        if col in pdf.columns:
            pdf = pdf.drop(columns=[col])
            
    # 4. Standard Scaler Transform mapping
    num_cols_to_scale = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age']
    
    for col in num_cols_to_scale:
        if col not in pdf.columns:
            pdf[col] = 0.0 
            
    pdf[num_cols_to_scale] = scaler.transform(pdf[num_cols_to_scale])
    
    return pdf

def run_prediction_single(row_dict, model, scaler, encoders):
    """Singular JSON wrapper execution."""
    df = pd.DataFrame([row_dict])
    df, _ = validate_and_fill_batch(df)
    processed_df = preprocess_inference_data(df, scaler, encoders)
    
    # Filter explicitly to signature
    model_cols = model.feature_names_in_
    for c in model_cols:
        if c not in processed_df.columns:
            processed_df[c] = 0
            
    processed_df = processed_df[model_cols]
    return model.predict(processed_df)[0]

def predict_batch(df, model, scaler, encoders):
    """
    Processes an entire raw DataFrame batch payload, returning structured JSON mapping dict lists.
    """
    raw_df = df.copy()
    raw_df, warnings = validate_and_fill_batch(raw_df)
    
    processed_df = preprocess_inference_data(raw_df, scaler, encoders)
    
    model_cols = model.feature_names_in_
    for c in model_cols:
        if c not in processed_df.columns:
            processed_df[c] = 0
    processed_df = processed_df[model_cols]
    
    # Model Infers entire multidimensional matrix rapidly
    preds = model.predict(processed_df)
    
    results_list = []
    
    # Compile intelligent results list output dicts
    for idx, row in raw_df.iterrows():
        prediction_val = float(preds[idx])
        curr_stock = int(row.get('current_stock', 20))
        
        restock_info = calculate_restock_quantity(prediction_val, curr_stock)
        alert_info = generate_alert(curr_stock, restock_info['reorder_point'], prediction_val)
        
        hist = generate_synthetic_history(prediction_val, str(row.get('Item_Type', 'Item')))
        trend_info = detect_trend(hist)

        results_list.append({
            'Item_Identifier': str(row.get('Item_Identifier', f'ITM-{idx}')),
            'Outlet_Identifier': str(row.get('Outlet_Identifier', 'OUT-UNKNOWN')),
            'Current_Stock': curr_stock,
            'Predicted_Demand': round(prediction_val, 2),
            'Restock_Qty': restock_info['suggested_order_qty'],
            'Item_MRP': float(row.get('Item_MRP', 140)),
            'Item_Type': str(row.get('Item_Type', 'Item')),
            'Alert_Level': alert_info['level'],
            'Alert_Message': alert_info['message'],
            'Alert_Color': alert_info['color'],
            'Trend': trend_info
        })
        
    return results_list, warnings

def generate_batch_report(results_list):
    """Analyzes a successful list response to yield a dashboard-ready reporting metrics package with financial loss simulations."""
    if not results_list:
        return {}
        
    df = pd.DataFrame(results_list)
    
    # Calculate financial impact (simulated)
    # Assume 15% holding cost for overstock and 30% margin loss for stockouts
    df['overstock_loss'] = df.apply(lambda r: max(0, r['Current_Stock'] - r['Predicted_Demand']) * r['Item_MRP'] * 0.15, axis=1)
    df['stockout_loss'] = df.apply(lambda r: max(0, r['Predicted_Demand'] - r['Current_Stock']) * r['Item_MRP'] * 0.30, axis=1)
    df['total_item_loss'] = df['overstock_loss'] + df['stockout_loss']
    
    overstock_total = float(df['overstock_loss'].sum())
    stockout_total = float(df['stockout_loss'].sum())
    
    top_loss_items = df.sort_values(by='total_item_loss', ascending=False).head(10)
    loss_items_list = []
    for _, r in top_loss_items.iterrows():
        loss_items_list.append({'id': r['Item_Identifier'], 'loss': r['total_item_loss']})

    report = {
        'total_items_analyzed': len(df),
        'critical_alerts_count': int((df['Alert_Level'] == 'CRITICAL').sum()),
        'warning_alerts_count': int((df['Alert_Level'] == 'WARNING').sum()),
        'average_demand_forecast': round(df['Predicted_Demand'].mean(), 2),
        'top_5_immediate_restocks': df.sort_values(by='Restock_Qty', ascending=False).head(5).to_dict(orient='records'),
        
        # Financial Block for PDF
        'total_loss': overstock_total + stockout_total,
        'overstock': overstock_total,
        'stockout': stockout_total,
        'savings': (overstock_total + stockout_total) * 0.45, # Est. optimization impact
        'top_items': loss_items_list,
        
        # Trend Summary
        'trend_summary': {
            'up': sum(1 for r in results_list if r.get('Trend', {}).get('direction') == 'up'),
            'down': sum(1 for r in results_list if r.get('Trend', {}).get('direction') == 'down'),
            'spike': sum(1 for r in results_list if r.get('Trend', {}).get('seasonal_flag'))
        }
    }
    
    return report

def detect_trend(sales_history: list, window: int = 4) -> dict:
    """
    sales_history: list of monthly sales values (from BigMart or synthetic)
    window: rolling window size
    Returns trend direction, strength, and seasonal flag.
    """
    import numpy as np
    
    if len(sales_history) < window * 2:
        return {"direction": "stable", "strength": 0, "seasonal_flag": False}
        
    # Simple linear trend using polyfit
    x = np.arange(len(sales_history))
    slope, intercept = np.polyfit(x, sales_history, 1)
        
    # Normalize slope as % change per period
    mean_sales = np.mean(sales_history)
    trend_pct = (slope / mean_sales) * 100 if mean_sales > 0 else 0
        
    # Rolling volatility to detect seasonal spikes
    rolling_std = np.std(sales_history[-window:])
    baseline_std = np.std(sales_history[:-window])
    volatility_ratio = rolling_std / baseline_std if baseline_std > 0 else 1
        
    direction = "up" if trend_pct > 3 else ("down" if trend_pct < -3 else "stable")
    seasonal_flag = volatility_ratio > 1.5  # Recent volatility is 1.5x baseline
        
    return {
        "direction": str(direction),
        "trend_pct": float(round(trend_pct, 1)),
        "strength": float(min(abs(trend_pct) / 10, 1.0)),  # 0-1 scale
        "seasonal_flag": bool(seasonal_flag),
        "volatility_ratio": float(round(volatility_ratio, 2)),
        "sparkline_data": sales_history[-8:]  # Last 8 periods for chart
    }

def generate_synthetic_history(base_demand, item_type):
    """
    Creates 12 months of plausible sales history for items that don't have one.
    Adds seasonal multipliers for Indian retail context:
    October-November: 1.4x (Diwali)
    January: 1.2x (New Year)
    March-April: 1.15x (Holi/summer start)
    """
    import numpy as np
    
    multipliers = [1.2, 1.0, 1.15, 1.15, 1.0, 1.0, 1.0, 1.0, 1.0, 1.4, 1.4, 1.0]
    
    history = []
    for m in multipliers:
        # Add slight random noise to make it look realistic (normal distribution, 5% std dev)
        noise = np.random.normal(1.0, 0.05)
        monthly_sales = base_demand * m * noise
        history.append(max(0, int(monthly_sales)))
        
    return history