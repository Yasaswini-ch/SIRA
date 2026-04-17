import os
import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_CHARTS = True
except ImportError:
    HAS_CHARTS = False

def apply_chart_style():
    """Sets the premium dark dashboard theme for all matplotlib/seaborn plots."""
    if not HAS_CHARTS: return
    sns.set_style("darkgrid", {"axes.facecolor": "#1E293B", "grid.color": "#2D3748"})
    plt.rcParams.update({
        'figure.facecolor': '#1E293B',
        'axes.facecolor': '#1E293B',
        'axes.edgecolor': '#2D3748',
        'axes.labelcolor': 'white',
        'text.color': 'white',
        'xtick.color': '#94A3B8',
        'ytick.color': '#94A3B8',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Arial']
    })

def generate_charts(results_df, model=None, feature_names=None, charts_dir='static/charts'):
    """
    Creates and saves 6 distinct analysis charts to the static directory 
    using a professional dark-mode premium aesthetic.
    """
    if not HAS_CHARTS:
        print("Charts skipped due to missing optional plotting dependencies on Vercel.")
        return
        
    os.makedirs(charts_dir, exist_ok=True)
    apply_chart_style()
    accent = "#F97316"
    
    # 1. sales_distribution.png
    plt.figure(figsize=(10, 6))
    if 'Predicted_Demand' in results_df.columns:
        sns.histplot(results_df['Predicted_Demand'], bins=30, kde=True, color=accent, alpha=0.6)
        plt.title('Predicted Item Sales Distribution', fontsize=14, fontweight='bold', color='white')
        plt.xlabel('Predicted Sales', color='white')
        plt.ylabel('Frequency', color='white')
        sns.despine()
        plt.savefig(os.path.join(charts_dir, 'sales_distribution.png'), dpi=150, bbox_inches='tight', facecolor='#1E293B')
    plt.close()

    # 2. demand_by_category.png
    if 'Item_Type' in results_df.columns:
        plt.figure(figsize=(10, 6))
        avg_demand = results_df.groupby('Item_Type')['Predicted_Demand'].mean().sort_values(ascending=False).head(10)
        sns.barplot(x=avg_demand.values, y=avg_demand.index, hue=avg_demand.index, legend=False, palette='Oranges_r')
        plt.title('Top 10 High Demand Categories', fontsize=14, fontweight='bold', color='white')
        plt.xlabel('Average Predicted Demand', color='white')
        sns.despine()
        plt.savefig(os.path.join(charts_dir, 'demand_by_category.png'), dpi=150, bbox_inches='tight', facecolor='#1E293B')
        plt.close()

    # 3. alert_distribution.png
    if 'Alert_Level' in results_df.columns:
        plt.figure(figsize=(10, 6))
        alert_counts = results_df['Alert_Level'].value_counts()
        colors = {'CRITICAL': '#ef4444', 'WARNING': '#f97316', 'LOW': '#eab308', 'SAFE': '#10b981'}
        plot_colors = [colors.get(idx, '#3b82f6') for idx in alert_counts.index]
        plt.pie(alert_counts, labels=alert_counts.index, autopct='%1.1f%%', startangle=140, colors=plot_colors, 
                textprops={'color':"white", 'weight':'bold'}, wedgeprops={'width': 0.4, 'edgecolor': '#1E293B'})
        plt.title('Inventory Alert Distribution', fontsize=14, fontweight='bold', color='white', pad=20)
        plt.savefig(os.path.join(charts_dir, 'alert_distribution.png'), dpi=150, bbox_inches='tight', facecolor='#1E293B')
        plt.close()

    # 4. restock_heatmap.png
    if 'Outlet_Type' in results_df.columns and 'Item_Type' in results_df.columns:
        plt.figure(figsize=(10, 6))
        pivot = results_df.pivot_table(index='Outlet_Type', columns='Item_Type', values='Restock_Qty', aggfunc='mean').fillna(0)
        sns.heatmap(pivot, cmap='Oranges', annot=False, cbar_kws={'label': 'Avg Restock Qty'})
        plt.title('Restock Intensity Heatmap', fontsize=14, fontweight='bold', color='white')
        plt.savefig(os.path.join(charts_dir, 'restock_heatmap.png'), dpi=150, bbox_inches='tight', facecolor='#1E293B')
        plt.close()

    # 5. feature_importance.png
    if model is not None and feature_names is not None:
        plt.figure(figsize=(10, 6))
        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        
        if importances is not None:
            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feat_df = feat_df.sort_values(by='Importance', ascending=False).head(15)
            sns.barplot(x='Importance', y='Feature', data=feat_df, hue='Feature', legend=False, palette='Oranges_r')
            plt.title('Top 15 Demand Predictors (Model Analysis)', fontsize=14, fontweight='bold', color='white')
            plt.xlabel('Importance Weight', color='white')
            sns.despine()
            plt.savefig(os.path.join(charts_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight', facecolor='#1E293B')
            plt.close()

    # 6. mrp_vs_demand.png
    if 'Item_MRP' in results_df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=results_df, x='Item_MRP', y='Predicted_Demand', 
                        hue='Outlet_Type' if 'Outlet_Type' in results_df.columns else None, 
                        alpha=0.6, palette='Oranges')
        plt.title('MRP vs Predicted Demand Analysis', fontsize=14, fontweight='bold', color='white')
        plt.xlabel('Item MRP ($)', color='white')
        plt.ylabel('Predicted Sales', color='white')
        sns.despine()
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, facecolor='#1E293B', labelcolor='white')
        plt.savefig(os.path.join(charts_dir, 'mrp_vs_demand.png'), dpi=150, bbox_inches='tight', facecolor='#1E293B')
        plt.close()

def calculate_restock_quantity(predicted_demand, current_stock, lead_time_days=7, safety_factor=1.3):
    """
    Calculates operational restock metrics using expected daily demand, safety stock 
    buffers, and reorder points based on Economic Order Quantity (EOQ) principles.
    """
    # Daily demand estimate (assuming prediction is for a 30-day month)
    daily_demand = predicted_demand / 30 
    
    # Safety stock (covers demand during lead time + buffer)
    safety_stock = daily_demand * lead_time_days * safety_factor 
    
    # Reorder point
    reorder_point = daily_demand * lead_time_days + safety_stock 
    
    # Suggested order quantity (Economic Order Quantity simplified)
    order_qty = max(0, reorder_point - current_stock) 
    
    return { 
        'daily_demand': round(daily_demand, 2), 
        'safety_stock': round(safety_stock, 2),
        'reorder_point': round(reorder_point, 2),
        'suggested_order_qty': round(order_qty, 2) 
    }

def generate_alert(current_stock, reorder_point, predicted_demand):
    """
    Evaluates current stock against the reorder point to trigger operational alerts 
    for UI rendering.
    """
    ratio = current_stock / reorder_point if reorder_point > 0 else 1
    
    if ratio < 0.3: 
        return {
            'level': 'CRITICAL', 
            'color': 'red', 
            'message': 'Immediate restock required'
        } 
    elif ratio < 0.6:  
        return {
            'level': 'WARNING', 
            'color': 'orange', 
            'message': 'Restock within 3 days'
        } 
    elif ratio < 1.0: 
        return {
            'level': 'LOW', 
            'color': 'yellow', 
            'message': 'Schedule restock this week'
        } 
    else:
        return {
            'level': 'SAFE', 
            'color': 'green', 
            'message': 'Stock levels adequate'
        }

def format_prediction_response(item_id, outlet_id, predicted_demand, current_stock, restock_info, alert_info):
    """
    Packages the demand forecast, operational restock quantities, and alert bounds 
    into a clean JSON dict structure for the frontend API.
    """
    return {
        "inventory_context": {
            "item_id": item_id,
            "outlet_id": outlet_id,
            "current_stock": current_stock
        },
        "forecast": {
            "total_monthly_demand": round(predicted_demand, 2),
            "estimated_daily_demand": restock_info.get("daily_demand")
        },
        "restock_directive": {
            "suggested_order_qty": restock_info.get("suggested_order_qty"),
            "safety_stock_limit": restock_info.get("safety_stock"),
            "reorder_point": restock_info.get("reorder_point")
        },
        "alert": alert_info
    }

def calculate_waste_loss(products: list) -> dict:
    """
    products: list of dicts with keys:
      item_id, item_mrp, current_stock, predicted_demand,
      reorder_point, item_type, alert_level
      
    Returns a comprehensive loss report.
    """
    SPOILAGE_RATES = {
        "Dairy": 0.12,       # 12% monthly spoilage if overstocked
        "Breads": 0.18,
        "Fruits and Vegetables": 0.25,
        "Meat": 0.20,
        "Frozen Foods": 0.05,
        "Canned": 0.02,
        "Household": 0.01,
        "default": 0.05
    }
    PROFIT_MARGIN = 0.15     # 15% average retail margin
    
    total_overstock_loss = 0
    total_stockout_loss = 0
    item_breakdown = []
    
    for p in products:
        # Overstock loss
        overstock_units = max(0, p["current_stock"] - p["reorder_point"])
        rate = SPOILAGE_RATES.get(p["item_type"], SPOILAGE_RATES["default"])
        overstock_loss = overstock_units * p["item_mrp"] * rate
        
        # Stockout loss (missed sales)
        stockout_units = max(0, p["predicted_demand"] - p["current_stock"])
        stockout_loss = stockout_units * p["item_mrp"] * PROFIT_MARGIN
        
        item_breakdown.append({
            "item_id": p["item_id"],
            "item_type": p["item_type"],
            "overstock_loss": round(overstock_loss, 2),
            "stockout_loss": round(stockout_loss, 2),
            "total_loss": round(overstock_loss + stockout_loss, 2)
        })
        
        total_overstock_loss += overstock_loss
        total_stockout_loss += stockout_loss
        
    total_loss = total_overstock_loss + total_stockout_loss
    potential_savings = total_loss * 0.85
    
    return {
        "total_overstock_loss_inr": round(total_overstock_loss, 2),
        "total_stockout_loss_inr": round(total_stockout_loss, 2),
        "total_monthly_loss_inr": round(total_loss, 2),
        "potential_savings_inr": round(potential_savings, 2),
        "worst_offenders": sorted(item_breakdown, key=lambda x: x["total_loss"], reverse=True)[:5],
        "item_breakdown": item_breakdown
    }
