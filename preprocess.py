import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Use the 'Agg' backend to avoid require display during plot generation
matplotlib.use('Agg')

def perform_eda(train_path='data/Train.csv', test_path='data/Test.csv', charts_dir='static/charts'):
    """
    Performs Exploratory Data Analysis (EDA) on the BigMart Sales Dataset.
    """
    print("-" * 50)
    print("1. LOADING DATASETS")
    print("-" * 50)
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Please place Dataset into data/")
        return None, None
        
    train_df = pd.read_csv(train_path)
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        print("Successfully loaded Train and Test datasets.\n")
    else:
        test_df = pd.DataFrame()
        print(f"Warning: {test_path} missing. Proceeding with Train dataset only.\n")

    print("-" * 50)
    print("2. BASIC STATISTICS (Train)")
    print("-" * 50)
    print(f"Shape: {train_df.shape}\n")
    
    print("Data Types:")
    print(train_df.dtypes, "\n")
    
    print("Null Counts:")
    print(train_df.isnull().sum(), "\n")
    
    print("Basic Stats (Numeric):")
    print(train_df.describe(), "\n")

    print("-" * 50)
    print("3. GENERATING VISUALIZATIONS")
    print("-" * 50)
    
    os.makedirs(charts_dir, exist_ok=True)
    
    # Configure theme and colors
    sns.set_style("whitegrid")
    sns.set_palette("Oranges")
    warm_orange = "#ff7f0e"
    
    # 3.1. Missing value heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(train_df.isnull(), cbar=False, cmap='Oranges', yticklabels=False)
    plt.title('Missing Values Heatmap', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'missing_values_heatmap.png'), dpi=150)
    plt.close()
    
    # 3.2. Sales distribution histogram
    if 'Item_Outlet_Sales' in train_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(train_df['Item_Outlet_Sales'], bins=50, kde=True, color=warm_orange)
        plt.title('Item Outlet Sales Distribution', fontsize=16)
        plt.xlabel('Sales')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'sales_distribution.png'), dpi=150)
        plt.close()
    
    # 3.3. Item_Type vs average sales bar chart
    if 'Item_Type' in train_df.columns and 'Item_Outlet_Sales' in train_df.columns:
        plt.figure(figsize=(14, 7))
        avg_sales_by_type = train_df.groupby('Item_Type')['Item_Outlet_Sales'].mean().sort_values(ascending=False).reset_index()
        sns.barplot(data=avg_sales_by_type, x='Item_Type', y='Item_Outlet_Sales', hue='Item_Type', legend=False, palette='Oranges_r')
        plt.title('Average Sales by Item Type', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Average Sales (Score)')
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'item_type_vs_sales_bar.png'), dpi=150)
        plt.close()
    
    # 3.4. Outlet_Type vs sales box plot
    if 'Outlet_Type' in train_df.columns and 'Item_Outlet_Sales' in train_df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=train_df, x='Outlet_Type', y='Item_Outlet_Sales', hue='Outlet_Type', legend=False, palette='Oranges')
        plt.title('Sales Distribution across Outlet Types', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'outlet_type_vs_sales_boxplot.png'), dpi=150)
        plt.close()
        
    # 3.5. Correlation heatmap of numeric columns
    plt.figure(figsize=(10, 8))
    numeric_cols = train_df.select_dtypes(include=[np.number])
    if not numeric_cols.empty:
        corr_matrix = numeric_cols.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='Oranges', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Heatmap (Numeric Features)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'correlation_heatmap.png'), dpi=150)
        plt.close()

    print(f"SUCCESS: All visualizations successfully saved as PNGs to '{charts_dir}' at 150 DPI.\n")
    
    print("-" * 50)
    print("4. SUMMARY REPORT")
    print("-" * 50)
    print(f"Total Observations (Rows): {len(train_df)}")
    print(f"Total Features (Columns): {len(train_df.columns)}")
    print("\nFeatures with Missing Values:")
    missing = train_df.isnull().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        print(missing_cols)
    else:
        print("None. Dataset is perfectly clean.")
    print("-" * 50)
    
    return train_df, test_df

def preprocess(train_df, test_df=None, models_dir='models'):
    """
    Data Cleaning, Feature Engineering, Encoding, and Scaling pipeline.
    """
    print("-" * 50)
    print("STARTING DATA PREPROCESSING & FEATURE ENGINEERING")
    print("-" * 50)
    
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. MISSING VALUE HANDLING (Using Train stats)
    # Item_Weight: median grouped by Item_Type
    weight_medians = train_df.groupby('Item_Type')['Item_Weight'].median()
    train_df['Item_Weight'] = train_df.apply(
        lambda row: weight_medians[row['Item_Type']] if pd.isnull(row['Item_Weight']) else row['Item_Weight'], axis=1
    )
    if test_df is not None and not test_df.empty:
        test_df['Item_Weight'] = test_df.apply(
            lambda row: weight_medians[row['Item_Type']] if pd.isnull(row['Item_Weight']) else row['Item_Weight'], axis=1
        )
        
    # Outlet_Size: mode grouped by Outlet_Type
    size_modes = train_df.dropna(subset=['Outlet_Size']).groupby('Outlet_Type')['Outlet_Size'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Medium')
    
    train_df['Outlet_Size'] = train_df.apply(
        lambda row: size_modes.get(row['Outlet_Type'], 'Medium') if pd.isnull(row['Outlet_Size']) else row['Outlet_Size'], axis=1
    )
    if test_df is not None and not test_df.empty:
        test_df['Outlet_Size'] = test_df.apply(
            lambda row: size_modes.get(row['Outlet_Type'], 'Medium') if pd.isnull(row['Outlet_Size']) else row['Outlet_Size'], axis=1
        )

    # Combine for easier feature engineering
    train_df['is_train'] = 1
    if test_df is not None and not test_df.empty:
        test_df['is_train'] = 0
        df = pd.concat([train_df, test_df], ignore_index=True)
    else:
        df = train_df.copy()

    # Normalize Item_Fat_Content
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})

    # 2. FEATURE ENGINEERING
    # Outlet_Age
    df['Outlet_Age'] = 2013 - df['Outlet_Establishment_Year']
    
    # Item_Visibility_MeanRatio
    visibility_means = train_df.groupby('Item_Type')['Item_Visibility'].mean()
    df['Item_Visibility'] = df.apply(
        lambda row: visibility_means.get(row['Item_Type'], 0.05) if row['Item_Visibility'] == 0 else row['Item_Visibility'], axis=1
    )
    df['Item_Visibility_MeanRatio'] = df.apply(lambda row: row['Item_Visibility'] / visibility_means.get(row['Item_Type'], 1.0), axis=1)
    
    # Price_Per_Unit_Weight
    df['Price_Per_Unit_Weight'] = df['Item_MRP'] / df['Item_Weight']
    
    # Bin Item_MRP into 4 price tiers
    df['MRP_Tier'] = pd.qcut(df['Item_MRP'], q=4, labels=['Budget', 'Mid', 'Premium', 'Luxury'])

    # 3. ENCODING
    encoders = {}
    
    # Label encode Item_Fat_Content
    le = LabelEncoder()
    df['Item_Fat_Content'] = le.fit_transform(df['Item_Fat_Content'])
    encoders['Item_Fat_Content'] = le
    
    # One-hot encode
    cat_cols = ['Item_Type', 'Outlet_Type', 'Outlet_Location_Type', 'Outlet_Size', 'MRP_Tier']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Fit OHE on train data
    train_cat_data = df[df['is_train'] == 1][cat_cols]
    ohe.fit(train_cat_data)
    encoders['ohe'] = ohe
    
    encoded_cats = ohe.transform(df[cat_cols])
    encoded_cat_df = pd.DataFrame(encoded_cats, columns=ohe.get_feature_names_out(cat_cols))
    
    # Drop original categorical + Identifiers + Establishment Year
    # Keep mapping dict
    mapping_dict = df[['Item_Identifier', 'Outlet_Identifier']].to_dict(orient='records')
    
    cols_to_drop = cat_cols + ['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year']
    df = df.drop(columns=cols_to_drop)
    
    # Concat OHE columns
    df = pd.concat([df.reset_index(drop=True), encoded_cat_df.reset_index(drop=True)], axis=1)

    # 4. SCALING
    num_cols_to_scale = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age']
    
    scaler = StandardScaler()
    # Fit on train
    scaler.fit(df[df['is_train'] == 1][num_cols_to_scale])
    df[num_cols_to_scale] = scaler.transform(df[num_cols_to_scale])

    # Split back to train and test
    train_processed = df[df['is_train'] == 1].drop(columns=['is_train'])
    
    y_train = train_processed['Item_Outlet_Sales'] if 'Item_Outlet_Sales' in train_processed.columns else None
    X_train = train_processed.drop(columns=['Item_Outlet_Sales']) if 'Item_Outlet_Sales' in train_processed.columns else train_processed
    
    X_test = None
    if test_df is not None and not test_df.empty:
        test_processed = df[df['is_train'] == 0].drop(columns=['is_train'])
        X_test = test_processed.drop(columns=['Item_Outlet_Sales'], errors='ignore')

    # Save scaler and encoders
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(encoders, os.path.join(models_dir, 'encoders.pkl'))
    
    # Create final dict
    feature_names = list(X_train.columns)
    
    result = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'feature_names': feature_names,
        'scaler': scaler,
        'encoders': encoders,
        'mapping_dict': mapping_dict
    }
    
    print("Preprocessing completed successfully:")
    print(f"- Saved scaler.pkl and encoders.pkl to {models_dir}/")
    print(f"- Final feature count: {len(feature_names)}\n")
    
    return result

if __name__ == "__main__":
    # Test script locally
    train_data, test_data = perform_eda()
    if train_data is not None:
        result_dict = preprocess(train_data, test_data)
        print(f"X_train shape: {result_dict['X_train'].shape}")
