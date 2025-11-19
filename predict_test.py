import joblib
import pandas as pd
import json
import os

# Load combined model artifacts with fallback
try:
    if os.path.exists('rf_combined_model.pkl'):
        rf_model = joblib.load('rf_combined_model.pkl')
        print("✅ Loaded combined model")
    elif os.path.exists('rf_model.pkl'):
        rf_model = joblib.load('rf_model.pkl')
        print("✅ Loaded Alzheimer-only model")
    else:
        raise FileNotFoundError("No model file found")

    if os.path.exists('combined_scaler.pkl'):
        scaler = joblib.load('combined_scaler.pkl')
        print("✅ Loaded combined scaler")
    elif os.path.exists('scaler.pkl'):
        scaler = joblib.load('scaler.pkl')
        print("✅ Loaded Alzheimer-only scaler")
    else:
        raise FileNotFoundError("No scaler file found")

    if os.path.exists('combined_imputer.pkl'):
        imputer = joblib.load('combined_imputer.pkl')
        print("✅ Loaded combined imputer")
    elif os.path.exists('imputer.pkl'):
        imputer = joblib.load('imputer.pkl')
        print("✅ Loaded Alzheimer-only imputer")
    else:
        raise FileNotFoundError("No imputer file found")

    # Load metadata
    if os.path.exists('combined_model_meta.json'):
        with open('combined_model_meta.json', 'r') as f:
            META = json.load(f)
        print("✅ Loaded combined metadata")
    elif os.path.exists('model_meta.json'):
        with open('model_meta.json', 'r') as f:
            META = json.load(f)
        print("✅ Loaded Alzheimer-only metadata")
    else:
        raise FileNotFoundError("No metadata file found")

except Exception as e:
    print(f"❌ Error loading artifacts: {e}")
    exit(1)

FEATURES = META.get('features', [])
USE_FIXED_BINS = META.get('use_fixed_bins', False)
BEST_THR = META.get('best_threshold', 0.7)
MID_THR = META.get('mid_threshold', 0.4)

print(f"✅ Model type: {'Combined Dataset' if len(FEATURES) > 11 else 'Alzheimer Only'}")
print(f"✅ Features: {len(FEATURES)} features")
print(f"✅ Threshold method: {'Fixed Bins' if USE_FIXED_BINS else 'ROC-Based'}")
print(f"✅ Thresholds - Mid: {MID_THR}, Best: {BEST_THR}")

# Input dari user - Combined features (Alzheimer + Mental Health)
input_data = pd.DataFrame([[
    85,  # Age
    1,   # Gender (laki-laki)
    1,   # Smoking (ya)
    0.0, # PhysicalActivity (sangat buruk)
    0.0, # DietQuality (sangat buruk)
    0.0, # SleepQuality (sangat buruk)
    1,   # FamilyHistoryAlzheimers (ya)
    1,   # CardiovascularDisease (ya)
    1,   # Hypertension (ya)
    1,   # Diabetes (ya)
    0.0, # FunctionalAssessment (sangat buruk)
    # Mental Health features (default values if not available)
    2.0, # StressLevel
    2.0, # DepressionScore
    2.0, # AnxietyLevel
    2.0, # SocialSupport
    0,   # MentalHealthConsultation
    0    # WellnessProgram
]], columns=FEATURES)

print(f"✅ Input shape: {input_data.shape}")
print(f"✅ Input features: {list(input_data.columns)}")

# Preprocess
input_imputed = imputer.transform(input_data)
input_scaled = scaler.transform(input_imputed)

# Predict
prob = rf_model.predict_proba(input_scaled)[0][1]
pct = prob * 100.0

print(f'🔍 Probabilitas risiko Alzheimer: {prob:.4f} ({pct:.2f}%)')

# Kategori risiko berdasarkan threshold dari metadata
if USE_FIXED_BINS:
    FIXED_LOW_MAX = META.get('fixed_low_max', 30)
    FIXED_HIGH_MAX = META.get('fixed_high_max', 70)
    if pct >= (FIXED_HIGH_MAX + 1):
        prediction = 'Tinggi'
    elif pct >= (FIXED_LOW_MAX + 1):
        prediction = 'Sedang'
    else:
        prediction = 'Rendah'
    print(f"✅ Fixed bins thresholds - Low: {FIXED_LOW_MAX}%, High: {FIXED_HIGH_MAX}%")
else:
    if prob >= BEST_THR:
        prediction = 'Tinggi'
    elif prob >= MID_THR:
        prediction = 'Sedang'
    else:
        prediction = 'Rendah'
    print(f"✅ ROC-based thresholds - Mid: {MID_THR}, Best: {BEST_THR}")

print(f'🎯 Kategori risiko: {prediction}')
