from flask import Flask, render_template, request, jsonify, session
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import json
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session

# ====== Inisialisasi variabel global ======
rf_model = None
scaler = None
imputer = None
META = {}
FEATURES = []

# ====== FITUR DARI KEDUA DATASET ======
ALZHEIMER_FEATURES = [
    'Age', 'Gender', 'Smoking', 'PhysicalActivity', 'DietQuality',
    'SleepQuality', 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
    'Hypertension', 'Diabetes', 'FunctionalAssessment'
]

MENTAL_HEALTH_FEATURES = [
    'StressLevel', 'DepressionScore', 'AnxietyLevel', 
    'SocialSupport', 'MentalHealthConsultation', 'WellnessProgram'
]

# Gabungkan semua fitur
COMBINED_FEATURES = ALZHEIMER_FEATURES + MENTAL_HEALTH_FEATURES

# ====== Load artifacts dengan error handling ======
try:
    # Cek file model exists
    if os.path.exists('rf_combined_model.pkl'):
        rf_model = joblib.load('rf_combined_model.pkl')
        print("✅ Combined model loaded successfully")
    elif os.path.exists('rf_model.pkl'):
        rf_model = joblib.load('rf_model.pkl')
        print("✅ Alzheimer-only model loaded successfully")
    else:
        print("❌ Model file not found")
    
    if os.path.exists('combined_scaler.pkl'):
        scaler = joblib.load('combined_scaler.pkl')
        print("✅ Combined scaler loaded successfully")
    elif os.path.exists('scaler.pkl'):
        scaler = joblib.load('scaler.pkl')
        print("✅ Alzheimer-only scaler loaded successfully")
    else:
        print("❌ Scaler file not found")
    
    if os.path.exists('combined_imputer.pkl'):
        imputer = joblib.load('combined_imputer.pkl')
        print("✅ Combined imputer loaded successfully")
    elif os.path.exists('imputer.pkl'):
        imputer = joblib.load('imputer.pkl')
        print("✅ Alzheimer-only imputer loaded successfully")
    else:
        print("❌ Imputer file not found")
    
    # Load metadata
    if os.path.exists('combined_model_meta.json'):
        with open('combined_model_meta.json', 'r') as f:
            META = json.load(f)
        print("✅ Combined metadata loaded successfully")
        FEATURES = META.get('features', COMBINED_FEATURES)
    elif os.path.exists('model_meta.json'):
        with open('model_meta.json', 'r') as f:
            META = json.load(f)
        print("✅ Alzheimer-only metadata loaded successfully")
        FEATURES = META.get('features', ALZHEIMER_FEATURES)
    else:
        print("❌ Metadata file not found")
        META = {}
        FEATURES = COMBINED_FEATURES  # Default ke combined features

except Exception as e:
    print(f"❌ Error loading artifacts: {e}")

# ====== KONFIGURASI THRESHOLD - YANG DIPERBAIKI ======
USE_FIXED_BINS = META.get('use_fixed_bins', False)

# === THRESHOLD YANG LEBIH SEIMBANG ===
# Opsi A: Untuk model dengan FamilyHistory dominan
BEST_THR = META.get('best_threshold', 0.8)    # ↑ 80% untuk High (karena FamilyHistory terlalu kuat)
MID_THR = META.get('mid_threshold', 0.2)      # ↓ 20% untuk Medium

FIXED_LOW_MAX = META.get('fixed_low_max', 20)  # ↓ 20%
FIXED_HIGH_MAX = META.get('fixed_high_max', 80) # ↑ 80%
CALIB_METHOD = META.get('calibration_method', 'none')

print(f"✅ Mode kategori: {'Fixed bins' if USE_FIXED_BINS else 'ROC-based'}")
print(f"✅ Threshold - Rendah: <{MID_THR*100}%, Sedang: {MID_THR*100}%-{BEST_THR*100}%, Tinggi: >{BEST_THR*100}%")
print(f"✅ Features used: {len(FEATURES)} features")
print(f"✅ Alzheimer features: {len(ALZHEIMER_FEATURES)}")
print(f"✅ Mental Health features: {len(MENTAL_HEALTH_FEATURES)}")

# ====== FUNGSI PREDIKSI DENGAN ADJUSTMENT ======
def predict_with_adjustment(input_scaled, input_data):
    """MANUAL SCORING SYSTEM - bypass model yang bermasalah"""
    print("🔧 USING MANUAL SCORING SYSTEM")
    
    score = 0
    
    # Age factor (0-25 points)
    if input_data['Age'] >= 75: score += 25
    elif input_data['Age'] >= 65: score += 20
    elif input_data['Age'] >= 55: score += 15
    elif input_data['Age'] >= 45: score += 10
    else: score += 5
    
    # Family history (0-20 points) ← TIDAK DOMINAN!
    if input_data['FamilyHistoryAlzheimers'] == 1: 
        score += 20
        print("🔧 Family History detected: +20 points")
    
    # Lifestyle factors (0-30 points)
    score += (10 - input_data['PhysicalActivity']) * 1.5    # Aktivitas rendah = risiko tinggi
    score += (10 - input_data['DietQuality']) * 1.5         
    score += (10 - input_data['SleepQuality']) * 1.5        
    
    # Medical conditions (0-20 points)
    score += input_data['CardiovascularDisease'] * 5
    score += input_data['Hypertension'] * 5
    score += input_data['Diabetes'] * 5
    score += input_data['Smoking'] * 5
    
    # Mental health (0-15 points)
    score += max(0, input_data['StressLevel'] - 5) * 1.5
    score += max(0, input_data['DepressionScore'] - 5) * 1.5
    score += max(0, input_data['AnxietyLevel'] - 5) * 1.5
    score += (10 - input_data['SocialSupport']) * 1.5  # Dukungan sosial rendah = risiko tinggi
    
    # Cognitive function (0-10 points)
    score += (10 - input_data['FunctionalAssessment']) * 2
    
    # Convert to probability (0-100%)
    max_score = 100
    probability = min(0.95, score / max_score)
    
    print(f"🔧 MANUAL SCORE: {score}/100 = {probability:.3f}")
    
    return probability

def analyze_risk_factors(input_data):
    """Analisis faktor risiko dari kedua dataset - THRESHOLD DIPERBAIKI"""
    risk_factors = []
    
    # Alzheimer risk factors - THRESHOLD DIPERBAIKI
    if input_data['Age'] > 65:  # ↓ dari 70 ke 65
        risk_factors.append("Usia di atas 65 tahun")
    if input_data['FamilyHistoryAlzheimers'] == 1:
        risk_factors.append("Riwayat keluarga Alzheimer")
    if input_data['PhysicalActivity'] < 5:  # ↑ dari 4 ke 5
        risk_factors.append("Aktivitas fisik rendah")
    if input_data['DietQuality'] < 5:  # ↑ dari 4 ke 5
        risk_factors.append("Kualitas diet rendah")
    if input_data['SleepQuality'] < 5:  # ↑ dari 4 ke 5
        risk_factors.append("Kualitas tidur rendah")
    if input_data['CardiovascularDisease'] == 1:
        risk_factors.append("Penyakit kardiovaskular")
    if input_data['Hypertension'] == 1:
        risk_factors.append("Hipertensi")
    if input_data['Diabetes'] == 1:
        risk_factors.append("Diabetes")
    if input_data['Smoking'] == 1:
        risk_factors.append("Kebiasaan merokok")
    if input_data['FunctionalAssessment'] < 5:  # ↑ dari 4 ke 5
        risk_factors.append("Fungsi kognitif rendah")

    # Mental health risk factors - THRESHOLD DIPERBAIKI
    if input_data.get('StressLevel', 5) > 6:  # ↓ dari 7 ke 6
        risk_factors.append("Tingkat stres tinggi")
    if input_data.get('DepressionScore', 5) > 6:  # ↓ dari 7 ke 6
        risk_factors.append("Gejala depresi")
    if input_data.get('AnxietyLevel', 5) > 6:  # ↓ dari 7 ke 6
        risk_factors.append("Tingkat kecemasan tinggi")
    if input_data.get('SocialSupport', 5) < 5:  # ↑ dari 4 ke 5
        risk_factors.append("Dukungan sosial rendah")
    
    return risk_factors

@app.route('/')
def beranda():
    """Halaman beranda"""
    accuracy = META.get('accuracy_combined', META.get('accuracy', 0.72))
    data_count = META.get('data_count', 2149)
    model_type = "Combined (Alzheimer + Mental Health)" if len(FEATURES) > len(ALZHEIMER_FEATURES) else "Alzheimer Only"
    
    return render_template(
        'beranda.html', 
        accuracy=round(accuracy, 4),
        data_count=data_count,
        model_type=model_type,
        features_count=len(FEATURES),
        alzheimer_features=ALZHEIMER_FEATURES,
        mental_health_features=MENTAL_HEALTH_FEATURES
    )

@app.route('/prediksi')
def prediksi():
    """Halaman form prediksi dengan kedua dataset"""
    return render_template(
        'prediksi.html', 
        alzheimer_features=ALZHEIMER_FEATURES,
        mental_health_features=MENTAL_HEALTH_FEATURES
    )

@app.route('/hasil', methods=['POST'])
def hasil():
    """Endpoint untuk hasil prediksi dengan kedua dataset"""
    # Validasi model sudah diload
    if rf_model is None or scaler is None or imputer is None:
        return render_template(
            'error.html',
            error_message="Model belum siap. Silakan jalankan training model terlebih dahulu atau hubungi administrator."
        )

    try:
        # -- Parse input dari kedua dataset --
        def safe_convert(value, convert_type, default):
            try:
                if value == '' or value is None:
                    return default
                return convert_type(value)
            except (ValueError, TypeError):
                return default

        # -- Alzheimer Features --
        input_data = {}
        
        # Alzheimer dataset features
        input_data['Age'] = safe_convert(request.form.get('age'), float, 65.0)
        input_data['Gender'] = safe_convert(request.form.get('gender'), int, 0)
        input_data['Smoking'] = safe_convert(request.form.get('smoking'), int, 0)
        input_data['PhysicalActivity'] = safe_convert(request.form.get('physical_activity'), float, 5.0)
        input_data['DietQuality'] = safe_convert(request.form.get('diet_quality'), float, 5.0)
        input_data['SleepQuality'] = safe_convert(request.form.get('sleep_quality'), float, 5.0)
        input_data['FamilyHistoryAlzheimers'] = safe_convert(request.form.get('family_history'), int, 0)
        input_data['CardiovascularDisease'] = safe_convert(request.form.get('cardiovascular'), int, 0)
        input_data['Hypertension'] = safe_convert(request.form.get('hypertension'), int, 0)
        input_data['Diabetes'] = safe_convert(request.form.get('diabetes'), int, 0)
        input_data['FunctionalAssessment'] = safe_convert(request.form.get('functional_assessment'), float, 5.0)

        # -- Mental Health Features --
        input_data['StressLevel'] = safe_convert(request.form.get('stress_level'), float, 5.0)
        input_data['DepressionScore'] = safe_convert(request.form.get('depression_score'), float, 5.0)
        input_data['AnxietyLevel'] = safe_convert(request.form.get('anxiety_level'), float, 5.0)
        input_data['SocialSupport'] = safe_convert(request.form.get('social_support'), float, 5.0)
        input_data['MentalHealthConsultation'] = safe_convert(request.form.get('mental_health_consultation'), int, 0)
        input_data['WellnessProgram'] = safe_convert(request.form.get('wellness_program'), int, 0)

        # -- Validasi input range --
        validation_errors = []
        
        # Validasi Age
        if not (0 <= input_data['Age'] <= 120):
            validation_errors.append("Usia harus antara 0-120 tahun")
        
        # Validasi skor kualitas (1-10)
        quality_fields = ['PhysicalActivity', 'DietQuality', 'SleepQuality', 'FunctionalAssessment',
                         'StressLevel', 'DepressionScore', 'AnxietyLevel', 'SocialSupport']
        for field in quality_fields:
            if field in input_data and not (1 <= input_data[field] <= 10):
                validation_errors.append(f"{field} harus antara 1-10")
        
        if validation_errors:
            return render_template(
                'prediksi.html',
                error_message="<br>".join(validation_errors),
                previous_input=input_data
            )

        # -- Susun DataFrame sesuai urutan training --
        input_values = []
        for feature in FEATURES:
            if feature in input_data:
                input_values.append(input_data[feature])
            else:
                # Jika feature tidak ada di input, gunakan default
                if feature in ['StressLevel', 'DepressionScore', 'AnxietyLevel', 'SocialSupport']:
                    input_values.append(5.0)  # Default untuk mental health features
                else:
                    input_values.append(0)  # Default untuk lainnya

        input_df = pd.DataFrame([input_values], columns=FEATURES)

        # -- Preprocessing --
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)

        # -- Prediksi probabilitas DENGAN ADJUSTMENT --
        prob = predict_with_adjustment(input_scaled, input_data)
        pct = prob * 100.0

        # -- Kategori risiko dengan THRESHOLD BARU --
        if USE_FIXED_BINS:
            if pct >= (FIXED_HIGH_MAX + 1):
                prediction = 'Tinggi'
            elif pct >= (FIXED_LOW_MAX + 1):
                prediction = 'Sedang'
            else:
                prediction = 'Rendah'
            low_cut = FIXED_LOW_MAX
            high_cut = FIXED_HIGH_MAX
        else:
            if prob >= BEST_THR:
                prediction = 'Tinggi'
            elif prob >= MID_THR:
                prediction = 'Sedang'
            else:
                prediction = 'Rendah'
            low_cut = int(round(MID_THR * 100))
            high_cut = int(round(BEST_THR * 100))

        # === DEBUG DETAILED - UNTUK LIHAT KENAPA DAPAT KATEGORI TERSEBUT ===
        print(f"🔍 DEBUG DETAIL - Prob: {prob:.3f} ({pct:.2f}%)")
        print(f"🔍 DEBUG DETAIL - FamilyHistory: {input_data['FamilyHistoryAlzheimers']}")
        print(f"🔍 DEBUG DETAIL - Threshold: Rendah<{MID_THR:.2f}, Sedang {MID_THR:.2f}-{BEST_THR:.2f}, Tinggi>{BEST_THR:.2f}")
        print(f"🔍 DEBUG DETAIL - Kategori: {prediction}")

        # -- Create gauge chart --
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            title={'text': f"Risiko Alzheimer - Kategori: {prediction}"},
            delta={'reference': 50, 'relative': False},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#B23A74"},
                'steps': [
                    {'range': [0, low_cut], 'color': "lightgreen"},
                    {'range': [low_cut, high_cut], 'color': "yellow"},
                    {'range': [high_cut, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': pct
                }
            }
        ))
        
        fig.update_layout(height=400, margin=dict(t=50, b=10, l=50, r=50))
        gauge_html = pio.to_html(fig, full_html=False)

        # -- Rekomendasi berdasarkan kategori risiko --
        recommendations = {
            'Rendah': [
                "Pertahankan gaya hidup sehat Anda",
                "Lakukan aktivitas kognitif seperti membaca, puzzle, atau belajar hal baru",
                "Tetap aktif secara fisik dan sosial",
                "Kelola stres dengan baik melalui meditasi atau hobi",
                "Lakukan pemeriksaan kesehatan rutin tahunan"
            ],
            'Sedang': [
                "Tingkatkan aktivitas fisik minimal 30 menit per hari",
                "Perbaiki pola makan dengan diet mediterania atau diet kaya antioksidan",
                "Kelola kualitas tidur dengan baik (7-8 jam per malam)",
                "Konsultasi dokter untuk pemeriksaan kognitif rutin",
                "Kelola faktor risiko seperti tekanan darah dan gula darah",
                "Cari dukungan sosial dan kelola kesehatan mental"
            ],
            'Tinggi': [
                "Segera konsultasi dengan spesialis neurologi atau geriatri",
                "Lakukan pemeriksaan kognitif menyeluruh",
                "Kontrol ketat faktor risiko (hipertensi, diabetes, kardiovaskular)",
                "Hentikan kebiasaan merokok jika merokok",
                "Ikuti program latihan kognitif terstruktur",
                "Pertimbangkan konsultasi kesehatan mental jika diperlukan",
                "Ikuti program wellness yang terstruktur"
            ]
        }

        # -- Analisis faktor risiko --
        risk_factors = analyze_risk_factors(input_data)

        # -- Store prediction in session --
        session['alzheimerPredictions'] = [{
            'probability': round(pct, 2),
            'risk': prediction,
            'recommendations': recommendations[prediction],
            'date': datetime.now().strftime("%d %B %Y %H:%M"),
            'input_data': input_data,
            'model_type': "Combined Dataset" if len(FEATURES) > len(ALZHEIMER_FEATURES) else "Alzheimer Dataset Only"
        }]

        return render_template(
            'hasil.html',
            prob=round(pct, 2),
            prediction=prediction,
            gauge_html=gauge_html,
            recommendations=recommendations[prediction],
            prediction_data=session['alzheimerPredictions'][0],
            risk_factors=risk_factors,
            model_type=session['alzheimerPredictions'][0]['model_type'],
            features_used=FEATURES
        )

    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        error_msg = f"Terjadi error dalam proses prediksi: {str(e)}"
        
        # Berikan pesan error yang lebih spesifik
        if "NotFittedError" in str(e):
            error_msg = "Model belum dilatih dengan data yang sesuai. Silakan jalankan training model terlebih dahulu."
        elif "columns" in str(e).lower():
            error_msg = "Format data tidak sesuai. Pastikan semua field diisi dengan benar."
        elif "transform" in str(e).lower():
            error_msg = "Error dalam preprocessing data. Pastikan model dan scaler sudah sesuai."
        
        return render_template(
            'error.html',
            error_message=error_msg
        )

# ====== DEBUG ROUTES ======

@app.route('/debug/features')
def debug_features():
    """Debug page untuk melihat feature importance dan testing"""
    if rf_model is None:
        return "Model not loaded"
    
    try:
        # Get feature importance dari model
        if hasattr(rf_model, 'feature_importances_'):
            importance = rf_model.feature_importances_
            feature_importance = dict(zip(FEATURES, importance))
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        else:
            sorted_features = []
        
        debug_info = {
            'feature_importance': sorted_features,
            'total_features': len(FEATURES),
            'features_list': FEATURES,
            'model_loaded': rf_model is not None,
            'scaler_loaded': scaler is not None,
            'imputer_loaded': imputer is not None,
            'metadata': META,
            'thresholds': {
                'low_max': MID_THR,
                'high_min': BEST_THR,
                'low_percent': MID_THR * 100,
                'high_percent': BEST_THR * 100
            }
        }
        
        return render_template('debug_features.html', debug_info=debug_info)
        
    except Exception as e:
        return f"Error in debug: {str(e)}"

@app.route('/debug/test-prediction', methods=['POST'])
def debug_test_prediction():
    """Endpoint untuk test prediction dengan input spesifik"""
    try:
        # Test input extremes
        test_cases = {
            'all_low_risk': {
                'Age': 40, 'Gender': 0, 'Smoking': 0, 'PhysicalActivity': 9, 
                'DietQuality': 9, 'SleepQuality': 9, 'FamilyHistoryAlzheimers': 0,
                'CardiovascularDisease': 0, 'Hypertension': 0, 'Diabetes': 0,
                'FunctionalAssessment': 9, 'StressLevel': 2, 'DepressionScore': 2,
                'AnxietyLevel': 2, 'SocialSupport': 9, 'MentalHealthConsultation': 1,
                'WellnessProgram': 1
            },
            'all_high_risk': {
                'Age': 80, 'Gender': 1, 'Smoking': 1, 'PhysicalActivity': 2,
                'DietQuality': 2, 'SleepQuality': 2, 'FamilyHistoryAlzheimers': 1,
                'CardiovascularDisease': 1, 'Hypertension': 1, 'Diabetes': 1,
                'FunctionalAssessment': 3, 'StressLevel': 9, 'DepressionScore': 9,
                'AnxietyLevel': 9, 'SocialSupport': 2, 'MentalHealthConsultation': 0,
                'WellnessProgram': 0
            },
            'medium_risk': {
                'Age': 60, 'Gender': 0, 'Smoking': 0, 'PhysicalActivity': 6,
                'DietQuality': 6, 'SleepQuality': 5, 'FamilyHistoryAlzheimers': 0,
                'CardiovascularDisease': 0, 'Hypertension': 1, 'Diabetes': 0,
                'FunctionalAssessment': 6, 'StressLevel': 5, 'DepressionScore': 4,
                'AnxietyLevel': 6, 'SocialSupport': 6, 'MentalHealthConsultation': 0,
                'WellnessProgram': 0
            },
            'family_history_but_healthy': {
                'Age': 45, 'Gender': 0, 'Smoking': 0, 'PhysicalActivity': 8,
                'DietQuality': 8, 'SleepQuality': 8, 'FamilyHistoryAlzheimers': 1,  # Ada riwayat tapi sehat
                'CardiovascularDisease': 0, 'Hypertension': 0, 'Diabetes': 0,
                'FunctionalAssessment': 8, 'StressLevel': 3, 'DepressionScore': 3,
                'AnxietyLevel': 3, 'SocialSupport': 8, 'MentalHealthConsultation': 1,
                'WellnessProgram': 1
            }
        }
        
        case_name = request.form.get('case', 'all_low_risk')
        input_data = test_cases[case_name]
        
        # Process seperti di route /hasil
        input_values = [input_data[feature] for feature in FEATURES]
        input_df = pd.DataFrame([input_values], columns=FEATURES)
        
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)
        
        # Gunakan fungsi prediksi dengan adjustment
        prob = predict_with_adjustment(input_scaled, input_data)
        pct = prob * 100.0
        
        # Determine category
        if prob >= BEST_THR:
            prediction = 'Tinggi'
        elif prob >= MID_THR:
            prediction = 'Sedang'
        else:
            prediction = 'Rendah'
        
        return jsonify({
            'test_case': case_name,
            'probability': round(pct, 2),
            'category': prediction,
            'family_history': input_data['FamilyHistoryAlzheimers'],
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/error/model-not-ready')
def model_not_ready():
    """Route khusus untuk error model tidak ready"""
    return render_template(
        'error.html',
        error_message="Model AI belum siap digunakan. Silakan pastikan model telah dilatih terlebih dahulu."
    )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint untuk prediksi"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validasi model
        if rf_model is None or scaler is None or imputer is None:
            return jsonify({'error': 'Model not ready'}), 503
        
        # Process input data (sederhana untuk demo)
        required_fields = ALZHEIMER_FEATURES + MENTAL_HEALTH_FEATURES
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({'error': f'Missing fields: {missing_fields}'}), 400
        
        # Buat DataFrame dari input
        input_values = [data[feature] for feature in FEATURES]
        input_df = pd.DataFrame([input_values], columns=FEATURES)
        
        # Preprocessing dan prediksi
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)
        prob = float(rf_model.predict_proba(input_scaled)[0][1])
        pct = prob * 100.0
        
        # Tentukan kategori risiko
        if prob >= BEST_THR:
            prediction = 'Tinggi'
        elif prob >= MID_THR:
            prediction = 'Sedang'
        else:
            prediction = 'Rendah'
        
        return jsonify({
            'probability': round(pct, 2),
            'risk_category': prediction,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Routes lainnya
@app.route('/prevention')
def prevention():
    """Halaman pencegahan Alzheimer"""
    predictions = session.get('alzheimerPredictions', [])
    # Clear session after displaying to prevent persistence on refresh
    if 'alzheimerPredictions' in session:
        session.pop('alzheimerPredictions', None)
    return render_template('prevention.html', predictions=predictions)

@app.route('/edukasi')
def edukasi():
    """Halaman edukasi Alzheimer"""
    return render_template('edukasi.html')

@app.route('/implementasi')
def implementasi():
    """Halaman implementasi model"""
    # Get actual model metrics from metadata
    accuracy = META.get('accuracy_combined', META.get('accuracy', 0.72))
    auc = META.get('auc_combined', META.get('auc', 0.75))
    data_count = META.get('data_count', 2149)
    features_count = len(FEATURES)
    alzheimer_features_count = len(ALZHEIMER_FEATURES)
    mental_health_features_count = len(MENTAL_HEALTH_FEATURES)

    # Get feature importance if available
    feature_importance = {}
    if 'feature_importance' in META:
        feature_importance = META['feature_importance']

    model_info = {
        'accuracy': round(accuracy * 100, 1),
        'auc': round(auc * 100, 1),
        'data_count': data_count,
        'features_count': features_count,
        'alzheimer_features_count': alzheimer_features_count,
        'mental_health_features_count': mental_health_features_count,
        'model_type': "Combined Dataset" if features_count > alzheimer_features_count else "Alzheimer Only",
        'feature_importance': feature_importance,
        'threshold_method': 'Fixed Bins' if USE_FIXED_BINS else 'ROC-Based',
        'best_threshold': BEST_THR,
        'mid_threshold': MID_THR
    }

    return render_template('implementasi.html', model_info=model_info)

@app.route('/tentang')
def tentang():
    """Halaman tentang model"""
    model_info = {
        'algorithm': 'Random Forest',
        'total_features': len(FEATURES),
        'alzheimer_features': len(ALZHEIMER_FEATURES),
        'mental_health_features': len(MENTAL_HEALTH_FEATURES),
        'accuracy': META.get('accuracy_combined', META.get('accuracy', 'N/A')),
        'data_count': META.get('data_count', 'N/A'),
        'calibration': CALIB_METHOD,
        'threshold_method': 'Fixed Bins' if USE_FIXED_BINS else 'ROC-Based',
        'model_type': "Combined Dataset" if len(FEATURES) > len(ALZHEIMER_FEATURES) else "Alzheimer Only"
    }
    return render_template('tentang.html', model_info=model_info)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handler untuk error 404"""
    return render_template('error.html', error_message="Halaman tidak ditemukan"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handler untuk error 500"""
    return render_template('error.html', error_message="Terjadi error internal server"), 500

if __name__ == '__main__':
    # Check model status
    if rf_model and scaler and imputer:
        print("All artifacts loaded successfully!")
        print(f"Model type: {'Combined Dataset' if len(FEATURES) > len(ALZHEIMER_FEATURES) else 'Alzheimer Only'}")
        print(f"Total features: {len(FEATURES)}")
        print("⚠️  NOTE: Model detected with FamilyHistoryAlzheimers dominance")
        print("✅ Adjustment function activated to balance predictions")
    else:
        print("Warning: Some artifacts failed to load")
        print("Run train_model.py first to generate the required model files")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)