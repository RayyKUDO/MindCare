import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

class AlzheimerModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.imputers = {}
        self.results = {}
        self.feature_importances = {}
        
    def load_alzheimer_data(self, filepath):
        """Load dataset Alzheimer utama"""
        try:
            df = pd.read_csv(filepath)
            print(f"✅ Alzheimer dataset loaded: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Error loading Alzheimer data: {e}")
            # Return sample data jika file tidak ada
            return self.create_sample_alzheimer_data()
    
    def load_mental_health_data(self, filepath):
        """Load dataset mental health"""
        try:
            df = pd.read_csv(filepath)
            print(f"✅ Mental Health dataset loaded: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Error loading mental health data: {e}")
            # Return sample data jika file tidak ada
            return self.create_sample_mental_health_data()
    
    def create_sample_alzheimer_data(self):
        """Create sample Alzheimer data untuk testing"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'Age': np.random.normal(65, 12, n_samples),
            'Gender': np.random.choice([0, 1], n_samples, p=[0.45, 0.55]),
            'Smoking': np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2]),
            'PhysicalActivity': np.random.randint(1, 11, n_samples),
            'DietQuality': np.random.randint(1, 11, n_samples),
            'SleepQuality': np.random.randint(1, 11, n_samples),
            'FamilyHistoryAlzheimers': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'CardiovascularDisease': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'Hypertension': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'Diabetes': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'FunctionalAssessment': np.random.randint(1, 11, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable based on risk factors
        risk_score = (
            (df['Age'] > 65) * 2 +
            (df['FamilyHistoryAlzheimers'] == 1) * 3 +
            (df['PhysicalActivity'] < 4) * 2 +
            (df['DietQuality'] < 4) * 2 +
            (df['SleepQuality'] < 4) * 1 +
            (df['CardiovascularDisease'] == 1) * 2 +
            (df['Hypertension'] == 1) * 1 +
            (df['Diabetes'] == 1) * 2 +
            (df['FunctionalAssessment'] < 4) * 2
        )
        
        df['Alzheimer_Risk'] = (risk_score > 8).astype(int)
        print("✅ Sample Alzheimer data created for testing")
        return df
    
    def create_sample_mental_health_data(self):
        """Create sample mental health data untuk testing"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'StressLevel': np.random.randint(1, 11, n_samples),
            'DepressionScore': np.random.randint(1, 11, n_samples),
            'AnxietyLevel': np.random.randint(1, 11, n_samples),
            'SocialSupport': np.random.randint(1, 11, n_samples),
            'MentalHealthConsultation': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'WellnessProgram': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        }
        
        df = pd.DataFrame(data)
        print("✅ Sample Mental Health data created for testing")
        return df
    
    def preprocess_alzheimer_data(self, df):
        """Preprocess dataset Alzheimer"""
        print("🔄 Preprocessing Alzheimer data...")
        
        # Fitur yang akan digunakan
        ALZHEIMER_FEATURES = [
            'Age', 'Gender', 'Smoking', 'PhysicalActivity', 'DietQuality',
            'SleepQuality', 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
            'Hypertension', 'Diabetes', 'FunctionalAssessment'
        ]
        
        # Cek ketersediaan fitur
        available_features = [f for f in ALZHEIMER_FEATURES if f in df.columns]
        missing_features = [f for f in ALZHEIMER_FEATURES if f not in df.columns]
        
        if missing_features:
            print(f"⚠️ Missing features in Alzheimer data: {missing_features}")
        
        df_selected = df[available_features].copy()
        
        # Cari target column
        target_column = self._find_target_column(df)
        print(f"🎯 Target column: {target_column}")
        
        y = df[target_column].copy()
        
        # Preprocess target
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            print(f"📊 Target classes: {le.classes_}")
        
        # Preprocess features
        X = df_selected.copy()
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            print(f"🔤 Encoded categorical features: {categorical_cols.tolist()}")
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        print(f"📈 Processed Alzheimer data: {X.shape}")
        return X, y, available_features, imputer
    
    def preprocess_mental_health_data(self, df):
        """Preprocess dataset mental health"""
        print("🔄 Preprocessing Mental Health data...")
        
        # Pilih fitur mental health yang relevan
        MENTAL_HEALTH_FEATURES = [
            'StressLevel', 'DepressionScore', 'AnxietyLevel', 
            'SocialSupport', 'MentalHealthConsultation', 'WellnessProgram'
        ]
        
        # Cek ketersediaan fitur
        available_features = [f for f in MENTAL_HEALTH_FEATURES if f in df.columns]
        missing_features = [f for f in MENTAL_HEALTH_FEATURES if f not in df.columns]
        
        if missing_features:
            print(f"⚠️ Missing mental health features: {missing_features}")
            # Gunakan semua kolom numerik jika fitur spesifik tidak ada
            available_features = df.select_dtypes(include=[np.number]).columns[:6].tolist()
        
        print(f"🧠 Mental Health features selected: {available_features}")
        
        X_mh = df[available_features].copy()
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_mh = pd.DataFrame(imputer.fit_transform(X_mh), columns=X_mh.columns)
        
        # Handle categorical jika ada
        categorical_cols = X_mh.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            X_mh = pd.get_dummies(X_mh, columns=categorical_cols, drop_first=True)
        
        print(f"📈 Processed Mental Health data: {X_mh.shape}")
        return X_mh, available_features, imputer
    
    def _find_target_column(self, df):
        """Find target column automatically"""
        # Prioritize 'Diagnosis' column if it exists
        if 'Diagnosis' in df.columns:
            return 'Diagnosis'

        target_keywords = ['alzheimer', 'diagnosis', 'status', 'target', 'class', 'dementia', 'result', 'risk']

        for col in df.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                return col

        # Jika tidak ditemukan, gunakan kolom terakhir
        return df.columns[-1]
    
    def combine_datasets(self, X_alz, y_alz, X_mh):
        """Gabungkan kedua dataset"""
        print("🔄 Combining datasets...")
        
        # Pastikan jumlah sampel sama
        min_samples = min(len(X_alz), len(X_mh))
        X_alz_combined = X_alz.iloc[:min_samples].copy()
        X_mh_combined = X_mh.iloc[:min_samples].copy()
        y_combined = y_alz.iloc[:min_samples].copy()
        
        # Reset index
        X_alz_combined.reset_index(drop=True, inplace=True)
        X_mh_combined.reset_index(drop=True, inplace=True)
        y_combined.reset_index(drop=True, inplace=True)
        
        # Gabungkan features
        X_combined = pd.concat([X_alz_combined, X_mh_combined], axis=1)
        
        print(f"✅ Combined dataset: {X_combined.shape}")
        return X_combined, y_combined
    
    def train_models_comparison(self, X_alz, y_alz, X_combined, y_combined):
        """Train dan bandingkan model dengan dataset berbeda"""
        print("\n🎯 Training models comparison...")

        # Split data untuk Alzheimer-only
        X_train_alz, X_test_alz, y_train_alz, y_test_alz = train_test_split(
            X_alz, y_alz, test_size=0.2, random_state=42, stratify=y_alz
        )

        # Split data untuk Combined
        X_train_comb, X_test_comb, y_train_comb, y_test_comb = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
        )

        # Fit imputers on training data (to avoid data leakage)
        imputer_alz = SimpleImputer(strategy='median')
        imputer_alz.fit(X_train_alz)

        imputer_comb = SimpleImputer(strategy='median')
        imputer_comb.fit(X_train_comb)

        # Transform data
        X_train_alz_imputed = imputer_alz.transform(X_train_alz)
        X_test_alz_imputed = imputer_alz.transform(X_test_alz)

        X_train_comb_imputed = imputer_comb.transform(X_train_comb)
        X_test_comb_imputed = imputer_comb.transform(X_test_comb)

        # Scale features
        scaler_alz = StandardScaler()
        scaler_comb = StandardScaler()

        X_train_alz_scaled = scaler_alz.fit_transform(X_train_alz_imputed)
        X_test_alz_scaled = scaler_alz.transform(X_test_alz_imputed)

        X_train_comb_scaled = scaler_comb.fit_transform(X_train_comb_imputed)
        X_test_comb_scaled = scaler_comb.transform(X_test_comb_imputed)
        
        # Model parameters
        rf_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        # Train Alzheimer-only model
        print("📊 Training Alzheimer-only model...")
        rf_alz = RandomForestClassifier(**rf_params)
        rf_alz.fit(X_train_alz_scaled, y_train_alz)
        
        # Train Combined model
        print("📊 Training Combined model...")
        rf_comb = RandomForestClassifier(**rf_params)
        rf_comb.fit(X_train_comb_scaled, y_train_comb)
        
        # Store imputers
        self.imputers = {
            'Alzheimer_Only': imputer_alz,
            'Combined': imputer_comb
        }

        # Evaluate models
        models = {
            'Alzheimer_Only': (rf_alz, X_test_alz_scaled, y_test_alz, scaler_alz, X_alz.columns),
            'Combined': (rf_comb, X_test_comb_scaled, y_test_comb, scaler_comb, X_combined.columns)
        }
        
        results = {}
        for model_name, (model, X_test, y_test, scaler, features) in models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
            
            results[model_name] = {
                'model': model,
                'scaler': scaler,
                'accuracy': accuracy,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'features': features.tolist(),
                'feature_importance': dict(zip(features, model.feature_importances_))
            }
            
            print(f"\n📈 {model_name} Model Performance:")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   AUC: {auc:.4f}")
            print(f"   CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.models = models
        self.results = results
        return results
    
    def plot_comparison(self):
        """Plot perbandingan performa model"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy and AUC comparison
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        auc_scores = [self.results[name]['auc'] for name in model_names]
        
        # Plot 1: Accuracy comparison
        axes[0, 0].bar(model_names, accuracies, color=['skyblue', 'lightcoral'])
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Plot 2: AUC comparison
        axes[0, 1].bar(model_names, auc_scores, color=['lightgreen', 'orange'])
        axes[0, 1].set_title('Model AUC Comparison')
        axes[0, 1].set_ylabel('AUC Score')
        for i, v in enumerate(auc_scores):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Plot 3: Feature importance Alzheimer-only
        if 'Alzheimer_Only' in self.results:
            imp_alz = self.results['Alzheimer_Only']['feature_importance']
            features_alz = list(imp_alz.keys())[:10]  # Top 10 features
            importance_alz = [imp_alz[f] for f in features_alz]
            
            axes[1, 0].barh(features_alz, importance_alz, color='skyblue')
            axes[1, 0].set_title('Top 10 Feature Importance (Alzheimer Only)')
        
        # Plot 4: Feature importance Combined
        if 'Combined' in self.results:
            imp_comb = self.results['Combined']['feature_importance']
            # Get top 15 features untuk combined
            sorted_imp = sorted(imp_comb.items(), key=lambda x: x[1], reverse=True)[:15]
            features_comb = [f[0] for f in sorted_imp]
            importance_comb = [f[1] for f in sorted_imp]
            
            axes[1, 1].barh(features_comb, importance_comb, color='lightcoral')
            axes[1, 1].set_title('Top 15 Feature Importance (Combined)')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self):
        """Save models and artifacts"""
        print("\n💾 Saving models and artifacts...")

        # Save combined model (prioritas)
        if 'Combined' in self.models:
            model, _, _, scaler, features = self.models['Combined']
            joblib.dump(model, 'rf_combined_model.pkl')
            joblib.dump(scaler, 'combined_scaler.pkl')

            # Save fitted imputer untuk combined features
            if 'Combined' in self.imputers:
                joblib.dump(self.imputers['Combined'], 'combined_imputer.pkl')
            else:
                combined_imputer = SimpleImputer(strategy='median')
                joblib.dump(combined_imputer, 'combined_imputer.pkl')
            print("✅ Combined model saved")

        # Save Alzheimer-only model
        if 'Alzheimer_Only' in self.models:
            model, _, _, scaler, features = self.models['Alzheimer_Only']
            joblib.dump(model, 'rf_model.pkl')
            joblib.dump(scaler, 'scaler.pkl')

            # Save fitted imputer untuk alzheimer features
            if 'Alzheimer_Only' in self.imputers:
                joblib.dump(self.imputers['Alzheimer_Only'], 'imputer.pkl')
            else:
                alzheimer_imputer = SimpleImputer(strategy='median')
                joblib.dump(alzheimer_imputer, 'imputer.pkl')
            print("✅ Alzheimer-only model saved")
        
        # Save metadata
        metadata = {
            'features': self.results.get('Combined', {}).get('features', []),
            'alzheimer_features': self.results.get('Alzheimer_Only', {}).get('features', []),
            'accuracy_combined': self.results.get('Combined', {}).get('accuracy', 0),
            'accuracy_alzheimer_only': self.results.get('Alzheimer_Only', {}).get('accuracy', 0),
            'auc_combined': self.results.get('Combined', {}).get('auc', 0),
            'auc_alzheimer_only': self.results.get('Alzheimer_Only', {}).get('auc', 0),
            'best_threshold': 0.7,
            'mid_threshold': 0.3,
            'use_fixed_bins': True,
            'fixed_low_max': 30,
            'fixed_high_max': 70,
            'calibration_method': 'none',
            'data_count': len(self.results.get('Combined', {}).get('features', [])),
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        with open('combined_model_meta.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ Metadata saved")
        print("\n📊 Final Results Summary:")
        print(f"   Alzheimer-only Accuracy: {metadata['accuracy_alzheimer_only']:.4f}")
        print(f"   Combined Model Accuracy: {metadata['accuracy_combined']:.4f}")
        improvement = metadata['accuracy_combined'] - metadata['accuracy_alzheimer_only']
        print(f"   Improvement: {improvement:.4f} ({improvement*100:.2f}%)")

def main():
    """Main training function"""
    print("🚀 Starting Alzheimer Prediction Model Training...")
    print("=" * 60)
    
    trainer = AlzheimerModelTrainer()
    
    # Load datasets
    df_alz = trainer.load_alzheimer_data('alzheimer_dataset.csv')
    df_mh = trainer.load_mental_health_data('mental_health_dataset.csv')
    
    if df_alz is None:
        print("❌ Cannot proceed without Alzheimer dataset")
        return
    
    # Preprocess Alzheimer data
    X_alz, y_alz, alz_features, imputer_alz = trainer.preprocess_alzheimer_data(df_alz)
    
    # Preprocess Mental Health data jika available
    if df_mh is not None:
        X_mh, mh_features, imputer_mh = trainer.preprocess_mental_health_data(df_mh)
        
        # Combine datasets
        X_combined, y_combined = trainer.combine_datasets(X_alz, y_alz, X_mh)
        
        # Train and compare models
        results = trainer.train_models_comparison(X_alz, y_alz, X_combined, y_combined)
        
        # Plot comparison
        trainer.plot_comparison()
        
        # Save models
        trainer.save_models()
        
        print("\n🎉 Training completed successfully!")
        print("=" * 60)
        print("📁 Files generated:")
        print("   - rf_combined_model.pkl (Model dengan kedua dataset)")
        print("   - rf_model.pkl (Model Alzheimer-only)")
        print("   - combined_scaler.pkl, scaler.pkl")
        print("   - combined_imputer.pkl, imputer.pkl")
        print("   - combined_model_meta.json")
        print("   - model_comparison.png")
        
    else:
        print("⚠️  Mental Health dataset not available, training Alzheimer-only model...")
        # Train hanya dengan Alzheimer data
        X_train, X_test, y_train, y_test = train_test_split(
            X_alz, y_alz, test_size=0.2, random_state=42, stratify=y_alz
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5, random_state=42
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = rf_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"📊 Alzheimer-only Model Accuracy: {accuracy:.4f}")
        
        # Save Alzheimer-only model
        joblib.dump(rf_model, 'rf_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(imputer_alz, 'imputer.pkl')
        
        # Save metadata
        metadata = {
            'features': X_alz.columns.tolist(),
            'accuracy': accuracy,
            'data_count': len(X_alz),
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        with open('model_meta.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ Alzheimer-only model saved")

if __name__ == "__main__":
    main()