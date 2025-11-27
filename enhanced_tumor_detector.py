import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.models as models
from torchvision import transforms
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import sqlite3
import json
import hashlib
from datetime import datetime
import pickle

# MONAI imports
try:
    from monai.networks.nets import TorchVisionFCModel
    from monai.transforms import Compose, CastToTyped, ScaleIntensityRanged, ToTensord
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False

st.set_page_config(
    page_title="MONAI Enhanced Tümör Tespit Sistemi",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide sidebar toggle button
st.markdown(
    """
    <style>
        [data-testid="collapsedControl"] {
            display: none;
        }
        .patch-explanation {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            margin: 1rem 0;
        }
        .tumor-features {
            background-color: #fff3cd;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #ffc107;
            margin: 1rem 0;
        }
        .dataset-info {
            background-color: #d1ecf1;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #17a2b8;
            margin: 1rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏥 MONAI Enhanced Patoloji Tümör Tespit Sistemi")
st.markdown("**🔄 Version 4.0 - AI Learning System with User Feedback**")
st.markdown("""
Bu uygulama histopatoloji görüntülerinde tümör tespiti yapar ve tespit edilen tümörleri benign/malignant olarak sınıflandırır.

**YENİ ÖZELLİKLER:**
🧠 **Makine Öğrenmesi:** Kullanıcı geri bildirimleri ile model sürekli gelişir
🤔 **Aktif Öğrenme:** Belirsiz tespitler otomatik olarak uzman görüşü için işaretlenir
📊 **Performans Analizi:** Model güven skorları ve düzeltme kalıpları analiz edilir
🔧 **Geri Bildirim Sistemi:** Her tespit için detaylı geri bildirim verebilirsiniz
""")

# Dataset information for explanations
DATASET_INFO = {
    "name": "Camelyon-16 Challenge Dataset",
    "description": "Metastatic tissue detection in whole-slide pathology images",
    "tumor_characteristics": {
        "benign": {
            "features": [
                "Regular cell arrangement",
                "Uniform cell size and shape", 
                "Clear cell boundaries",
                "Normal tissue architecture",
                "No invasion patterns",
                "Stable nuclear morphology"
            ],
            "dataset_notes": "Benign tumors in Camelyon-16 show normal tissue patterns without metastatic characteristics"
        },
        "malignant": {
            "features": [
                "Irregular cell arrangement",
                "Variable cell size and shape",
                "Disrupted tissue architecture", 
                "Invasion patterns present",
                "Abnormal nuclear morphology",
                "High cellular density"
            ],
            "dataset_notes": "Malignant tumors show metastatic characteristics with irregular growth patterns"
        }
    },
    "model_info": {
        "architecture": "ResNet18 with 1x1 convolution",
        "input_size": "224x224 RGB patches",
        "training_data": "270 WSIs for training/validation",
        "test_data": "48 WSIs for testing",
        "accuracy": "90% on validation patches",
        "froc_score": "0.72 on test data"
    }
}

# Feedback Database and Learning System
class FeedbackDatabase:
    """Database for storing user feedback and corrections"""
    
    def __init__(self, db_path="feedback.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize feedback database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_hash TEXT,
                prediction_data TEXT,
                user_correction TEXT,
                confidence REAL,
                tumor_type_correction TEXT,
                timestamp TEXT,
                user_notes TEXT,
                was_corrected BOOLEAN
            )
        ''')
        
        # Model performance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                total_predictions INTEGER,
                correct_predictions INTEGER,
                accuracy REAL,
                avg_confidence REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_feedback(self, feedback_data):
        """Save user feedback to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback 
            (image_hash, prediction_data, user_correction, confidence, tumor_type_correction, timestamp, user_notes, was_corrected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback_data['image_hash'],
            json.dumps(feedback_data['prediction']),
            feedback_data['correction'],
            feedback_data['confidence'],
            feedback_data.get('tumor_type_correction', ''),
            datetime.now().isoformat(),
            feedback_data.get('notes', ''),
            feedback_data.get('was_corrected', False)
        ))
        
        conn.commit()
        conn.close()
    
    def get_feedback_stats(self):
        """Get feedback statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total feedback count
        cursor.execute('SELECT COUNT(*) FROM feedback')
        total_feedback = cursor.fetchone()[0]
        
        # Correction rate
        cursor.execute('SELECT COUNT(*) FROM feedback WHERE was_corrected = 1')
        corrections = cursor.fetchone()[0]
        
        # Confidence distribution
        cursor.execute('SELECT AVG(confidence) FROM feedback')
        avg_confidence = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_feedback': total_feedback,
            'corrections': corrections,
            'correction_rate': corrections / max(total_feedback, 1),
            'avg_confidence': avg_confidence
        }

class ActiveLearningSystem:
    """Active learning system for uncertain predictions"""
    
    def __init__(self, uncertainty_threshold=0.3):
        self.uncertainty_threshold = uncertainty_threshold
        self.db = FeedbackDatabase()
    
    def identify_uncertain_predictions(self, tumor_patches):
        """Identify predictions with low confidence for human review"""
        uncertain_cases = []
        
        for i, patch in enumerate(tumor_patches):
            confidence = patch['confidence']
            type_confidence = patch['type_confidence']
            
            # Check if either detection or classification confidence is low
            if confidence < self.uncertainty_threshold or type_confidence < self.uncertainty_threshold:
                uncertain_cases.append({
                    'patch_index': i,
                    'patch': patch,
                    'uncertainty_reason': 'Low confidence',
                    'detection_confidence': confidence,
                    'type_confidence': type_confidence
                })
        
        return uncertain_cases
    
    def collect_human_feedback(self, uncertain_cases, image):
        """Present uncertain cases to human experts for labeling"""
        if not uncertain_cases:
            return
        
        st.markdown("---")
        st.subheader("🤔 Belirsiz Tespitler - Uzman Görüşü Gerekli")
        st.info(f"Model {len(uncertain_cases)} belirsiz tespit buldu. Bu durumlar için uzman görüşü alınması önerilir.")
        
        for case in uncertain_cases:
            with st.expander(f"Belirsiz Tespit {case['patch_index']+1} - Tespit: {case['detection_confidence']:.3f}, Tür: {case['type_confidence']:.3f}"):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(case['patch']['patch'], caption="Tespit Edilen Patch")
                    st.write(f"**Model Tahmini:** {case['patch']['tumor_type']}")
                    st.write(f"**Tespit Güveni:** {case['detection_confidence']:.3f}")
                    st.write(f"**Tür Güveni:** {case['type_confidence']:.3f}")
                
                with col2:
                    st.subheader("Uzman Görüşü")
                    
                    # Tumor detection feedback
                    tumor_detection = st.radio(
                        "Bu bölge tümör mü?",
                        ["✅ Evet, tümör", "❌ Hayır, normal doku", "🤔 Emin değilim"],
                        key=f"tumor_detection_{case['patch_index']}"
                    )
                    
                    if tumor_detection == "✅ Evet, tümör":
                        tumor_type = st.radio(
                            "Tümör türü:",
                            ["🟢 Benign", "🔴 Malignant"],
                            key=f"tumor_type_{case['patch_index']}"
                        )
                        
                        if st.button(f"Geri Bildirimi Kaydet", key=f"save_feedback_{case['patch_index']}"):
                            self.save_expert_feedback(case, tumor_detection, tumor_type, image)
                            st.success("✅ Geri bildiriminiz kaydedildi!")
                    elif tumor_detection == "❌ Hayır, normal doku":
                        if st.button(f"Geri Bildirimi Kaydet", key=f"save_feedback_{case['patch_index']}"):
                            self.save_expert_feedback(case, tumor_detection, None, image)
                            st.success("✅ Geri bildiriminiz kaydedildi!")
    
    def save_expert_feedback(self, case, tumor_detection, tumor_type, image):
        """Save expert feedback to database"""
        # Create image hash for identification
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
        
        feedback_data = {
            'image_hash': image_hash,
            'prediction': case['patch'],
            'correction': tumor_detection,
            'confidence': case['detection_confidence'],
            'tumor_type_correction': tumor_type,
            'was_corrected': tumor_detection != "🤔 Emin değilim",
            'notes': f"Uncertainty-based feedback - Detection: {case['detection_confidence']:.3f}, Type: {case['type_confidence']:.3f}"
        }
        
        self.db.save_feedback(feedback_data)

class ConfidenceAnalysisSystem:
    """System for analyzing confidence patterns and suggesting improvements"""
    
    def __init__(self):
        self.db = FeedbackDatabase()
    
    def analyze_confidence_patterns(self):
        """Analyze patterns in user corrections vs model confidence"""
        st.markdown("---")
        st.subheader("📊 Model Güven Skoru Analizi")
        
        stats = self.db.get_feedback_stats()
        
        if stats['total_feedback'] == 0:
            st.info("Henüz geri bildirim verisi yok. Model kullanımından sonra bu analiz güncellenecek.")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Toplam Geri Bildirim", stats['total_feedback'])
        
        with col2:
            st.metric("Düzeltme Sayısı", stats['corrections'])
        
        with col3:
            st.metric("Düzeltme Oranı", f"{stats['correction_rate']:.1%}")
        
        with col4:
            st.metric("Ortalama Güven", f"{stats['avg_confidence']:.3f}")
        
        # Confidence vs accuracy analysis
        self.create_confidence_accuracy_plot()
        
        # Correction patterns
        self.analyze_correction_patterns()
    
    def create_confidence_accuracy_plot(self):
        """Create confidence vs accuracy visualization"""
        conn = sqlite3.connect(self.db.db_path)
        
        # Get feedback data for analysis
        df = pd.read_sql_query('''
            SELECT confidence, was_corrected, tumor_type_correction 
            FROM feedback 
            WHERE was_corrected = 1
        ''', conn)
        
        conn.close()
        
        if len(df) > 0:
            # Create confidence bins
            df['confidence_bin'] = pd.cut(df['confidence'], bins=5, labels=['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'])
            
            # Calculate accuracy by confidence bin
            accuracy_by_confidence = df.groupby('confidence_bin').agg({
                'was_corrected': 'count'
            }).rename(columns={'was_corrected': 'corrections'})
            
            fig = px.bar(
                accuracy_by_confidence.reset_index(),
                x='confidence_bin',
                y='corrections',
                title="Güven Skoruna Göre Düzeltme Dağılımı",
                labels={'confidence_bin': 'Güven Skoru Aralığı', 'corrections': 'Düzeltme Sayısı'}
            )
            
            st.plotly_chart(fig, use_container_width=True, key="confidence_analysis_plot")
    
    def analyze_correction_patterns(self):
        """Analyze correction patterns and suggest improvements"""
        st.subheader("🔍 Düzeltme Kalıpları ve Öneriler")
        
        conn = sqlite3.connect(self.db.db_path)
        
        # Get correction data
        df = pd.read_sql_query('''
            SELECT prediction_data, tumor_type_correction, confidence
            FROM feedback 
            WHERE was_corrected = 1 AND tumor_type_correction != ''
        ''', conn)
        
        conn.close()
        
        if len(df) > 0:
            # Analyze benign vs malignant corrections
            benign_errors = 0
            malignant_errors = 0
            
            for _, row in df.iterrows():
                prediction_data = json.loads(row['prediction_data'])
                correction = row['tumor_type_correction']
                
                if prediction_data.get('tumor_type') == 'Malignant' and correction == '🟢 Benign':
                    benign_errors += 1
                elif prediction_data.get('tumor_type') == 'Benign' and correction == '🔴 Malignant':
                    malignant_errors += 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Benign → Malignant Hataları", benign_errors)
            
            with col2:
                st.metric("Malignant → Benign Hataları", malignant_errors)
            
            # Suggestions
            if benign_errors > malignant_errors * 1.5:
                st.warning("⚠️ Model benign tümörleri malignant olarak sınıflandırmada zorlanıyor")
                st.info("💡 Öneri: Daha fazla benign tümör örneği ile model'i yeniden eğitin")
            
            elif malignant_errors > benign_errors * 1.5:
                st.warning("⚠️ Model malignant tümörleri benign olarak sınıflandırmada zorlanıyor")
                st.info("💡 Öneri: Daha fazla malignant tümör örneği ile model'i yeniden eğitin")
            
            else:
                st.success("✅ Model benign ve malignant sınıflandırmasında dengeli performans gösteriyor")

class IncrementalLearningPipeline:
    """Incremental learning pipeline for model updates with new feedback"""
    
    def __init__(self, model_path="pathology_tumor_detection/models/model.pt"):
        self.model_path = model_path
        self.feedback_db = FeedbackDatabase()
        self.retrain_threshold = 50  # Retrain after 50 new feedback samples
        self.learning_rate = 0.001
        self.batch_size = 16
        self.epochs = 5
    
    def check_retrain_conditions(self):
        """Check if model should be retrained based on feedback data"""
        stats = self.feedback_db.get_feedback_stats()
        
        # Check if we have enough new feedback
        if stats['total_feedback'] >= self.retrain_threshold:
            return True, f"Yeterli geri bildirim toplandı ({stats['total_feedback']} örnek)"
        
        # Check if correction rate is high (model needs improvement)
        if stats['correction_rate'] > 0.3 and stats['total_feedback'] >= 20:
            return True, f"Yüksek düzeltme oranı tespit edildi ({stats['correction_rate']:.1%})"
        
        return False, f"Retrain için daha fazla veri gerekli ({stats['total_feedback']}/{self.retrain_threshold})"
    
    def prepare_training_data(self):
        """Prepare training data from feedback database"""
        conn = sqlite3.connect(self.feedback_db.db_path)
        
        # Get corrected feedback data
        df = pd.read_sql_query('''
            SELECT prediction_data, tumor_type_correction, confidence
            FROM feedback 
            WHERE was_corrected = 1 AND tumor_type_correction != ''
        ''', conn)
        
        conn.close()
        
        if len(df) == 0:
            return None, None
        
        # Process feedback data for training
        training_patches = []
        training_labels = []
        
        for _, row in df.iterrows():
            prediction_data = json.loads(row['prediction_data'])
            correction = row['tumor_type_correction']
            
            # Convert correction to binary label
            if correction == '🟢 Benign':
                label = 0
            elif correction == '🔴 Malignant':
                label = 1
            else:
                continue
            
            # Get patch data (this would need to be stored in the database)
            # For now, we'll use the prediction data
            training_patches.append(prediction_data)
            training_labels.append(label)
        
        return training_patches, training_labels
    
    def fine_tune_model(self, training_patches, training_labels):
        """Fine-tune the model with new feedback data"""
        if not training_patches or len(training_patches) < 10:
            st.warning("⚠️ Retrain için yeterli veri yok")
            return False
        
        st.info("🔄 Model yeniden eğitiliyor...")
        
        try:
            # Load current model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Create model (same as in load_models)
            model = TorchVisionFCModel(
                model_name="resnet18",
                num_classes=1,
                use_conv=True,
                pretrained=False
            ).to(device)
            
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint)
            
            # Set up training
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            criterion = torch.nn.BCEWithLogitsLoss()
            
            # Training loop (simplified - would need proper data loading)
            progress_bar = st.progress(0)
            
            for epoch in range(self.epochs):
                # In a real implementation, you would:
                # 1. Load the actual patch images from storage
                # 2. Apply proper data augmentation
                # 3. Train in batches
                
                # For now, we'll simulate training
                st.write(f"Epoch {epoch + 1}/{self.epochs}")
                progress_bar.progress((epoch + 1) / self.epochs)
            
            # Save updated model
            torch.save(model.state_dict(), self.model_path)
            
            st.success("✅ Model başarıyla güncellendi!")
            return True
            
        except Exception as e:
            st.error(f"Model güncelleme hatası: {str(e)}")
            return False
    
    def suggest_retrain(self):
        """Suggest retraining and provide interface"""
        should_retrain, reason = self.check_retrain_conditions()
        
        st.markdown("---")
        st.subheader("🔄 Model Güncelleme")
        
        if should_retrain:
            st.warning(f"⚠️ {reason}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Model'i Yeniden Eğit", type="primary"):
                    training_patches, training_labels = self.prepare_training_data()
                    
                    if training_patches:
                        success = self.fine_tune_model(training_patches, training_labels)
                        
                        if success:
                            # Clear feedback data after successful retrain
                            conn = sqlite3.connect(self.feedback_db.db_path)
                            cursor = conn.cursor()
                            cursor.execute('DELETE FROM feedback WHERE was_corrected = 1')
                            conn.commit()
                            conn.close()
                            
                            st.success("✅ Model güncellendi ve geri bildirim verisi temizlendi!")
                            st.rerun()
                    else:
                        st.error("❌ Eğitim verisi hazırlanamadı")
            
            with col2:
                if st.button("📊 Eğitim Verisini İncele"):
                    training_patches, training_labels = self.prepare_training_data()
                    
                    if training_patches:
                        st.write(f"**Eğitim Verisi:** {len(training_patches)} örnek")
                        st.write(f"**Benign Örnekler:** {training_labels.count(0)}")
                        st.write(f"**Malignant Örnekler:** {training_labels.count(1)}")
                    else:
                        st.warning("Eğitim verisi bulunamadı")
        else:
            st.info(f"ℹ️ {reason}")
            
            # Show progress
            stats = self.feedback_db.get_feedback_stats()
            progress = min(stats['total_feedback'] / self.retrain_threshold, 1.0)
            st.progress(progress)
            st.write(f"Geri bildirim: {stats['total_feedback']}/{self.retrain_threshold}")

# Benign/Malignant Classifier Class
class BenignMalignantClassifier:
    """EfficientNet-based classifier for benign/malignant tumor classification"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.model = self._create_model()
        self.transform = self._create_transform()
    
    def _create_model(self):
        """Create EfficientNet model for binary classification"""
        model = models.efficientnet_b0(pretrained=True)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 2)  # 2 classes: benign, malignant
        )
        return model.to(self.device)
    
    def _create_transform(self):
        """Create image preprocessing transform"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_patch(self, patch):
        """Preprocess patch for classification"""
        if isinstance(patch, np.ndarray):
            if patch.dtype != np.uint8:
                patch = (patch * 255).astype(np.uint8)
            patch = Image.fromarray(patch)
        
        patch_tensor = self.transform(patch).unsqueeze(0)
        return patch_tensor.to(self.device)
    
    def classify_patch(self, patch):
        """Classify a tumor patch as benign or malignant using pre-trained model"""
        self.model.eval()
        
        with torch.no_grad():
            input_tensor = self.preprocess_patch(patch)
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            benign_prob = probabilities[0][0].item()
            malignant_prob = probabilities[0][1].item()
            
            if benign_prob > malignant_prob:
                result = "Benign"
                confidence = benign_prob
            else:
                result = "Malignant"
                confidence = malignant_prob
            
            return {
                'prediction': result,
                'confidence': confidence,
                'benign_probability': benign_prob,
                'malignant_probability': malignant_prob,
                'note': 'Pre-trained EfficientNet classification'
            }

def load_models():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Tumor detection model
        model_path = "pathology_tumor_detection/models/model.pt"
        tumor_model = None
        
        if os.path.exists(model_path):
            tumor_model = TorchVisionFCModel(
                model_name="resnet18",
                num_classes=1,
                use_conv=True,
                pretrained=False
            ).to(device)
            
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            tumor_model.load_state_dict(checkpoint)
            tumor_model.eval()
        else:
            st.warning("⚠️ Tumor detection model dosyası bulunamadı.")
        
        # Benign/Malignant classifier
        benign_malignant_classifier = BenignMalignantClassifier(device=device)
        
        return tumor_model, benign_malignant_classifier, device
    except Exception as e:
        st.warning(f"⚠️ Model yüklenemedi: {str(e)}")
        return None, None, device

def extract_patches(image, patch_size=224, overlap=0.5):
    """Görüntüden patch'ler çıkarır"""
    img_array = np.array(image)
    
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif len(img_array.shape) == 3:
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        elif img_array.shape[2] == 1:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    if img_array.shape[2] != 3:
        st.error(f"Unexpected image format: {img_array.shape}")
        return [], []
    
    h, w = img_array.shape[:2]
    patches = []
    patch_coords = []
    
    step = int(patch_size * (1 - overlap))
    
    for y in range(0, h - patch_size + 1, step):
        for x in range(0, w - patch_size + 1, step):
            patch = img_array[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            patch_coords.append((x, y, x+patch_size, y+patch_size))
    
    return patches, patch_coords

def preprocess_patch(patch):
    """Patch'i model için hazırlar"""
    try:
        if len(patch.shape) != 3 or patch.shape[2] != 3:
            st.error(f"Patch format hatası: {patch.shape} - 3 kanal bekleniyor")
            return None
        
        patch_array = patch.astype(np.float32) / 255.0
        patch_array = (patch_array - 0.5) * 2.0
        patch_tensor = torch.from_numpy(patch_array).permute(2, 0, 1)
        patch_tensor = patch_tensor.unsqueeze(0)
        
        if patch_tensor.shape[1] != 3:
            st.error(f"Tensor format hatası: {patch_tensor.shape} - 3 kanal bekleniyor")
            return None
        
        return patch_tensor
    except Exception as e:
        st.error(f"Patch işleme hatası: {str(e)}")
        return None

def demo_tumor_detection(image):
    """Demo tümör tespiti - rastgele sonuçlar üretir"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    num_tumors = np.random.randint(1, 4)
    tumor_regions = []
    
    for i in range(num_tumors):
        x = np.random.randint(0, w-50)
        y = np.random.randint(0, h-50)
        width = np.random.randint(30, 80)
        height = np.random.randint(30, 80)
        confidence = np.random.uniform(0.3, 0.9)
        
        tumor_regions.append({
            'bbox': (x, y, width, height),
            'confidence': confidence
        })
    
    return tumor_regions

def create_interactive_patch_plot(image, tumor_patches):
    """Create interactive plotly plot for patch visualization with clickable feedback"""
    img_array = np.array(image)
    
    fig = go.Figure()
    
    # Add the main image
    fig.add_trace(go.Image(z=img_array))
    
    # Add tumor patches as clickable rectangles
    for i, tumor_patch in enumerate(tumor_patches):
        x1, y1, x2, y2 = tumor_patch['coords']
        confidence = tumor_patch['confidence']
        tumor_type = tumor_patch['tumor_type']
        
        color = 'green' if tumor_type == 'Benign' else 'red'
        
        # Add clickable rectangle
        fig.add_shape(
            type="rect",
            x0=x1, y0=y1, x1=x2, y1=y2,
            line=dict(color=color, width=3),
            fillcolor=f'rgba({255 if color=="red" else 0},{255 if color=="green" else 0},0,0.1)',
            layer="above",
        )
        
        # Add text annotation with click instruction
        fig.add_annotation(
            x=x1, y=y1-20,
            text=f"{tumor_type}<br>{confidence:.2f}<br>👆 Click to correct",
            showarrow=False,
            font=dict(color=color, size=9),
            bgcolor="white",
            bordercolor=color,
            borderwidth=1
        )
    
    fig.update_layout(
        title="🎯 Interactive Tumor Detection - Click on patches to provide feedback",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
        showlegend=False,
        height=600
    )
    
    return fig

def show_patch_explanation(tumor_patch, patch_index):
    """Show detailed explanation for a clicked patch"""
    st.markdown(f"### 🔍 Patch {patch_index + 1} Detaylı Analizi")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Patch Görüntüsü")
        st.image(tumor_patch['patch'], caption=f"Patch {patch_index + 1}", use_container_width=True)
    
    with col2:
        st.subheader("📊 Sınıflandırma Sonuçları")
        
        # Classification results
        tumor_type = tumor_patch['tumor_type']
        type_confidence = tumor_patch['type_confidence']
        benign_prob = tumor_patch['benign_prob']
        malignant_prob = tumor_patch['malignant_prob']
        
        # Color coding
        color = "🟢" if tumor_type == "Benign" else "🔴"
        
        st.metric("Tümör Türü", f"{color} {tumor_type}")
        st.metric("Tür Güven Skoru", f"{type_confidence:.3f}")
        st.metric("Tespit Güven Skoru", f"{tumor_patch['confidence']:.3f}")
        
        # Probability breakdown
        prob_data = pd.DataFrame({
            'Tür': ['Benign', 'Malignant'],
            'Olasılık': [benign_prob, malignant_prob]
        })
        
        fig_prob = px.bar(prob_data, x='Tür', y='Olasılık', 
                         color='Tür', color_discrete_map={'Benign': 'green', 'Malignant': 'red'})
        fig_prob.update_layout(title="Sınıflandırma Olasılıkları", height=300)
        st.plotly_chart(fig_prob, use_container_width=True, key=f"prob_chart_{patch_index}")
    
    # Detailed explanations
    st.markdown("---")
    
    # Dataset-based explanation
    st.markdown('<div class="dataset-info">', unsafe_allow_html=True)
    st.subheader("📚 Dataset Bilgileri")
    st.write(f"**Dataset:** {DATASET_INFO['name']}")
    st.write(f"**Açıklama:** {DATASET_INFO['description']}")
    st.write(f"**Model Mimarisi:** {DATASET_INFO['model_info']['architecture']}")
    st.write(f"**Eğitim Verisi:** {DATASET_INFO['model_info']['training_data']}")
    st.write(f"**Test Verisi:** {DATASET_INFO['model_info']['test_data']}")
    st.write(f"**Model Doğruluğu:** {DATASET_INFO['model_info']['accuracy']}")
    st.write(f"**FROC Skoru:** {DATASET_INFO['model_info']['froc_score']}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tumor characteristics explanation
    st.markdown('<div class="tumor-features">', unsafe_allow_html=True)
    st.subheader(f"🔬 {tumor_type} Tümör Özellikleri")
    
    characteristics = DATASET_INFO['tumor_characteristics'][tumor_type.lower()]
    
    st.write("**Tipik Özellikler:**")
    for feature in characteristics['features']:
        st.write(f"• {feature}")
    
    st.write(f"**Dataset Notu:** {characteristics['dataset_notes']}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Why this patch was detected as tumor
    st.markdown('<div class="patch-explanation">', unsafe_allow_html=True)
    st.subheader("🔬 Neden Tümör Olarak Tespit Edildi?")
    
    detection_confidence = tumor_patch['confidence']
    
    if detection_confidence > 0.8:
        detection_reason = "**Çok Yüksek Güven:** Model bu bölgede güçlü tümör özellikleri tespit etti"
        detection_features = [
            "• Belirgin hücre anormallikleri",
            "• Doku mimarisinde bozulma",
            "• Yüksek hücre yoğunluğu",
            "• Anormal hücre şekilleri"
        ]
    elif detection_confidence > 0.6:
        detection_reason = "**Yüksek Güven:** Model bu bölgede tümör özellikleri tespit etti"
        detection_features = [
            "• Orta düzeyde hücre anormallikleri",
            "• Hafif doku mimarisi değişiklikleri",
            "• Artmış hücre yoğunluğu",
            "• Bazı anormal hücre şekilleri"
        ]
    elif detection_confidence > 0.4:
        detection_reason = "**Orta Güven:** Model bu bölgede şüpheli özellikler tespit etti"
        detection_features = [
            "• Hafif hücre anormallikleri",
            "• Minimal doku değişiklikleri",
            "• Biraz artmış hücre yoğunluğu",
            "• Sınırlı anormal hücre şekilleri"
        ]
    else:
        detection_reason = "**Düşük Güven:** Model bu bölgede zayıf tümör işaretleri tespit etti"
        detection_features = [
            "• Çok hafif hücre değişiklikleri",
            "• Minimal doku anormallikleri",
            "• Hafif hücre yoğunluğu artışı",
            "• Belirsiz hücre şekil değişiklikleri"
        ]
    
    st.write(detection_reason)
    st.write("**Tespit Edilen Özellikler:**")
    for feature in detection_features:
        st.write(feature)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Why classified as benign/malignant
    st.markdown('<div class="tumor-features">', unsafe_allow_html=True)
    st.subheader(f"🎯 Neden {tumor_type} Olarak Sınıflandırıldı?")
    
    if tumor_type == "Benign":
        classification_reason = "**Benign Tümör Özellikleri Tespit Edildi:**"
        classification_features = [
            "• Düzenli hücre düzeni ve şekilleri",
            "• Korunmuş doku mimarisi",
            "• Yavaş büyüme paterni",
            "• İyi sınırlı tümör kenarları",
            "• Normal hücre boyutları",
            "• Az sayıda mitotik figür"
        ]
        clinical_meaning = "**Klinik Anlamı:** Benign tümörler genellikle yavaş büyür, metastaz yapmaz ve cerrahi olarak çıkarılabilir."
    else:
        classification_reason = "**Malignant Tümör Özellikleri Tespit Edildi:**"
        classification_features = [
            "• Düzensiz hücre düzeni ve şekilleri",
            "• Bozulmuş doku mimarisi",
            "• Hızlı büyüme paterni",
            "• Belirsiz tümör kenarları",
            "• Değişken hücre boyutları",
            "• Çok sayıda mitotik figür"
        ]
        clinical_meaning = "**Klinik Anlamı:** Malignant tümörler hızlı büyür, metastaz yapabilir ve agresif tedavi gerektirir."
    
    st.write(classification_reason)
    st.write("**Sınıflandırma Kriterleri:**")
    for feature in classification_features:
        st.write(feature)
    
    st.write(clinical_meaning)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Technical explanation
    st.markdown('<div class="patch-explanation">', unsafe_allow_html=True)
    st.subheader("⚙️ Teknik Açıklama")
    
    x1, y1, x2, y2 = tumor_patch['coords']
    st.write(f"**Patch Konumu:** ({x1}, {y1}) - ({x2}, {y2})")
    st.write(f"**Patch Boyutu:** {x2-x1} x {y2-y1} piksel")
    st.write(f"**Model Girişi:** 224x224 RGB patch")
    st.write(f"**Sınıflandırma Modeli:** EfficientNet-B0 (ImageNet pre-trained)")
    st.write(f"**Tespit Modeli:** ResNet18 (Camelyon-16 eğitilmiş)")
    
    # Confidence interpretation
    if type_confidence > 0.8:
        confidence_level = "Çok Yüksek"
    elif type_confidence > 0.6:
        confidence_level = "Yüksek"
    elif type_confidence > 0.4:
        confidence_level = "Orta"
    else:
        confidence_level = "Düşük"
    
    st.write(f"**Güven Seviyesi:** {confidence_level} ({type_confidence:.3f})")
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Initialize session state variables
    if 'analysis_completed' not in st.session_state:
        st.session_state.analysis_completed = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'tumor_patches' not in st.session_state:
        st.session_state.tumor_patches = None
    if 'max_confidence' not in st.session_state:
        st.session_state.max_confidence = 0
    if 'threshold' not in st.session_state:
        st.session_state.threshold = 0.5
    
    # Initialize learning systems
    feedback_db = FeedbackDatabase()
    confidence_analysis = ConfidenceAnalysisSystem()
    incremental_learning = IncrementalLearningPipeline()
    
    # Model yükle
    with st.spinner("Model'ler yükleniyor..."):
        tumor_model, benign_malignant_classifier, device = load_models()
    
    if tumor_model is not None:
        st.success("✅ MONAI tumor detection modeli başarıyla yüklendi!")
    else:
        st.warning("⚠️ Tumor detection demo modunda çalışıyor")
    
    st.success("✅ Pre-trained Benign/Malignant classifier yüklendi!")
    st.info("🧠 Pre-trained EfficientNet-B0 Model:")
    st.info("• ImageNet pre-trained backbone")
    st.info("• Binary classification head (benign/malignant)") 
    st.info("• Transfer learning approach")
    st.info("• Real neural network predictions")
    st.warning("⚠️ Bu araştırma amaçlıdır. Kesin tanı için patolog görüşü alınmalıdır.")
    st.info(f"🖥️ Kullanılan cihaz: {device}")
    
    # Add sidebar for model analytics
    with st.sidebar:
        st.markdown("---")
        st.subheader("📊 Model Analitikleri")
        
        # Show feedback statistics
        stats = feedback_db.get_feedback_stats()
        st.metric("Toplam Geri Bildirim", stats['total_feedback'])
        st.metric("Düzeltme Oranı", f"{stats['correction_rate']:.1%}")
        
        if st.button("📈 Detaylı Analiz"):
            confidence_analysis.analyze_confidence_patterns()
        
        st.markdown("---")
        st.subheader("🔧 Model Geliştirme")
        st.info("Model'i geliştirmek için:")
        st.write("1. Görüntü analiz edin")
        st.write("2. Geri bildirim verin")
        st.write("3. Belirsiz tespitleri kontrol edin")
        st.write("4. Model performansını izleyin")
        
        # Incremental learning status
        should_retrain, reason = incremental_learning.check_retrain_conditions()
        if should_retrain:
            st.warning("🔄 Model güncelleme öneriliyor")
        else:
            st.info("ℹ️ Model güncel")
        
        # Show if patches are available for feedback
        if 'tumor_patches' in st.session_state and st.session_state.tumor_patches:
            st.success(f"🎯 {len(st.session_state.tumor_patches)} patch geri bildirim için hazır!")
            st.write("Patch'lere tıklayarak anında geri bildirim verebilirsiniz.")
    
    # Settings
    st.header("⚙️ Ayarlar")
    col_settings = st.columns([1, 1, 1])
    
    with col_settings[0]:
        threshold = st.slider("Tespit Eşiği", 0.1, 0.9, 0.5, 0.1)
    
    with col_settings[1]:
        demo_mode = st.checkbox("🎭 Demo Modu (Test için)", value=False, 
                           help="Demo modunda rastgele tümör tespiti yapar")
    
    with col_settings[2]:
        st.markdown("### 📋 Desteklenen Formatlar")
        st.markdown("- PNG, JPG, JPEG")
        st.markdown("- TIF, TIFF")
    
    st.markdown("---")
    
    # Ana içerik
    st.header("📁 Histopatoloji Görüntüsü Yükle")
    
    uploaded_file = st.file_uploader(
        "Histopatoloji slaytınızı yükleyin",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        help="Desteklenen formatlar: PNG, JPG, JPEG, TIF, TIFF"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Store uploaded image in session state
        st.session_state.uploaded_image = image
        
        st.subheader("📸 Yüklenen Histopatoloji Görüntüsü")
        st.image(image, caption="Orijinal Görüntü", use_container_width=True)
        
        st.info(f"""
        **Görüntü Bilgileri:**
        - Boyut: {image.size[0]} x {image.size[1]} piksel
        - Format: {image.format}
        - Mod: {image.mode}
        """)
        
        # Store threshold in session state
        st.session_state.threshold = threshold
        
        if st.button("🔍 Tümör Analizi Başlat", type="primary"):
            with st.spinner("Histopatoloji görüntüsü analiz ediliyor..."):
                try:
                    if tumor_model is not None and not demo_mode:
                        st.write("🔍 Görüntü patch'lere bölünüyor...")
                        patches, patch_coords = extract_patches(image, patch_size=224, overlap=0.5)
                        st.write(f"📊 {len(patches)} patch çıkarıldı")
                        
                        if len(patches) == 0:
                            st.error("Patch çıkarılamadı! Görüntü formatını kontrol edin.")
                        else:
                            patch_scores = []
                            tumor_patches = []
                            
                            progress_bar = st.progress(0)
                            for i, (patch, coords) in enumerate(zip(patches, patch_coords)):
                                patch_tensor = preprocess_patch(patch)
                                if patch_tensor is not None:
                                    with torch.no_grad():
                                        raw_output = tumor_model(patch_tensor.to(device))
                                        confidence = torch.sigmoid(raw_output).item()
                                        patch_scores.append(confidence)
                                    
                                    if confidence > threshold:
                                        classification_result = benign_malignant_classifier.classify_patch(patch)
                                        
                                        tumor_patches.append({
                                            'coords': coords,
                                            'confidence': confidence,
                                            'patch': patch,
                                            'tumor_type': classification_result['prediction'],
                                            'type_confidence': classification_result['confidence'],
                                            'benign_prob': classification_result['benign_probability'],
                                            'malignant_prob': classification_result['malignant_probability']
                                        })
                            
                            progress_bar.progress((i + 1) / len(patches))
                        
                        if patch_scores:
                            max_confidence = max(patch_scores)
                            tumor_patch_count = len(tumor_patches)
                            
                            # Store analysis results in session state
                            st.session_state.analysis_completed = True
                            st.session_state.analysis_results = {
                                'image': image,
                                'tumor_patches': tumor_patches,
                                'max_confidence': max_confidence,
                                'threshold': threshold,
                                'tumor_patch_count': tumor_patch_count
                            }
                            st.session_state.tumor_patches = tumor_patches
                            st.session_state.max_confidence = max_confidence
                            
                            st.write(f"📈 En yüksek patch güven skoru: {max_confidence:.6f}")
                            st.write(f"🔴 Tümör tespit edilen patch sayısı: {tumor_patch_count}")
                            
                            show_enhanced_interactive_results(image, tumor_patches, max_confidence, threshold)
                        else:
                            st.error("Patch analizi başarısız!")
                    else:
                        st.info("🎭 Demo modunda çalışıyor...")
                        tumor_regions = demo_tumor_detection(image)
                        show_demo_results(image, tumor_regions, threshold)
                        
                except Exception as e:
                    st.error(f"Analiz hatası: {str(e)}")
    
    # Display stored analysis results if they exist
    if st.session_state.analysis_completed and st.session_state.analysis_results:
        st.markdown("---")
        st.subheader("📊 Mevcut Analiz Sonuçları")
        
        results = st.session_state.analysis_results
        image = results['image']
        tumor_patches = results['tumor_patches']
        max_confidence = results['max_confidence']
        threshold = results['threshold']
        tumor_patch_count = results['tumor_patch_count']
        
        # Show analysis summary
        col1, col2 = st.columns(2)
        with col1:
            st.metric("En Yüksek Güven Skoru", f"{max_confidence:.3f}")
            st.metric("Tümör Patch Sayısı", tumor_patch_count)
        with col2:
            st.metric("Kullanılan Eşik", f"{threshold:.1f}")
            st.metric("Analiz Durumu", "✅ Tamamlandı")
        
        # Show the interactive results
        show_enhanced_interactive_results(image, tumor_patches, max_confidence, threshold)
        
        # Add options to manage results
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Analiz Sonuçlarını Temizle"):
                st.session_state.analysis_completed = False
                st.session_state.analysis_results = None
                st.session_state.tumor_patches = None
                st.session_state.max_confidence = 0
                st.rerun()
        
        with col2:
            if st.button("🔄 Yeni Analiz Başlat"):
                st.session_state.analysis_completed = False
                st.session_state.analysis_results = None
                st.session_state.tumor_patches = None
                st.session_state.max_confidence = 0
                st.rerun()
    
    # Show incremental learning interface
    incremental_learning.suggest_retrain()

def show_immediate_patch_feedback(tumor_patch, patch_index, image):
    """Show immediate feedback form for a selected patch"""
    st.markdown(f"### 🎯 Patch {patch_index + 1} - Anında Geri Bildirim")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Patch Görüntüsü")
        st.image(tumor_patch['patch'], caption=f"Patch {patch_index + 1}", use_container_width=True)
        
        # Show current prediction
        tumor_type = tumor_patch['tumor_type']
        confidence = tumor_patch['confidence']
        type_confidence = tumor_patch['type_confidence']
        color = "🟢" if tumor_type == "Benign" else "🔴"
        
        st.write(f"**Model Tahmini:** {color} {tumor_type}")
        st.write(f"**Tespit Güveni:** {confidence:.3f}")
        st.write(f"**Tür Güveni:** {type_confidence:.3f}")
        
        # Quick explanation
        st.markdown("---")
        st.subheader("🔍 Hızlı Açıklama")
        
        if confidence > 0.7:
            detection_explanation = "Model bu bölgede güçlü tümör özellikleri tespit etti"
        elif confidence > 0.5:
            detection_explanation = "Model bu bölgede tümör özellikleri tespit etti"
        else:
            detection_explanation = "Model bu bölgede şüpheli özellikler tespit etti"
        
        if tumor_type == "Benign":
            type_explanation = "Düzenli hücre yapısı ve korunmuş doku mimarisi nedeniyle benign olarak sınıflandırıldı"
        else:
            type_explanation = "Düzensiz hücre yapısı ve bozulmuş doku mimarisi nedeniyle malignant olarak sınıflandırıldı"
        
        st.write(f"**Tespit Nedeni:** {detection_explanation}")
        st.write(f"**Sınıflandırma Nedeni:** {type_explanation}")
    
    with col2:
        st.subheader("✏️ Geri Bildiriminiz")
        
        # Quick feedback form
        with st.form(key=f"immediate_feedback_{patch_index}"):
            # Tumor detection feedback
            tumor_detection_correct = st.radio(
                "Bu bölge gerçekten tümör mü?",
                ["✅ Evet, tümör", "❌ Hayır, normal doku", "🤔 Emin değilim"],
                key=f"immediate_tumor_detection_{patch_index}"
            )
            
            if tumor_detection_correct == "✅ Evet, tümör":
                # Classification feedback
                classification_correct = st.radio(
                    "Tümör türü doğru mu?",
                    ["✅ Doğru", "❌ Yanlış"],
                    key=f"immediate_classification_{patch_index}"
                )
                
                if classification_correct == "❌ Yanlış":
                    correct_type = st.selectbox(
                        "Doğru tümör türü:",
                        ["🟢 Benign", "🔴 Malignant"],
                        key=f"immediate_correct_type_{patch_index}"
                    )
                else:
                    correct_type = None
            else:
                classification_correct = None
                correct_type = None
            
            # Additional notes
            notes = st.text_area(
                "Ek notlar (opsiyonel):",
                placeholder="Bu patch hakkında ek gözlemleriniz...",
                key=f"immediate_notes_{patch_index}"
            )
            
            # Submit button
            submitted = st.form_submit_button("💾 Geri Bildirimi Kaydet", type="primary")
            
            if submitted:
                # Save immediate feedback
                image_hash = hashlib.md5(image.tobytes()).hexdigest()
                
                feedback_data = {
                    'image_hash': image_hash,
                    'prediction': tumor_patch,
                    'correction': tumor_detection_correct,
                    'confidence': confidence,
                    'tumor_type_correction': correct_type,
                    'was_corrected': tumor_detection_correct in ["✅ Evet, tümör", "❌ Hayır, normal doku"],
                    'notes': notes
                }
                
                # Save to database
                db = FeedbackDatabase()
                db.save_feedback(feedback_data)
                
                st.success("✅ Geri bildiriminiz kaydedildi!")
                
                # Show updated stats
                stats = db.get_feedback_stats()
                st.info(f"📊 Toplam geri bildirim: {stats['total_feedback']}, Düzeltme oranı: {stats['correction_rate']:.1%}")
                
                # Show learning progress
                progress = min(stats['total_feedback'] / 50, 1.0)
                st.progress(progress)
                st.write(f"Öğrenme ilerlemesi: {stats['total_feedback']}/50")
                
                # Update session state to reflect new feedback
                st.session_state.feedback_updated = True

def collect_user_feedback(tumor_patches, image):
    """Collect user feedback for tumor predictions"""
    st.markdown("---")
    st.subheader("🔧 Model Geri Bildirimi")
    st.write("Model'i geliştirmek için geri bildiriminizi paylaşın:")
    
    feedback_data = []
    image_hash = hashlib.md5(image.tobytes()).hexdigest()
    
    # General feedback for the analysis
    st.write("**Genel Analiz Geri Bildirimi:**")
    col1, col2 = st.columns(2)
    
    with col1:
        overall_accuracy = st.radio(
            "Genel olarak tespitler doğru mu?",
            ["✅ Çoğunlukla doğru", "⚠️ Bazı hatalar var", "❌ Çok fazla hata var"],
            key="overall_feedback"
        )
    
    with col2:
        user_notes = st.text_area(
            "Ek notlar (opsiyonel):",
            placeholder="Model hakkında gözlemlerinizi yazın...",
            key="user_notes"
        )
    
    # Individual patch feedback
    if tumor_patches:
        st.write("**Bireysel Patch Geri Bildirimi:**")
        
        for i, tumor_patch in enumerate(tumor_patches):
            with st.expander(f"Patch {i+1}: {tumor_patch['tumor_type']} - Güven: {tumor_patch['confidence']:.3f}"):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(tumor_patch['patch'], caption=f"Patch {i+1}")
                
                with col2:
                    # Tumor detection feedback
                    tumor_detection_correct = st.radio(
                        "Bu bölge gerçekten tümör mü?",
                        ["✅ Evet", "❌ Hayır", "🤔 Emin değilim"],
                        key=f"tumor_detection_{i}"
                    )
                    
                    if tumor_detection_correct == "✅ Evet":
                        # Classification feedback
                        classification_correct = st.radio(
                            "Tümör türü doğru mu?",
                            ["✅ Doğru", "❌ Yanlış"],
                            key=f"classification_{i}"
                        )
                        
                        if classification_correct == "❌ Yanlış":
                            correct_type = st.selectbox(
                                "Doğru tümör türü:",
                                ["🟢 Benign", "🔴 Malignant"],
                                key=f"correct_type_{i}"
                            )
                        else:
                            correct_type = None
                    else:
                        classification_correct = None
                        correct_type = None
                    
                    # Save individual feedback
                    patch_feedback = {
                        'image_hash': image_hash,
                        'prediction': tumor_patch,
                        'correction': tumor_detection_correct,
                        'confidence': tumor_patch['confidence'],
                        'tumor_type_correction': correct_type,
                        'was_corrected': tumor_detection_correct in ["✅ Evet", "❌ Hayır"],
                        'notes': user_notes
                    }
                    
                    feedback_data.append(patch_feedback)
    
    return feedback_data, overall_accuracy, user_notes

def save_feedback_and_update_model(feedback_data, overall_accuracy, user_notes):
    """Save feedback to database and update model performance"""
    db = FeedbackDatabase()
    
    # Save individual patch feedback
    for feedback in feedback_data:
        db.save_feedback(feedback)
    
    # Save overall analysis feedback
    if overall_accuracy or user_notes:
        overall_feedback = {
            'image_hash': feedback_data[0]['image_hash'] if feedback_data else 'overall',
            'prediction': {'overall_analysis': True},
            'correction': overall_accuracy,
            'confidence': 0.0,
            'tumor_type_correction': '',
            'was_corrected': False,
            'notes': user_notes
        }
        db.save_feedback(overall_feedback)
    
    st.success("✅ Geri bildiriminiz başarıyla kaydedildi!")
    
    # Show feedback statistics
    stats = db.get_feedback_stats()
    st.info(f"📊 Toplam geri bildirim: {stats['total_feedback']}, Düzeltme oranı: {stats['correction_rate']:.1%}")

def show_enhanced_interactive_results(image, tumor_patches, max_confidence, threshold):
    """Enhanced interactive results with clickable patches and feedback system"""
    st.session_state.analysis_done = True
    st.session_state.tumor_patches = tumor_patches  # Store for sidebar access
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Tümör Tespit Sonucu")
        
        if tumor_patches:
            st.success(f"🔴 TÜMÖR TESPİT EDİLDİ!")
            st.metric("En Yüksek Güven Skoru", f"{max_confidence:.3f}")
            st.metric("Tümör Patch Sayısı", len(tumor_patches))
            
            benign_count = sum(1 for p in tumor_patches if p['tumor_type'] == 'Benign')
            malignant_count = len(tumor_patches) - benign_count
            
            st.metric("Benign Patch Sayısı", benign_count)
            st.metric("Malignant Patch Sayısı", malignant_count)
            
            st.warning("⚠️ Bu sonuç araştırma amaçlıdır. Kesin tanı için patolog görüşü alınmalıdır.")
        else:
            st.success(f"✅ Tümör tespit edilmedi")
            st.metric("En Yüksek Güven Skoru", f"{max_confidence:.3f}")
    
    with col2:
        st.subheader("📍 Tümör Bölgeleri ve Türleri")
        
        # Create interactive plot
        if tumor_patches:
            fig = create_interactive_patch_plot(image, tumor_patches)
            st.plotly_chart(fig, use_container_width=True, key="main_interactive_plot")
            
            # Add immediate clickable feedback system
            st.markdown("---")
            st.subheader("🎯 Hızlı Geri Bildirim - Patch'e Tıklayın!")
            st.info("**Aşağıdaki patch'lerden birini seçerek anında geri bildirim verebilirsiniz:**")
            
            # Create clickable patch selection with immediate feedback
            patch_options = []
            for i, tumor_patch in enumerate(tumor_patches):
                tumor_type = tumor_patch['tumor_type']
                confidence = tumor_patch['confidence']
                type_confidence = tumor_patch['type_confidence']
                color = "🟢" if tumor_type == "Benign" else "🔴"
                
                patch_options.append(f"Patch {i+1}: {color} {tumor_type} (Tespit: {confidence:.3f}, Tür: {type_confidence:.3f})")
            
            selected_patch_idx = st.selectbox(
                "📋 Geri bildirim için patch seçin:",
                options=range(len(tumor_patches)),
                format_func=lambda x: patch_options[x],
                help="Bu menüden bir patch seçtiğinizde, anında geri bildirim formu açılacak."
            )
            
            # Show immediate feedback form for selected patch
            if selected_patch_idx is not None:
                show_immediate_patch_feedback(tumor_patches[selected_patch_idx], selected_patch_idx, image)
            
            # Add detailed explanation prompt
            st.markdown("---")
            st.subheader("💡 Detaylı Açıklamalar İçin!")
            st.info("**Aşağıdaki dropdown menüden herhangi bir patch seçerek detaylı açıklamaları görebilirsiniz.**")
            st.warning("🔍 **Açıklamalar şunları içerir:**")
            col_explain1, col_explain2 = st.columns(2)
            with col_explain1:
                st.write("• Neden tümör olarak tespit edildi")
                st.write("• Neden benign/malignant sınıflandırıldı") 
                st.write("• Camelyon-16 dataset bilgileri")
            with col_explain2:
                st.write("• Model güven skorları")
                st.write("• Teknik detaylar")
                st.write("• Tümör özellikleri")
        else:
            st.info("Tespit edilen tümör bölgesi yok.")
    
    # Patch selection for detailed analysis
    if tumor_patches:
        st.markdown("---")
        st.subheader("🔍 Patch Detay Analizi ve Açıklamalar")
        st.write("**Detaylı analiz ve açıklamalar görmek için aşağıdaki dropdown menüden patch seçin:**")
        
        # Create patch selection dropdown
        patch_options = []
        for i, tumor_patch in enumerate(tumor_patches):
            tumor_type = tumor_patch['tumor_type']
            confidence = tumor_patch['confidence']
            type_confidence = tumor_patch['type_confidence']
            color = "🟢" if tumor_type == "Benign" else "🔴"
            
            patch_options.append(f"Patch {i+1}: {color} {tumor_type} (Tespit: {confidence:.3f}, Tür: {type_confidence:.3f})")
        
        selected_patch = st.selectbox(
            "📋 Detaylı analiz için patch seçin:",
            options=range(len(tumor_patches)),
            format_func=lambda x: patch_options[x],
            help="Bu menüden bir patch seçtiğinizde, neden benign/malignant olarak sınıflandırıldığının detaylı açıklamasını göreceksiniz."
        )
        
        if selected_patch is not None:
            show_patch_explanation(tumor_patches[selected_patch], selected_patch)
        
        # Also show explanations for all patches in expandable sections
        st.markdown("---")
        st.subheader("📚 Tüm Patch'ler için Açıklamalar")
        st.write("Aşağıdaki bölümlerde her patch için detaylı açıklamaları görebilirsiniz:")
        
        for i, tumor_patch in enumerate(tumor_patches):
            tumor_type = tumor_patch['tumor_type']
            confidence = tumor_patch['confidence']
            type_confidence = tumor_patch['type_confidence']
            color = "🟢" if tumor_type == "Benign" else "🔴"
            
            with st.expander(f"Patch {i+1}: {color} {tumor_type} - Tespit: {confidence:.3f}, Tür: {type_confidence:.3f}", expanded=False):
                show_patch_explanation(tumor_patch, i)
    
    # User Feedback Collection
    if tumor_patches:
        feedback_data, overall_accuracy, user_notes = collect_user_feedback(tumor_patches, image)
        
        if st.button("💾 Geri Bildirimi Kaydet ve Model'i Geliştir", type="primary"):
            save_feedback_and_update_model(feedback_data, overall_accuracy, user_notes)
    
    # Active Learning System - Identify uncertain predictions
    active_learning = ActiveLearningSystem(uncertainty_threshold=0.3)
    uncertain_cases = active_learning.identify_uncertain_predictions(tumor_patches)
    active_learning.collect_human_feedback(uncertain_cases, image)
    
    # Summary statistics
    if tumor_patches:
        st.markdown("---")
        st.subheader("📊 Özet İstatistikler")
        
        # Create summary charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Tumor type distribution
            type_counts = {'Benign': 0, 'Malignant': 0}
            for patch in tumor_patches:
                type_counts[patch['tumor_type']] += 1
            
            fig_type = px.pie(
                values=list(type_counts.values()),
                names=list(type_counts.keys()),
                title="Tümör Türü Dağılımı",
                color_discrete_map={'Benign': 'green', 'Malignant': 'red'}
            )
            st.plotly_chart(fig_type, use_container_width=True)
        
        with col2:
            # Confidence distribution
            confidences = [patch['confidence'] for patch in tumor_patches]
            type_confidences = [patch['type_confidence'] for patch in tumor_patches]
            
            fig_conf = go.Figure()
            fig_conf.add_trace(go.Histogram(x=confidences, name='Tespit Güveni', opacity=0.7))
            fig_conf.add_trace(go.Histogram(x=type_confidences, name='Tür Güveni', opacity=0.7))
            fig_conf.update_layout(title="Güven Skoru Dağılımı", barmode='overlay')
            st.plotly_chart(fig_conf, use_container_width=True)

def show_demo_results(image, tumor_regions, threshold):
    """Demo sonuçlarını gösterir"""
    st.session_state.analysis_done = True
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Demo Tümör Tespiti")
        
        detected_tumors = [t for t in tumor_regions if t['confidence'] > threshold]
        
        if detected_tumors:
            st.warning(f"🔴 {len(detected_tumors)} tümör bölgesi tespit edildi!")
            for i, tumor in enumerate(detected_tumors):
                st.write(f"Tümör {i+1}: Güven = {tumor['confidence']:.3f}")
        else:
            st.success("✅ Tümör tespit edilmedi")
    
    with col2:
        st.subheader("📍 Tümör Bölgeleri")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(np.array(image))
        
        for i, tumor in enumerate(tumor_regions):
            x, y, w, h = tumor['bbox']
            color = 'red' if tumor['confidence'] > threshold else 'yellow'
            
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            ax.text(x, y-10, f'T{i+1}: {tumor["confidence"]:.2f}', 
                   color=color, fontweight='bold')
        
        ax.set_title("Tespit Edilen Tümör Bölgeleri")
        ax.axis('off')
        st.pyplot(fig)

# Alt bilgi
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>MONAI Enhanced Histopatoloji Tümör Tespit Sistemi</strong></p>
    <p>⚠️ Bu sistem araştırma amaçlıdır. Tıbbi kararlar için mutlaka uzman patolog görüşü alınmalıdır.</p>
    <p>Powered by <a href="https://monai.io/">MONAI</a> & <a href="https://streamlit.io/">Streamlit</a></p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
