#!/usr/bin/env python3
"""
Enhanced Cardiovascular and Respiratory Case Classifier
Addresses class imbalance and optimizes threshold to achieve >90% recall while maintaining >94% precision
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import logging
from datetime import datetime

# ML and evaluation imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, precision_recall_curve,
    roc_curve, auc, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Imbalanced learning
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedCardioRespiratoryClassifier:
    """Enhanced classifier with class imbalance handling and threshold optimization"""
    
    def __init__(self, use_smote=True, calibrate=True, target_recall=0.90, target_precision=0.94):
        """
        Initialize enhanced classifier
        
        Args:
            use_smote: Whether to use SMOTE for oversampling minority class
            calibrate: Whether to calibrate probabilities
            target_recall: Target recall threshold (default 0.90)
            target_precision: Target precision threshold (default 0.94)
        """
        # Comprehensive cardiovascular keywords
        self.cardiovascular_keywords = {
            'heart', 'cardiac', 'myocardial', 'infarction', 'mi', 'angina', 'chest pain',
            'hypertension', 'htn', 'blood pressure', 'bp', 'tachycardia', 'bradycardia',
            'arrhythmia', 'atrial fibrillation', 'afib', 'ventricular', 'coronary',
            'artery', 'arteries', 'aorta', 'valve', 'valvular', 'stenosis', 'regurgitation',
            'cardiomyopathy', 'heart failure', 'chf', 'congestive', 'edema', 'swelling',
            'pericardium', 'pericardial', 'endocarditis', 'myocarditis', 'ischemia',
            'thrombosis', 'embolism', 'clot', 'stroke', 'cva', 'cerebrovascular',
            'carotid', 'peripheral', 'vascular', 'circulation', 'pulse', 'rhythm',
            'ecg', 'ekg', 'echocardiogram', 'stress test', 'catheterization',
            'stent', 'bypass', 'cabg', 'pci', 'angioplasty', 'defibrillator',
            'pacemaker', 'cardioversion', 'nitroglycerin', 'aspirin', 'warfarin',
            'heparin', 'ace inhibitor', 'beta blocker', 'statin', 'diuretic'
        }
        
        # Comprehensive respiratory keywords
        self.respiratory_keywords = {
            'lung', 'lungs', 'pulmonary', 'respiratory', 'breathing', 'breath', 'airway',
            'trachea', 'bronchi', 'bronchial', 'alveoli', 'pneumonia', 'copd', 'asthma',
            'bronchitis', 'emphysema', 'pneumothorax', 'pleural', 'pleurisy', 'effusion',
            'tuberculosis', 'tb', 'pneumocystis', 'influenza', 'flu',
            'respiratory failure', 'ards', 'ventilation', 'ventilator', 'oxygen', 'o2',
            'saturation', 'spo2', 'hypoxia', 'hypercapnia', 'dyspnea', 'shortness of breath',
            'wheezing', 'cough', 'sputum', 'hemoptysis', 'chest x-ray', 'ct scan',
            'pulmonary function', 'spirometry', 'bronchoscopy', 'intubation', 'extubation',
            'tracheostomy', 'cpap', 'bipap', 'nebulizer', 'inhaler',
            'albuterol', 'steroid', 'prednisone', 'antibiotic', 'ventolin', 'proventil'
        }
        
        # Other medical categories to exclude
        self.other_medical_keywords = {
            # Neurological
            'brain', 'neurological', 'seizure', 'headache', 'migraine',
            'epilepsy', 'dementia', 'alzheimer', 'parkinson', 'multiple sclerosis',
            'neuropathy', 'paralysis', 'coma', 'consciousness', 'cognitive',
            
            # Gastrointestinal
            'stomach', 'gastric', 'intestinal', 'bowel', 'colon', 'rectum', 'anus',
            'liver', 'hepatic', 'gallbladder', 'bile', 'pancreas', 'pancreatic',
            'esophagus', 'gastroesophageal', 'gerd', 'ulcer', 'gastritis',
            'crohn', 'colitis', 'diverticulitis', 'appendicitis', 'hernia',
            
            # Endocrine
            'diabetes', 'diabetic', 'glucose', 'insulin', 'thyroid', 'hormone',
            'adrenal', 'pituitary', 'metabolic', 'hypoglycemia', 'hyperglycemia',
            
            # Infectious (non-respiratory)
            'infection', 'infectious', 'sepsis', 'bacterial', 'viral', 'fungal',
            'immunity', 'immune', 'vaccine',
            
            # Orthopedic
            'bone', 'fracture', 'joint', 'arthritis', 'orthopedic', 'skeletal',
            'spine', 'vertebra', 'disc', 'ligament', 'tendon', 'muscle',
            
            # Oncology
            'cancer', 'tumor', 'malignancy', 'oncology', 'chemotherapy', 'radiation',
            'metastasis', 'carcinoma', 'sarcoma', 'lymphoma', 'leukemia',
            
            # Psychiatric
            'depression', 'anxiety', 'mental', 'psychiatric', 'behavioral',
            'psychosis', 'schizophrenia', 'bipolar', 'mood', 'cognitive',
            
            # Renal
            'kidney', 'renal', 'nephrology', 'dialysis', 'urinary', 'bladder',
            'ureter', 'urethra', 'creatinine', 'bun', 'glomerular',
            
            # Hematology
            'blood', 'anemia', 'hemoglobin', 'platelet', 'coagulation', 'bleeding',
            'clotting', 'hematology', 'transfusion',
            
            # Dermatology
            'skin', 'dermatology', 'rash', 'lesion', 'wound', 'cutaneous',
            'dermatitis', 'eczema', 'psoriasis', 'melanoma', 'basal cell',
            
            # Ophthalmology
            'eye', 'ocular', 'vision', 'retinal', 'ophthalmology', 'glaucoma',
            'cataract', 'macular', 'cornea', 'iris', 'pupil'
        }
        
        self.use_smote = use_smote
        self.calibrate = calibrate
        self.target_recall = target_recall
        self.target_precision = target_precision
        
        # Initialize models with class weight support
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'  # Handle class imbalance
            ),
            'svm': SVC(
                random_state=42, 
                kernel='linear',
                class_weight='balanced',  # Handle class imbalance
                probability=True  # Enable probability estimates
            ),
            'random_forest': RandomForestClassifier(
                random_state=42, 
                n_estimators=100,
                class_weight='balanced'  # Handle class imbalance
            ),
            'naive_bayes': MultinomialNB()
        }
        
        # Enhanced vectorizer with character n-grams
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),  # Word 1-3 grams
            analyzer='word'
        )
        
        # Character n-gram vectorizer for abbreviations
        self.char_vectorizer = TfidfVectorizer(
            max_features=2000,
            analyzer='char',
            ngram_range=(3, 5)  # Character 3-5 grams for abbreviations like "sob", "abg"
        )
        
        self.best_model = None
        self.best_threshold = 0.5
        self.performance_metrics = {}
        
    def extract_keywords_from_text(self, text: str) -> Dict:
        """Extract all medical keywords from text with counts"""
        text_lower = text.lower()
        
        cardio_keywords = [kw for kw in self.cardiovascular_keywords if kw in text_lower]
        resp_keywords = [kw for kw in self.respiratory_keywords if kw in text_lower]
        other_keywords = [kw for kw in self.other_medical_keywords if kw in text_lower]
        
        return {
            'cardio_count': len(cardio_keywords),
            'resp_count': len(resp_keywords),
            'other_count': len(other_keywords),
            'cardio_keywords': cardio_keywords,
            'resp_keywords': resp_keywords,
            'other_keywords': other_keywords
        }
    
    def classify_case(self, text: str) -> Dict:
        """Classify a case as cardio-respiratory only or mixed/other"""
        keyword_info = self.extract_keywords_from_text(text)
        
        cardio_count = keyword_info['cardio_count']
        resp_count = keyword_info['resp_count']
        other_count = keyword_info['other_count']
        
        # More flexible classification logic
        has_cardio_resp = cardio_count > 0 or resp_count > 0
        
        # If case has cardio/resp keywords and minimal other medical terms, classify as cardio-resp
        if has_cardio_resp and (other_count == 0 or (cardio_count + resp_count) > other_count * 2):
            classification = 'cardio_respiratory_only'
            confidence = min(1.0, (cardio_count + resp_count) / max(1, other_count + 1))
        else:
            classification = 'mixed_or_other'
            confidence = min(1.0, other_count / max(1, cardio_count + resp_count + 1))
        
        return {
            'classification': classification,
            'confidence': confidence,
            'cardio_keywords': cardio_count,
            'resp_keywords': resp_count,
            'other_keywords': other_count,
            'total_keywords': cardio_count + resp_count + other_count
        }
    
    def process_cases(self, data_directory: str) -> Tuple[List[str], List[int], List[Dict]]:
        """Process all cases and return texts, labels, and metadata"""
        logger.info("🔄 Processing medical cases for classification...")
        
        cases_path = Path(data_directory)
        if not cases_path.exists():
            logger.error(f"❌ Data directory not found: {data_directory}")
            return [], [], []
        
        case_files = sorted(list(cases_path.glob("*.txt")))
        logger.info(f"📁 Found {len(case_files)} medical cases")
        
        texts = []
        labels = []
        metadata = []
        
        for case_file in case_files:
            try:
                with open(case_file, 'r', encoding='utf-8') as f:
                    case_text = f.read()
                
                # Classify the case
                classification_result = self.classify_case(case_text)
                label = 1 if classification_result['classification'] == 'cardio_respiratory_only' else 0
                
                texts.append(case_text)
                labels.append(label)
                metadata.append({
                    'case_id': case_file.stem,
                    'file_path': str(case_file),
                    **classification_result
                })
                
            except Exception as e:
                logger.error(f"❌ Failed to process {case_file}: {e}")
        
        logger.info(f"✅ Processed {len(texts)} cases")
        logger.info(f"📊 Cardio-respiratory cases: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
        
        return texts, labels, metadata
    
    def find_optimal_threshold(self, y_true, y_proba, min_precision=0.94, min_recall=0.90):
        """Find optimal threshold that satisfies precision and recall constraints"""
        thresholds = np.arange(0.05, 0.96, 0.01)
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if len(np.unique(y_pred)) < 2:
                continue
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Check if constraints are satisfied
            if recall >= min_recall and precision >= min_precision:
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        
        # If no threshold satisfies both constraints, prioritize recall
        if best_f1 == 0:
            logger.warning(f"⚠️ Could not find threshold satisfying both constraints. Optimizing for recall >= {min_recall}")
            for threshold in thresholds:
                y_pred = (y_proba >= threshold).astype(int)
                
                if len(np.unique(y_pred)) < 2:
                    continue
                
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                if recall >= min_recall and f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        
        return best_threshold
    
    def train_and_evaluate(self, texts: List[str], labels: List[int], output_dir: str = "cardio_resp_results"):
        """Train models with enhanced features and evaluate"""
        logger.info("🤖 Training enhanced models with class imbalance handling...")
        
        # Convert to numpy arrays
        X_text = np.array(texts)
        y = np.array(labels)
        
        # Stratified train-test split
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            X_text, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"📊 Train: {len(y_train)} cases ({sum(y_train)} positive)")
        logger.info(f"📊 Test: {len(y_test)} cases ({sum(y_test)} positive)")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = {}
        
        for model_name, base_model in self.models.items():
            logger.info(f"🔄 Training {model_name} with enhanced features...")
            
            try:
                # Build pipeline with SMOTE if enabled
                if self.use_smote and model_name != 'naive_bayes':
                    # SMOTE only works with certain models
                    pipeline = ImbPipeline([
                        ('vectorizer', self.vectorizer),
                        ('smote', SMOTE(random_state=42, k_neighbors=min(5, sum(y_train)-1))),
                        ('classifier', base_model)
                    ])
                else:
                    pipeline = Pipeline([
                        ('vectorizer', self.vectorizer),
                        ('classifier', base_model)
                    ])
                
                # Train model
                start_time = time.time()
                pipeline.fit(X_train_text, y_train)
                training_time = time.time() - start_time
                
                # Get probability predictions
                if hasattr(pipeline, 'predict_proba'):
                    y_train_proba = pipeline.predict_proba(X_train_text)[:, 1]
                    y_test_proba = pipeline.predict_proba(X_test_text)[:, 1]
                    
                    # Find optimal threshold
                    optimal_threshold = self.find_optimal_threshold(
                        y_train, y_train_proba,
                        min_precision=self.target_precision,
                        min_recall=self.target_recall
                    )
                    
                    # Apply optimal threshold
                    y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
                else:
                    y_test_pred = pipeline.predict(X_test_text)
                    optimal_threshold = 0.5
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_test_pred)
                precision = precision_score(y_test, y_test_pred, zero_division=0)
                recall = recall_score(y_test, y_test_pred, zero_division=0)
                f1 = f1_score(y_test, y_test_pred, zero_division=0)
                cm = confusion_matrix(y_test, y_test_pred)
                
                results[model_name] = {
                    'pipeline': pipeline,
                    'threshold': optimal_threshold,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'training_time': training_time,
                    'confusion_matrix': cm.tolist(),
                    'classification_report': classification_report(
                        y_test, y_test_pred,
                        target_names=['Mixed/Other', 'Cardio-Respiratory Only']
                    )
                }
                
                logger.info(f"✅ {model_name}: Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}, Threshold={optimal_threshold:.3f}")
                
            except Exception as e:
                logger.error(f"❌ Failed to train {model_name}: {e}")
                continue
        
        # Select best model based on F1 score
        if results:
            best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
            self.best_model = results[best_model_name]['pipeline']
            self.best_threshold = results[best_model_name]['threshold']
            
            logger.info(f"🏆 Best model: {best_model_name} (F1={results[best_model_name]['f1_score']:.3f}, Recall={results[best_model_name]['recall']:.3f})")
        
        # Save results
        self.save_results(results, output_path)
        self.plot_results(results, y_test, output_path)
        
        return results

    def save_results(self, results: Dict, output_path: Path):
        """Save model results to JSON"""
        # Convert non-serializable objects
        serializable_results = {}
        for model_name, metrics in results.items():
            serializable_results[model_name] = {
                'threshold': float(metrics['threshold']),
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1_score']),
                'training_time': float(metrics['training_time']),
                'confusion_matrix': metrics['confusion_matrix'],
                'classification_report': metrics['classification_report']
            }
        
        with open(output_path / "enhanced_performance_report.json", 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"💾 Results saved to {output_path}")
    
    def plot_results(self, results: Dict, y_test, output_path: Path):
        """Create visualization plots"""
        # Plot confusion matrices
        n_models = len(results)
        fig, axes = plt.subplots(1, min(n_models, 4), figsize=(5*min(n_models, 4), 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, metrics) in enumerate(list(results.items())[:4]):
            cm = np.array(metrics['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f"{model_name}\nRecall={metrics['recall']:.3f}")
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(output_path / "enhanced_confusion_matrices.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Plots saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced Cardio-Respiratory Classifier')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing medical case files')
    parser.add_argument('--output-dir', type=str, default='cardio_resp_results', help='Output directory for results')
    parser.add_argument('--use-smote', action='store_true', default=True, help='Use SMOTE for oversampling')
    parser.add_argument('--no-smote', dest='use_smote', action='store_false', help='Disable SMOTE')
    parser.add_argument('--target-recall', type=float, default=0.90, help='Target recall threshold')
    parser.add_argument('--target-precision', type=float, default=0.94, help='Target precision threshold')
    
    args = parser.parse_args()
    
    logger.info("🏥 Starting Enhanced Cardio-Respiratory Classifier")
    logger.info(f"⚙️ Target: Recall >= {args.target_recall}, Precision >= {args.target_precision}")
    logger.info(f"⚙️ SMOTE: {'Enabled' if args.use_smote else 'Disabled'}")
    
    # Initialize classifier
    classifier = EnhancedCardioRespiratoryClassifier(
        use_smote=args.use_smote,
        target_recall=args.target_recall,
        target_precision=args.target_precision
    )
    
    # Process cases
    texts, labels, metadata = classifier.process_cases(args.data_dir)
    
    if not texts:
        logger.error("❌ No cases processed successfully")
        return
    
    # Train and evaluate
    results = classifier.train_and_evaluate(texts, labels, args.output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 ENHANCED CLASSIFIER RESULTS")
    print("="*80)
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f} {'✅' if metrics['recall'] >= args.target_recall else '❌'}")
        print(f"  F1-Score:  {metrics['f1_score']:.3f}")
        print(f"  Threshold: {metrics['threshold']:.3f}")
    print("="*80)


if __name__ == "__main__":
    main()
