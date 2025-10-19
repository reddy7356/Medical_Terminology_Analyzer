#!/usr/bin/env python3
"""
Cardiovascular and Respiratory Case Classifier
Separates cases that contain only cardiovascular and respiratory keywords
Uses scikit-learn for ML evaluation and performance comparison
"""

import os
import re
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict
import logging
from datetime import datetime

# ML and evaluation imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CardioRespiratoryClassifier:
    """Classifier for cardiovascular and respiratory medical cases"""
    
    def __init__(self):
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
            'tuberculosis', 'tb', 'pneumocystis', 'pneumonia', 'influenza', 'flu',
            'respiratory failure', 'ards', 'ventilation', 'ventilator', 'oxygen', 'o2',
            'saturation', 'spo2', 'hypoxia', 'hypercapnia', 'dyspnea', 'shortness of breath',
            'wheezing', 'cough', 'sputum', 'hemoptysis', 'chest x-ray', 'ct scan',
            'pulmonary function', 'spirometry', 'bronchoscopy', 'intubation', 'extubation',
            'tracheostomy', 'ventilator', 'cpap', 'bipap', 'nebulizer', 'inhaler',
            'albuterol', 'steroid', 'prednisone', 'antibiotic', 'ventolin', 'proventil'
        }
        
        # Other medical categories to exclude
        self.other_medical_keywords = {
            # Neurological
            'brain', 'neurological', 'stroke', 'seizure', 'headache', 'migraine',
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
            
            # Infectious
            'infection', 'infectious', 'sepsis', 'bacterial', 'viral', 'fungal',
            'antibiotic', 'antiviral', 'immunity', 'immune', 'vaccine',
            
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
            'clotting', 'hematology', 'leukemia', 'lymphoma', 'transfusion',
            
            # Dermatology
            'skin', 'dermatology', 'rash', 'lesion', 'wound', 'cutaneous',
            'dermatitis', 'eczema', 'psoriasis', 'melanoma', 'basal cell',
            
            # Ophthalmology
            'eye', 'ocular', 'vision', 'retinal', 'ophthalmology', 'glaucoma',
            'cataract', 'macular', 'cornea', 'iris', 'pupil'
        }
        
        # Initialize ML models
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'svm': SVC(random_state=42, kernel='linear'),
            'naive_bayes': MultinomialNB()
        }
        
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.best_model = None
        self.performance_metrics = {}
        
    def extract_keywords_from_text(self, text: str) -> Set[str]:
        """Extract all medical keywords from text"""
        text_lower = text.lower()
        found_keywords = set()
        
        # Check cardiovascular keywords
        for keyword in self.cardiovascular_keywords:
            if keyword in text_lower:
                found_keywords.add(keyword)
        
        # Check respiratory keywords
        for keyword in self.respiratory_keywords:
            if keyword in text_lower:
                found_keywords.add(keyword)
        
        # Check other medical keywords
        for keyword in self.other_medical_keywords:
            if keyword in text_lower:
                found_keywords.add(keyword)
        
        return found_keywords
    
    def classify_case(self, text: str) -> Dict:
        """Classify a case as cardio-respiratory only or mixed/other"""
        keywords = self.extract_keywords_from_text(text)
        
        # Count keywords by category
        cardio_count = len([k for k in keywords if k in self.cardiovascular_keywords])
        resp_count = len([k for k in keywords if k in self.respiratory_keywords])
        other_count = len([k for k in keywords if k in self.other_medical_keywords])
        
        # More flexible classification logic
        has_cardio_resp = cardio_count > 0 or resp_count > 0
        has_other_medical = other_count > 0
        
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
            'total_keywords': len(keywords),
            'found_keywords': list(keywords)
        }
    
    def process_cases(self, data_directory: str) -> Tuple[List[Dict], List[str]]:
        """Process all cases and classify them"""
        logger.info("🔄 Processing medical cases for classification...")
        
        cases_path = Path(data_directory)
        if not cases_path.exists():
            logger.error(f"❌ Data directory not found: {data_directory}")
            return [], []
        
        case_files = list(cases_path.glob("*.txt"))
        logger.info(f"📁 Found {len(case_files)} medical cases")
        
        results = []
        case_texts = []
        cardio_resp_count = 0
        
        start_time = time.time()
        
        for i, case_file in enumerate(case_files):
            try:
                with open(case_file, 'r', encoding='utf-8') as f:
                    case_text = f.read()
                
                # Classify the case
                classification_result = self.classify_case(case_text)
                
                if classification_result['classification'] == 'cardio_respiratory_only':
                    cardio_resp_count += 1
                
                result = {
                    'case_id': case_file.stem,
                    'file_path': str(case_file),
                    'text_length': len(case_text),
                    'word_count': len(case_text.split()),
                    **classification_result
                }
                
                results.append(result)
                case_texts.append(case_text)
                
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"📊 Processed {i + 1}/{len(case_files)} cases in {elapsed:.2f}s (Cardio-Resp: {cardio_resp_count})")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {case_file}: {e}")
        
        total_time = time.time() - start_time
        logger.info(f"✅ Processed {len(results)} cases in {total_time:.2f}s ({total_time/len(results):.3f}s per case)")
        logger.info(f"📊 Found {cardio_resp_count} cardio-respiratory only cases ({cardio_resp_count/len(results)*100:.1f}%)")
        
        return results, case_texts
    
    def create_ml_dataset(self, results: List[Dict], case_texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Create ML dataset for evaluation"""
        # Create labels (1 for cardio-respiratory only, 0 for mixed/other)
        labels = np.array([1 if r['classification'] == 'cardio_respiratory_only' else 0 for r in results])
        
        # Vectorize text data
        X = self.vectorizer.fit_transform(case_texts).toarray()
        
        return X, labels
    
    def train_and_evaluate_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train and evaluate multiple ML models"""
        logger.info("🤖 Training and evaluating ML models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        model_results = {}
        
        for name, model in self.models.items():
            logger.info(f"🔄 Training {name}...")
            
            start_time = time.time()
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            
            training_time = time.time() - start_time
            
            model_results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, target_names=['Mixed/Other', 'Cardio-Respiratory Only'])
            }
            
            logger.info(f"✅ {name}: Accuracy={accuracy:.3f}, F1={f1:.3f}, Time={training_time:.2f}s")
        
        # Select best model based on F1 score
        best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['f1_score'])
        self.best_model = model_results[best_model_name]['model']
        
        logger.info(f"🏆 Best model: {best_model_name} (F1={model_results[best_model_name]['f1_score']:.3f})")
        
        return model_results
    
    def generate_performance_report(self, results: List[Dict], model_results: Dict) -> Dict:
        """Generate comprehensive performance report"""
        # Basic statistics
        total_cases = len(results)
        cardio_resp_only = len([r for r in results if r['classification'] == 'cardio_respiratory_only'])
        mixed_other = total_cases - cardio_resp_only
        
        # Calculate processing efficiency
        total_processing_time = sum([r.get('processing_time', 0) for r in results])
        avg_processing_time = total_processing_time / total_cases if total_processing_time > 0 else 0
        
        # Human vs Machine comparison (estimated)
        # Assuming human takes 2-5 minutes per case for detailed review
        human_time_per_case = 3 * 60  # 3 minutes in seconds
        total_human_time = total_cases * human_time_per_case
        
        # Machine efficiency
        machine_efficiency = (total_human_time / (total_processing_time + 1)) if total_processing_time > 0 else float('inf')
        
        report = {
            'dataset_info': {
                'total_cases': total_cases,
                'cardio_respiratory_only': cardio_resp_only,
                'mixed_or_other': mixed_other,
                'cardio_resp_percentage': (cardio_resp_only / total_cases) * 100
            },
            'processing_efficiency': {
                'total_processing_time': total_processing_time,
                'avg_time_per_case': avg_processing_time,
                'cases_per_second': 1 / avg_processing_time if avg_processing_time > 0 else 0,
                'estimated_human_time': total_human_time,
                'machine_efficiency_multiplier': machine_efficiency
            },
            'ml_model_performance': model_results,
            'cardio_respiratory_cases': [
                r for r in results if r['classification'] == 'cardio_respiratory_only'
            ]
        }
        
        return report
    
    def save_results(self, results: List[Dict], report: Dict, output_dir: str = "cardio_resp_results"):
        """Save results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save detailed results
        with open(output_path / "classification_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save performance report
        with open(output_path / "performance_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save cardio-respiratory only cases
        cardio_resp_cases = [r for r in results if r['classification'] == 'cardio_respiratory_only']
        with open(output_path / "cardio_respiratory_cases.json", 'w') as f:
            json.dump(cardio_resp_cases, f, indent=2, default=str)
        
        # Create summary CSV
        df = pd.DataFrame(results)
        df.to_csv(output_path / "classification_summary.csv", index=False)
        
        logger.info(f"💾 Results saved to {output_path}")
        
        return output_path

def main():
    """Main function to run the cardio-respiratory classifier"""
    logger.info("🏥 Starting Cardiovascular and Respiratory Case Classifier")
    
    # Initialize classifier
    classifier = CardioRespiratoryClassifier()
    
    # Process cases
    data_directory = "/Users/saiofocalallc/Medical_Terminology_Analyzer/data"
    results, case_texts = classifier.process_cases(data_directory)
    
    if not results:
        logger.error("❌ No cases processed successfully")
        return
    
    # Create ML dataset
    X, y = classifier.create_ml_dataset(results, case_texts)
    
    # Train and evaluate models
    model_results = classifier.train_and_evaluate_models(X, y)
    
    # Generate performance report
    report = classifier.generate_performance_report(results, model_results)
    
    # Save results
    output_path = classifier.save_results(results, report)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 CARDIOVASCULAR AND RESPIRATORY CASE CLASSIFICATION RESULTS")
    print("="*80)
    print(f"Total Cases Processed: {report['dataset_info']['total_cases']}")
    print(f"Cardio-Respiratory Only: {report['dataset_info']['cardio_respiratory_only']} ({report['dataset_info']['cardio_resp_percentage']:.1f}%)")
    print(f"Mixed/Other Cases: {report['dataset_info']['mixed_or_other']}")
    print(f"Average Processing Time: {report['processing_efficiency']['avg_time_per_case']:.3f}s per case")
    print(f"Machine Efficiency: {report['processing_efficiency']['machine_efficiency_multiplier']:.1f}x faster than human")
    
    print("\n🤖 ML MODEL PERFORMANCE:")
    for model_name, metrics in report['ml_model_performance'].items():
        print(f"{model_name}: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}")
    
    print(f"\n💾 Results saved to: {output_path}")
    print("="*80)

if __name__ == "__main__":
    main()
