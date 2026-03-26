import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib # For saving the final production model
import logging # Industry standard for tracking execution

# 1. Setup Logging (Enterprise standard for monitoring)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FraudDetectionPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.model = None
        self.scaler = StandardScaler()

    def load_data(self):
        """Phase 1: Ingesting data from source"""
        logging.info("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        return self.df

    def preprocess_data(self):
        """Phase 2 & 3: Cleaning and Scaling"""
        logging.info("Preprocessing and scaling features...")
        # Scaling Amount and Time
        self.df['scaled_amount'] = self.scaler.fit_transform(self.df['Amount'].values.reshape(-1, 1))
        self.df['scaled_time'] = self.scaler.fit_transform(self.df['Time'].values.reshape(-1, 1))
        
        # Drop original unscaled columns
        self.df.drop(['Time', 'Amount'], axis=1, inplace=True)
        
        # Prepare Features and Target
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']
        
        # Stratified Split (Ensures test set reflects real-world imbalance)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    def handle_imbalance(self):
        """Phase 4: Synthetic Oversampling (SMOTE)"""
        logging.info("Applying SMOTE to balance training data...")
        sm = SMOTE(random_state=42)
        self.X_train, self.y_train = sm.fit_resample(self.X_train, self.y_train)

    def train_model(self):
        """Phase 5 & 7: Training Optimized Random Forest"""
        logging.info("Training production-grade Random Forest model...")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        """Phase 6: Performance Validation"""
        logging.info("Evaluating model on unseen test data...")
        predictions = self.model.predict(self.X_test)
        print("\n--- FINAL PRODUCTION EVALUATION ---")
        print(classification_report(self.y_test, predictions))
        
    def save_artifacts(self):
        """Enterprise Step: Saving the model for deployment"""
        logging.info("Saving model and scaler for production deployment...")
        joblib.dump(self.model, 'fraud_model_v1.pkl')
        joblib.dump(self.scaler, 'scaler_v1.pkl')
        logging.info("Artifacts saved successfully.")

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # Create the pipeline instance
    pipeline = FraudDetectionPipeline('creditcard.csv')
    
    # Execute the end-to-end flow
    pipeline.load_data()
    pipeline.preprocess_data()
    pipeline.handle_imbalance()
    pipeline.train_model()
    pipeline.evaluate()
    pipeline.save_artifacts()
    
    logging.info("Pipeline Execution Complete.")