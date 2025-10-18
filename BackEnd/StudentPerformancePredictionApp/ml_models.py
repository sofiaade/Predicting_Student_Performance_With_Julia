import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

# Configuration
CONFIG = {
    'at_risk_threshold': 65,
    'decision_tree_depth': 5,
    'logistic_max_iter': 1000,
    'test_size': 0.2,
    'random_state': 42,
    'top_at_risk_limit': 20,
    'cv_folds': 5
}

def validate_data(df: pd.DataFrame, model_type: str) -> Dict[str, Any]:
    """Validate that required columns exist"""
    required_cols = ['student_id']
    
    if model_type in ['logistic', 'decision_tree']:
        required_cols.append('at_risk')
    
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        return {
            "valid": False,
            "message": f"Missing required columns: {', '.join(missing)}"
        }
    
    if len(df) < 10:
        return {
            "valid": False,
            "message": "Dataset too small. Need at least 10 rows."
        }
    
    return {"valid": True}

def create_visualizations(y_true, y_pred, model_type: str) -> Dict[str, str]:
    """Create visualization plots with proper memory management"""
    plots = {}
    
    if model_type in ['logistic', 'decision_tree']:
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        try:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Not At Risk', 'At Risk'],
                       yticklabels=['Not At Risk', 'At Risk'], ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            plots['confusion_matrix'] = base64.b64encode(buf.read()).decode('utf-8')
        finally:
            plt.close(fig)
            buf.close()
        
        # Distribution plot
        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            actual_counts = pd.Series(y_true).value_counts()
            pred_counts = pd.Series(y_pred).value_counts()
            
            x = np.arange(2)
            width = 0.35
            
            ax.bar(x - width/2, [actual_counts.get(False, 0), actual_counts.get(True, 0)], 
                   width, label='Actual', color='steelblue')
            ax.bar(x + width/2, [pred_counts.get(False, 0), pred_counts.get(True, 0)], 
                   width, label='Predicted', color='coral')
            
            ax.set_xlabel('Student Status')
            ax.set_ylabel('Count')
            ax.set_title('Actual vs Predicted Distribution')
            ax.set_xticks(x)
            ax.set_xticklabels(['Not At Risk', 'At Risk'])
            ax.legend()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            plots['distribution'] = base64.b64encode(buf.read()).decode('utf-8')
        finally:
            plt.close(fig)
            buf.close()
        
    else:  # linear regression
        # Scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            ax.scatter(y_true, y_pred, alpha=0.5, color='steelblue')
            
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title('Predicted vs Actual Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            plots['scatter'] = base64.b64encode(buf.read()).decode('utf-8')
        finally:
            plt.close(fig)
            buf.close()
    
    return plots

def calculate_risk_level(attendance: float, test_score: float, homework: float) -> str:
    """Calculate risk level based on multiple factors"""
    risk_score = 0
    
    if attendance < 0.7:
        risk_score += 2
    elif attendance < 0.8:
        risk_score += 1
    
    if test_score < 60:
        risk_score += 2
    elif test_score < 70:
        risk_score += 1
    
    if homework < 0.6:
        risk_score += 2
    elif homework < 0.75:
        risk_score += 1
    
    if risk_score >= 4:
        return "High Risk"
    elif risk_score >= 2:
        return "Medium Risk"
    else:
        return "Low Risk"

def train_and_predict(csv_path: str, model_type: str) -> Dict[str, Any]:
    """Main function to train model and return results"""
    try:
        # Load data
        df = pd.read_csv(csv_path)
        df = df.dropna()
        
        # Validate data
        validation = validate_data(df, model_type)
        if not validation["valid"]:
            return {
                "status": "error",
                "message": validation["message"]
            }
        
        # Prepare features
        exclude_cols = ['at_risk', 'student_id']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if len(feature_cols) == 0:
            return {
                "status": "error",
                "message": "No feature columns found in dataset"
            }
        
        X = df[feature_cols]
        y = df['at_risk']
        
        # Train-test split for proper evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=CONFIG['test_size'], 
            random_state=CONFIG['random_state'],
            stratify=y if model_type in ['logistic', 'decision_tree'] else None
        )
        
        # Train model
        if model_type == 'logistic':
            model = LogisticRegression(
                max_iter=CONFIG['logistic_max_iter'], 
                random_state=CONFIG['random_state']
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]
            
            # Cross-validation for robustness
            cv_scores = cross_val_score(model, X, y, cv=CONFIG['cv_folds'], scoring='accuracy')
            
            metrics = {
                "accuracy": float(accuracy_score(y_test, predictions) * 100),
                "precision": float(precision_score(y_test, predictions, zero_division=0) * 100),
                "recall": float(recall_score(y_test, predictions, zero_division=0) * 100),
                "f1_score": float(f1_score(y_test, predictions, zero_division=0) * 100),
                "cv_mean": float(cv_scores.mean() * 100),
                "cv_std": float(cv_scores.std() * 100)
            }
            
            # Get predictions for all data with probabilities
            all_predictions = model.predict(X)
            all_probabilities = model.predict_proba(X)[:, 1]
            
        elif model_type == 'decision_tree':
            model = DecisionTreeClassifier(
                max_depth=CONFIG['decision_tree_depth'], 
                random_state=CONFIG['random_state']
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=CONFIG['cv_folds'], scoring='accuracy')
            
            # Feature importance
            feature_importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
            feature_importance = {k: float(v) for k, v in 
                                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)}
            
            metrics = {
                "accuracy": float(accuracy_score(y_test, predictions) * 100),
                "precision": float(precision_score(y_test, predictions, zero_division=0) * 100),
                "recall": float(recall_score(y_test, predictions, zero_division=0) * 100),
                "f1_score": float(f1_score(y_test, predictions, zero_division=0) * 100),
                "cv_mean": float(cv_scores.mean() * 100),
                "cv_std": float(cv_scores.std() * 100),
                "feature_importance": feature_importance
            }
            
            # Get predictions for all data
            all_predictions = model.predict(X)
            all_probabilities = None
            
        elif model_type == 'linear':
            if 'test_score' in df.columns:
                y_continuous = df['test_score']
            else:
                y_continuous = df[feature_cols[0]]
            
            X_train, X_test, y_train_cont, y_test_cont = train_test_split(
                X, y_continuous, 
                test_size=CONFIG['test_size'], 
                random_state=CONFIG['random_state']
            )
            
            model = LinearRegression()
            model.fit(X_train, y_train_cont)
            predictions = model.predict(X_test)
            
            mse = mean_squared_error(y_test_cont, predictions)
            mae = np.mean(np.abs(y_test_cont - predictions))
            r2 = r2_score(y_test_cont, predictions)
            
            metrics = {
                "mse": float(mse),
                "rmse": float(np.sqrt(mse)),
                "mae": float(mae),
                "r2_score": float(r2 * 100)
            }
            
            # Get predictions for all data
            all_predictions = model.predict(X)
            all_probabilities = None
            y_test = y_test_cont
        
        # Create visualizations
        plots = create_visualizations(y_test, predictions, model_type)
        
        # Identify at-risk students
        if model_type in ['logistic', 'decision_tree']:
            at_risk_mask = all_predictions == True
        else:
            at_risk_mask = all_predictions < CONFIG['at_risk_threshold']
        
        at_risk_df = df[at_risk_mask].copy()
        at_risk_students = []
        
        for idx, row in at_risk_df.iterrows():
            attendance = pd.to_numeric(row.get('attendance_rate', 0), errors='coerce') or 0
            test_score = pd.to_numeric(row.get('test_score', 0), errors='coerce') or 0
            homework = pd.to_numeric(row.get('homework_completion', 0), errors='coerce') or 0

            
            # Calculate granular risk level
            if model_type == 'logistic' and all_probabilities is not None:
                prob = all_probabilities[idx]
                if prob > 0.7:
                    risk_level = "High Risk"
                elif prob > 0.4:
                    risk_level = "Medium Risk"
                else:
                    risk_level = "Low Risk"
            else:
                risk_level = calculate_risk_level(attendance, test_score, homework)
            
            student = {
                
                "student_id": str(row.get("student_id", idx)),
                "risk_level": risk_level,
                "attendance": f"{int(attendance * 100)}%" if isinstance(attendance, (int, float)) else "N/A",
                "test_score": str(int(test_score)) if isinstance(test_score, (int, float)) else "N/A",
                "homework": f"{int(homework * 100)}%" if isinstance(homework, (int, float)) else "N/A",
                "action": "Needs Immediate Attention" if risk_level == "High Risk" else "Monitor Progress"
            }
            at_risk_students.append(student)
        
        # Sort by risk level
        risk_order = {"High Risk": 0, "Medium Risk": 1, "Low Risk": 2}
        at_risk_students.sort(key=lambda x: risk_order.get(x["risk_level"], 3))
        
        result = {
            "status": "success",
            "metrics": metrics,
            "at_risk_count": len(at_risk_students),
            "at_risk_students": at_risk_students[:CONFIG['top_at_risk_limit']],
            "plots": plots,
            "total_students": len(df),
            "train_size": len(X_train),
            "test_size": len(X_test)
        }
        
        return result
        
    except KeyError as e:
        return {
            "status": "error",
            "message": f"Missing required column: {str(e)}"
        }
    except ValueError as e:
        return {
            "status": "error",
            "message": f"Invalid data format: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Training failed: {type(e).__name__} - {str(e)}"
        }

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({
            "status": "error", 
            "message": "Usage: python ml_models.py <csv_path> <model_type>"
        }))
        sys.exit(1)
    
    csv_path = sys.argv[1]
    model_type = sys.argv[2]
    
    if model_type not in ['logistic', 'decision_tree', 'linear']:
        print(json.dumps({
            "status": "error",
            "message": f"Invalid model type: {model_type}. Must be 'logistic', 'decision_tree', or 'linear'"
        }))
        sys.exit(1)
    
    result = train_and_predict(csv_path, model_type)
    print(json.dumps(result))