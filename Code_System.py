import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, learning_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, roc_curve, auc, precision_recall_curve, 
                             classification_report, roc_auc_score, average_precision_score,
                             f1_score, accuracy_score, precision_score, recall_score)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from scipy.stats import randint, uniform
import warnings
import joblib
import os
import io
import base64
from datetime import datetime
import ast
import hashlib
import sqlite3
import json
import threading
from pathlib import Path

# Remove all Chinese font settings and use default English fonts
plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 14, 'xtick.labelsize': 13, 
    'ytick.labelsize': 13, 'legend.fontsize': 12, 'figure.figsize': (10, 8),
    'figure.titlesize': 16, 'axes.titlesize': 16, 'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Machine learning Mineralization Potential discrimination System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get current directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# ===============================
# USER ISOLATION SYSTEM
# ===============================

def generate_user_id():
    """Generate a unique user ID based on browser session"""
    # Use Streamlit's internal session ID and timestamp
    import time
    session_id = st.runtime.scriptrunner.add_script_run_ctx().streamlit_script_run_ctx.session_id
    timestamp = str(time.time())
    unique_str = f"{session_id}_{timestamp}"
    
    # Create hash for user ID
    user_id = hashlib.md5(unique_str.encode()).hexdigest()[:12]
    
    # Store in session state
    if 'user_id' not in st.session_state:
        st.session_state.user_id = user_id
    
    return st.session_state.user_id

def get_user_workspace():
    """Get user-specific workspace directory"""
    user_id = generate_user_id()
    user_dir = Path(f"user_data/{user_id}")
    user_dir.mkdir(parents=True, exist_ok=True)
    return str(user_dir), user_id

def get_user_db_path():
    """Get user-specific database path"""
    workspace, user_id = get_user_workspace()
    return os.path.join(workspace, f"user_{user_id}.db")

def init_user_database():
    """Initialize user-specific SQLite database"""
    db_path = get_user_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        model_path TEXT NOT NULL,
        parameters TEXT,
        accuracy REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        data_type TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_hash TEXT,
        rows INTEGER,
        columns INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        input_file TEXT,
        predictions TEXT,
        metrics TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()

def save_user_model(model_name, model_dict, accuracy=None):
    """Save model info to user database"""
    workspace, user_id = get_user_workspace()
    db_path = get_user_db_path()
    
    # Save model file
    model_filename = f"{model_name.replace(' ', '_')}_model.pkl"
    model_path = os.path.join(workspace, model_filename)
    joblib.dump(model_dict, model_path)
    
    # Save to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    parameters = json.dumps(model_dict.get('best_params', {}))
    
    cursor.execute('''
    INSERT INTO user_models (model_name, model_path, parameters, accuracy)
    VALUES (?, ?, ?, ?)
    ''', (model_name, model_path, parameters, accuracy))
    
    conn.commit()
    conn.close()
    
    return model_path

def load_user_models():
    """Load models from user database"""
    workspace, user_id = get_user_workspace()
    db_path = get_user_db_path()
    
    if not os.path.exists(db_path):
        return {}
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('SELECT model_name, model_path FROM user_models ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    
    models = {}
    for model_name, model_path in rows:
        if os.path.exists(model_path):
            try:
                model_dict = joblib.load(model_path)
                models[model_name] = model_dict
            except Exception as e:
                st.sidebar.warning(f"⚠️ {model_name}: {e}")
    
    return models

def save_user_scaler(scaler):
    """Save scaler for current user"""
    workspace, user_id = get_user_workspace()
    scaler_path = os.path.join(workspace, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    return scaler_path

def load_user_scaler():
    """Load scaler for current user"""
    workspace, user_id = get_user_workspace()
    scaler_path = os.path.join(workspace, 'scaler.pkl')
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None

def save_user_best_model(model):
    """Save best model for current user"""
    workspace, user_id = get_user_workspace()
    model_path = os.path.join(workspace, 'best_model.pkl')
    joblib.dump(model, model_path)
    return model_path

def load_user_best_model():
    """Load best model for current user"""
    workspace, user_id = get_user_workspace()
    model_path = os.path.join(workspace, 'best_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def clear_user_data():
    """Clear all data for current user"""
    workspace, user_id = get_user_workspace()
    
    # Clear database
    db_path = get_user_db_path()
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Clear workspace directory
    import shutil
    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    
    # Reinitialize
    get_user_workspace()
    init_user_database()
    
    return True

# ===============================
# END USER ISOLATION SYSTEM
# ===============================

# Initialize session state
def initialize_session_state():
    # Initialize user isolation
    workspace, user_id = get_user_workspace()
    init_user_database()
    
    # User-specific session state
    user_key_prefix = f"user_{user_id}_"
    
    if f'{user_key_prefix}trained_models' not in st.session_state:
        st.session_state[f'{user_key_prefix}trained_models'] = {}
    if f'{user_key_prefix}data_loaded' not in st.session_state:
        st.session_state[f'{user_key_prefix}data_loaded'] = False
    if f'{user_key_prefix}X_train' not in st.session_state:
        st.session_state[f'{user_key_prefix}X_train'] = None
    if f'{user_key_prefix}X_test' not in st.session_state:
        st.session_state[f'{user_key_prefix}X_test'] = None
    if f'{user_key_prefix}y_train' not in st.session_state:
        st.session_state[f'{user_key_prefix}y_train'] = None
    if f'{user_key_prefix}y_test' not in st.session_state:
        st.session_state[f'{user_key_prefix}y_test'] = None
    if f'{user_key_prefix}feature_names' not in st.session_state:
        st.session_state[f'{user_key_prefix}feature_names'] = None
    if f'{user_key_prefix}current_page' not in st.session_state:
        st.session_state[f'{user_key_prefix}current_page'] = "Home"
    if f'{user_key_prefix}uploaded_file' not in st.session_state:
        st.session_state[f'{user_key_prefix}uploaded_file'] = None
    if f'{user_key_prefix}custom_params' not in st.session_state:
        st.session_state[f'{user_key_prefix}custom_params'] = {}
    if f'{user_key_prefix}training_complete' not in st.session_state:
        st.session_state[f'{user_key_prefix}training_complete'] = False
    if f'{user_key_prefix}evaluation_results' not in st.session_state:
        st.session_state[f'{user_key_prefix}evaluation_results'] = None
    if f'{user_key_prefix}data_preprocessed' not in st.session_state:
        st.session_state[f'{user_key_prefix}data_preprocessed'] = False
    if f'{user_key_prefix}models_trained' not in st.session_state:
        st.session_state[f'{user_key_prefix}models_trained'] = False
    if f'{user_key_prefix}evaluation_done' not in st.session_state:
        st.session_state[f'{user_key_prefix}evaluation_done'] = False
    
    # Global settings (shared but user can customize)
    if 'cv_folds' not in st.session_state:
        st.session_state.cv_folds = 5
    if 'test_size' not in st.session_state:
        st.session_state.test_size = 0.3
    if 'random_state' not in st.session_state:
        st.session_state.random_state = 42

def get_user_session(key):
    """Get user-specific session state value"""
    workspace, user_id = get_user_workspace()
    user_key = f"user_{user_id}_{key}"
    return st.session_state.get(user_key)

def set_user_session(key, value):
    """Set user-specific session state value"""
    workspace, user_id = get_user_workspace()
    user_key = f"user_{user_id}_{key}"
    st.session_state[user_key] = value

# Helper functions
def get_table_download_link(df, filename, link_text):
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def get_image_download_link(fig, filename, link_text):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">{link_text}</a>'
    return href

def create_evaluation_report(metrics_df, rank_df, models, feature_names, cv_folds):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_content = []
    
    report_content.append(f"Zircon Mineralization potential Model Evaluation Report")
    report_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"Cross-validation folds: {cv_folds}")
    report_content.append("="*50)
    report_content.append("")
    
    report_content.append("Model Performance Metrics Summary:")
    report_content.append("")
    for model_name in metrics_df.index:
        report_content.append(f"{model_name}:")
        report_content.append(f"  Accuracy: {metrics_df.loc[model_name, 'Accuracy']:.4f}")
        report_content.append(f"  Precision: {metrics_df.loc[model_name, 'Precision']:.4f}")
        report_content.append(f"  Recall: {metrics_df.loc[model_name, 'Recall']:.4f}")
        report_content.append(f"  F1 Score: {metrics_df.loc[model_name, 'F1']:.4f}")
        report_content.append(f"  ROC AUC: {metrics_df.loc[model_name, 'ROC_AUC']:.4f}")
        report_content.append(f"  PR AUC: {metrics_df.loc[model_name, 'PR_AUC']:.4f}")
        report_content.append("")
    
    report_content.append("Model Performance Ranking (1=Best):")
    for model_name in rank_df.index:
        report_content.append(f"{model_name}: Average Rank {rank_df.loc[model_name, 'Average_Rank']:.2f}")
    
    best_model = rank_df['Average_Rank'].idxmin()
    report_content.append("")
    report_content.append(f"Best Model: {best_model}")
    report_content.append(f"Best Parameters: {models[best_model]['best_params']}")
    
    return "\n".join(report_content), timestamp

# Helper function to parse hidden layer sizes
def parse_hidden_layer_sizes(layer_str):
    """Parse hidden layer sizes from string input like '100,50,25' to tuple (100, 50, 25)"""
    try:
        if not layer_str.strip():
            return (100,)  # default value
        layers = tuple(int(x.strip()) for x in layer_str.split(','))
        return layers
    except Exception as e:
        st.error(f"Error parsing hidden layer sizes: {e}. Using default (100,)")
        return (100,)

# Data loading and preprocessing
def load_and_preprocess_data(uploaded_file, test_size=0.3, random_state=42):
    try:
        if uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        else:
            data = pd.read_csv(uploaded_file)
        
        if data.isnull().sum().any():
            data = data.dropna()
        
        X = data.iloc[:, :-1]
        feature_names = X.columns.tolist()
        y = data['Label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Save user-specific scaler
        save_user_scaler(scaler)
        
        return X_train, X_test, y_train, y_test, feature_names, data.shape
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return None, None, None, None, None, None

# Model training
def train_models(X_train, y_train, selected_models, search_method, cv_folds, custom_params=None):
    base_models = {
        'XGBoost': {
            'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'random_params': {
                'n_estimators': randint(50, 300),
                'max_depth': randint(3, 10),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4)
            },
            'grid_params': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'random_params': {
                'n_estimators': randint(50, 300),
                'max_depth': randint(3, 20),
                'min_samples_split': randint(2, 10),
                'min_samples_leaf': randint(1, 5)
            },
            'grid_params': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5]
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'random_params': {
                'C': uniform(0.1, 10),
                'gamma': uniform(0.01, 1),
                'kernel': ['linear', 'rbf', 'poly']
            },
            'grid_params': {
                'C': [0.1, 1, 10, 100],
                'gamma': [0.01, 0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly']
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'random_params': {
                'C': uniform(0.1, 10),
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']
            },
            'grid_params': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']
            }
        },
        'Neural Network': {
            'model': MLPClassifier(max_iter=1000, random_state=42),
            'random_params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 30), (100, 50), (100, 50, 25)],
                'alpha': uniform(0.0001, 0.1),
                'activation': ['relu', 'tanh', 'logistic'],
                'learning_rate_init': uniform(0.001, 0.01),
                'solver': ['lbfgs', 'sgd', 'adam'],
                'batch_size': ['auto', 32, 64, 128]
            },
            'grid_params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 30), (100, 50), (100, 50, 25)],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'activation': ['relu', 'tanh', 'logistic'],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'solver': ['lbfgs', 'sgd', 'adam'],
                'batch_size': ['auto', 32, 64, 128]
            }
        }
    }
    
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'f1': 'f1_macro',
        'roc_auc': 'roc_auc_ovo'
    }
    
    models = {name: config for name, config in base_models.items() if name in selected_models}
    trained_models = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, config) in enumerate(models.items()):
        status_text.text(f"Training {name}... ({cv_folds}-fold CV)")
        
        if custom_params and name in custom_params:
            params = custom_params[name]
            
            # Handle hidden layer sizes parsing for Neural Network
            if name == 'Neural Network' and 'hidden_layer_sizes' in params:
                if isinstance(params['hidden_layer_sizes'], str):
                    params['hidden_layer_sizes'] = parse_hidden_layer_sizes(params['hidden_layer_sizes'])
            
            model = config['model']
            model.set_params(**params)
            model.fit(X_train, y_train)
            
            # Calculate accuracy
            y_train_pred = model.predict(X_train)
            accuracy = accuracy_score(y_train, y_train_pred)
            
            model_dict = {
                'model': model,
                'best_params': params,
                'cv_metrics': {},
                'train_metrics': {},
                'cv_folds': cv_folds
            }
            
            # Save to user database
            save_user_model(name, model_dict, accuracy)
            
        else:
            if search_method == "Random Search":
                search = RandomizedSearchCV(
                    config['model'], 
                    config['random_params'], 
                    n_iter=20,
                    cv=cv,
                    scoring=scoring_metrics,
                    refit='f1',
                    n_jobs=-1,
                    random_state=42,
                    return_train_score=True
                )
            else:
                search = GridSearchCV(
                    config['model'], 
                    config['grid_params'], 
                    cv=cv,
                    scoring=scoring_metrics,
                    refit='f1',
                    n_jobs=-1,
                    return_train_score=True
                )
            
            search.fit(X_train, y_train)
            
            best_index = search.best_index_
            cv_metrics = {}
            for metric in scoring_metrics.keys():
                mean_key = f'mean_test_{metric}'
                std_key = f'std_test_{metric}'
                if mean_key in search.cv_results_ and std_key in search.cv_results_:
                    cv_metrics[metric] = {
                        'mean': search.cv_results_[mean_key][best_index],
                        'std': search.cv_results_[std_key][best_index]
                    }
            
            model = search.best_estimator_
            y_train_pred = model.predict(X_train)
            train_metrics = {
                'accuracy': accuracy_score(y_train, y_train_pred),
                'precision': precision_score(y_train, y_train_pred, average='weighted'),
                'recall': recall_score(y_train, y_train_pred, average='weighted'),
                'f1': f1_score(y_train, y_train_pred, average='weighted')
            }
            
            accuracy = train_metrics['accuracy']
            
            model_dict = {
                'model': model,
                'best_params': search.best_params_,
                'cv_metrics': cv_metrics,
                'train_metrics': train_metrics,
                'cv_folds': cv_folds
            }
            
            # Save to user database
            save_user_model(name, model_dict, accuracy)
        
        trained_models[name] = model_dict
        progress_bar.progress((i + 1) / len(models))
    
    status_text.text("Training completed!")
    set_user_session('training_complete', True)
    set_user_session('models_trained', True)
    return trained_models

# Plot confusion matrices
def plot_confusion_matrices(models, X_train, X_test, y_train, y_test):
    figures = {}
    for model_name, model_dict in models.items():
        model = model_dict['model']
        
        y_train_pred = model.predict(X_train)
        cm_train = confusion_matrix(y_train, y_train_pred)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'],
                    annot_kws={"size": 14},
                    cbar=False, ax=ax1)
        ax1.set_title(f'{model_name} - Training Set', fontsize=14)
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        
        y_test_pred = model.predict(X_test)
        cm_test = confusion_matrix(y_test, y_test_pred)
        
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'],
                    annot_kws={"size": 14},
                    cbar=False, ax=ax2)
        ax2.set_title(f'{model_name} - Test Set', fontsize=14)
        ax2.set_ylabel('True Label', fontsize=12)
        ax2.set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        figures[model_name] = fig
        st.pyplot(fig)
        plt.close()
    
    return figures

# Plot ROC curves
def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Random')
    
    for model_name, model_dict in models.items():
        model = model_dict['model']
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close()
    return fig

# Plot PR curves
def plot_pr_curves(models, X_test, y_test):
    plt.figure(figsize=(10, 8))
    
    random_precision = sum(y_test) / len(y_test)
    plt.plot([0, 1], [random_precision, random_precision], 'k--', alpha=0.7, label='Random')
    
    for model_name, model_dict in models.items():
        model = model_dict['model']
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            avg_precision = average_precision_score(y_test, y_prob)
            plt.plot(recall, precision, lw=2, label=f'{model_name} (AP = {avg_precision:.3f})')
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close()
    return fig

# Plot feature importance
def plot_feature_importance(models, feature_names):
    figures = {}
    for model_name, model_dict in models.items():
        model = model_dict['model']
        
        if model_name == 'Logistic Regression':
            continue
            
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title(f"{model_name} - Feature Importance", fontsize=16)
            bars = plt.bar(range(len(importances)), importances[indices], align="center")
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90, fontsize=12)
            plt.xlim([-1, len(importances)])
            plt.ylabel("Importance Score", fontsize=12)
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            fig = plt.gcf()
            figures[model_name] = fig
            st.pyplot(fig)
            plt.close()
            
        elif hasattr(model, 'coef_'):
            coef = model.coef_[0]
            indices = np.argsort(np.abs(coef))[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title(f"{model_name} - Feature Coefficients", fontsize=16)
            bars = plt.bar(range(len(coef)), np.abs(coef)[indices], align="center", color='salmon')
            plt.xticks(range(len(coef)), [feature_names[i] for i in indices], rotation=90, fontsize=12)
            plt.xlim([-1, len(coef)])
            plt.ylabel("Coefficient Value", fontsize=12)
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            fig = plt.gcf()
            figures[model_name] = fig
            st.pyplot(fig)
            plt.close()
    
    return figures

# Performance ranking
def calculate_model_ranks(metrics_df):
    rank_df = metrics_df.copy()
    
    for column in rank_df.columns:
        rank_df[column] = rank_df[column].rank(ascending=False)
    
    rank_df['Average_Rank'] = rank_df.mean(axis=1)
    rank_df = rank_df.sort_values(by='Average_Rank')
    
    return rank_df

# Evaluation function
def evaluate_and_visualize(models, X_train, X_test, y_train, y_test, feature_names, cv_folds):
    metrics = []
    
    st.subheader("Model Parameters and Performance Summary")
    
    for model_name, model_dict in models.items():
        with st.expander(f"{model_name} Details"):
            st.write(f"**Best Parameters:** {model_dict['best_params']}")
            st.write(f"**CV Folds:** {model_dict.get('cv_folds', cv_folds)}")
            
            if 'cv_metrics' in model_dict and model_dict['cv_metrics']:
                st.write("**Cross-validation Metrics:**")
                cv_data = []
                for metric, value in model_dict['cv_metrics'].items():
                    cv_data.append({
                        'Metric': metric,
                        'Mean': f"{value['mean']:.4f}",
                        'Std': f"± {value['std']:.4f}"
                    })
                st.table(pd.DataFrame(cv_data))
            
            if 'train_metrics' in model_dict and model_dict['train_metrics']:
                st.write("**Training Metrics:**")
                train_data = []
                for metric, value in model_dict['train_metrics'].items():
                    train_data.append({
                        'Metric': metric,
                        'Value': f"{value:.4f}"
                    })
                st.table(pd.DataFrame(train_data))
    
    st.subheader("Confusion Matrices")
    confusion_figures = plot_confusion_matrices(models, X_train, X_test, y_train, y_test)
    
    st.subheader("Test Set Performance")
    for model_name, model_dict in models.items():
        model = model_dict['model']
        
        y_test_pred = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
        
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        if hasattr(model, "predict_proba") or hasattr(model, "decision_function"):
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            roc_auc = 0.5
        
        try:
            avg_precision = average_precision_score(y_test, y_prob)
        except:
            avg_precision = 0.0
        
        metrics.append({
            'Model': model_name,
            'Accuracy': test_accuracy,
            'Precision': test_precision,
            'Recall': test_recall,
            'F1': test_f1,
            'ROC_AUC': roc_auc,
            'PR_AUC': avg_precision
        })
        
        with st.expander(f"{model_name} Classification Report"):
            st.text(classification_report(y_test, y_test_pred, target_names=['Class 0', 'Class 1']))
    
    metrics_df = pd.DataFrame(metrics).set_index('Model')
    rank_df = calculate_model_ranks(metrics_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Performance Metrics:**")
        st.dataframe(metrics_df.style.format("{:.4f}"))
    
    with col2:
        st.write("**Model Ranking (1=Best):**")
        st.dataframe(rank_df.style.format("{:.2f}"))
    
    best_model_name = rank_df['Average_Rank'].idxmin()
    best_model = models[best_model_name]['model']
    st.success(f"**Best Model: {best_model_name}** (Avg Rank {rank_df.loc[best_model_name, 'Average_Rank']:.2f})")
    
    # Save best model for user
    save_user_best_model(best_model)
    
    st.subheader("ROC Curves")
    roc_fig = plot_roc_curves(models, X_test, y_test)
    
    st.subheader("Precision-Recall Curves")
    pr_fig = plot_pr_curves(models, X_test, y_test)
    
    st.subheader("Feature Importance")
    feature_figures = plot_feature_importance(models, feature_names)
    
    evaluation_results = {
        'metrics_df': metrics_df,
        'rank_df': rank_df,
        'confusion_figures': confusion_figures,
        'roc_fig': roc_fig,
        'pr_fig': pr_fig,
        'feature_figures': feature_figures,
        'best_model': best_model_name,
        'models': models,
        'cv_folds': cv_folds
    }
    
    set_user_session('evaluation_results', evaluation_results)
    set_user_session('evaluation_done', True)
    
    return metrics_df, rank_df, evaluation_results

# Prediction function
def predict_new_dataset(models, uploaded_file, selected_models_for_prediction):
    st.subheader("New Dataset Prediction")
    
    try:
        if uploaded_file.name.endswith('.xlsx'):
            new_data = pd.read_excel(uploaded_file)
        else:
            new_data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"File error: {e}")
        return
    
    if new_data.isnull().sum().any():
        st.warning("Missing values found, removing rows...")
        new_data = new_data.dropna()
    
    if 'Label' in new_data.columns:
        X_new = new_data.drop('Label', axis=1)
        y_new = new_data['Label']
        has_labels = True
        st.info("Label column found.")
    else:
        X_new = new_data
        has_labels = False
        st.info("No Label column, prediction only.")
    
    try:
        scaler = load_user_scaler()
        if scaler is None:
            st.error("Scaler not found, train models first.")
            return
        X_new_scaled = scaler.transform(X_new)
        st.success("Data standardized.")
    except Exception as e:
        st.error(f"Error loading scaler: {str(e)}")
        return
    
    for model_name in selected_models_for_prediction:
        if model_name not in models:
            st.warning(f"{model_name} not trained, skipping.")
            continue
            
        st.subheader(f"{model_name} Results")
        model_dict = models[model_name]
        model = model_dict['model']
        
        try:
            y_pred = model.predict(X_new_scaled)
            
            confidence = np.ones(len(y_pred))
            if hasattr(model, "predict_proba"):
                confidence = np.max(model.predict_proba(X_new_scaled), axis=1)
            elif hasattr(model, "decision_function"):
                decision_values = model.decision_function(X_new_scaled)
                confidence = 1 / (1 + np.exp(-decision_values))
            
            prediction_df = pd.DataFrame({
                'Predicted_Label': y_pred,
                'Prediction_Confidence': confidence
            })
            
            prediction_df = pd.concat([X_new.reset_index(drop=True), prediction_df], axis=1)
            
            workspace, user_id = get_user_workspace()
            prediction_filename = f"{workspace}/{model_name.replace(' ', '_')}_predictions.csv"
            prediction_df.to_csv(prediction_filename, index=False)
            
            st.write(f"**Prediction Sample:**")
            st.dataframe(prediction_df.head())
            
            csv = prediction_df.to_csv(index=False)
            st.download_button(
                label=f"Download {model_name} Results",
                data=csv,
                file_name=f"{model_name.replace(' ', '_')}_predictions.csv",
                mime="text/csv"
            )
            
            if has_labels:
                accuracy = accuracy_score(y_new, y_pred)
                precision = precision_score(y_new, y_pred, average='weighted')
                recall = recall_score(y_new, y_pred, average='weighted')
                f1 = f1_score(y_new, y_pred, average='weighted')
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{accuracy:.4f}")
                col2.metric("Precision", f"{precision:.4f}")
                col3.metric("Recall", f"{recall:.4f}")
                col4.metric("F1 Score", f"{f1:.4f}")
                
                cm = confusion_matrix(y_new, y_pred)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Class 0', 'Class 1'],
                            yticklabels=['Class 0', 'Class 1'],
                            annot_kws={"size": 14}, ax=ax)
                ax.set_title(f'{model_name} - New Data', fontsize=14)
                ax.set_ylabel('True Label', fontsize=12)
                ax.set_xlabel('Predicted Label', fontsize=12)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                with st.expander(f"{model_name} Classification Report"):
                    st.text(classification_report(y_new, y_pred, target_names=['Class 0', 'Class 1']))
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.countplot(x='Predicted_Label', data=prediction_df, ax=ax)
                ax.set_title(f'{model_name} - Prediction Distribution', fontsize=16)
                ax.set_xlabel('Predicted Label', fontsize=12)
                ax.set_ylabel('Count', fontsize=12) 
                ax.set_xticklabels(['Class 0', 'Class 1'])
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
        except Exception as e:
            st.error(f"Error with {model_name}: {str(e)}")

# Load saved models for current user
def load_user_saved_models():
    """Load models for current user from database"""
    workspace, user_id = get_user_workspace()
    
    # Get models from database
    db_path = get_user_db_path()
    if not os.path.exists(db_path):
        return {}
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT model_name, model_path, accuracy 
    FROM user_models 
    ORDER BY created_at DESC
    ''')
    
    rows = cursor.fetchall()
    conn.close()
    
    saved_models = {}
    for model_name, model_path, accuracy in rows:
        if os.path.exists(model_path):
            try:
                model_dict = joblib.load(model_path)
                saved_models[model_name] = model_dict
                # Show status in sidebar
                accuracy_str = f"{accuracy:.2%}" if accuracy else "N/A"
                st.sidebar.success(f"✅ {model_name} ({accuracy_str})")
            except Exception as e:
                st.sidebar.warning(f"⚠️ {model_name}: {e}")
        else:
            st.sidebar.info(f"📝 {model_name}")
    
    return saved_models

# Download results
def download_evaluation_results(evaluation_results, feature_names):
    if not evaluation_results:
        st.warning("No results to download.")
        return
    
    st.subheader("📥 Download Results")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report_content, report_timestamp = create_evaluation_report(
        evaluation_results['metrics_df'],
        evaluation_results['rank_df'],
        evaluation_results['models'],
        feature_names,
        evaluation_results['cv_folds']
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Tables")
        st.markdown(get_table_download_link(
            evaluation_results['metrics_df'], 
            f"metrics_{timestamp}.csv", 
            "📊 Metrics Table"
        ), unsafe_allow_html=True)
        
        st.markdown(get_table_download_link(
            evaluation_results['rank_df'], 
            f"ranks_{timestamp}.csv", 
            "🏆 Ranking Table"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Charts")
        
        if 'roc_fig' in evaluation_results:
            st.markdown(get_image_download_link(
                evaluation_results['roc_fig'],
                f"roc_{timestamp}.png",
                "📈 ROC Curves"
            ), unsafe_allow_html=True)
        
        if 'pr_fig' in evaluation_results:
            st.markdown(get_image_download_link(
                evaluation_results['pr_fig'],
                f"pr_{timestamp}.png",
                "📊 PR Curves"
            ), unsafe_allow_html=True)
    
    with col3:
        st.markdown("### Model Charts")
        
        if 'confusion_figures' in evaluation_results:
            for model_name, fig in evaluation_results['confusion_figures'].items():
                st.markdown(get_image_download_link(
                    fig,
                    f"confusion_{model_name}_{timestamp}.png",
                    f"🎯 {model_name} Confusion"
                ), unsafe_allow_html=True)
        
        if 'feature_figures' in evaluation_results:
            for model_name, fig in evaluation_results['feature_figures'].items():
                st.markdown(get_image_download_link(
                    fig,
                    f"features_{model_name}_{timestamp}.png",
                    f"🔍 {model_name} Features"
                ), unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Full Report")
    st.download_button(
        label="📄 Full Report (TXT)",
        data=report_content,
        file_name=f"report_{timestamp}.txt",
        mime="text/plain"
    )
    
    with st.expander("Report Preview"):
        st.text(report_content)

# Main application
def main():
    initialize_session_state()
    
    # Display user ID in sidebar for debugging
    workspace, user_id = get_user_workspace()
    st.sidebar.markdown("---")
    st.sidebar.subheader("👤 User Info")
    st.sidebar.info(f"User ID: {user_id[:8]}...")
    
    st.title("🔬 Machine learning Mineralization Potential discrimination System")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("🚀 Navigation")
    
    if st.sidebar.button("🏠 Home", use_container_width=True):
        set_user_session('current_page', "Home")
    
    if st.sidebar.button("📊 Data Upload", use_container_width=True):
        set_user_session('current_page', "Data Upload")
    
    if st.sidebar.button("🤖 Model Training", use_container_width=True):
        set_user_session('current_page', "Model Training")
    
    if st.sidebar.button("📈 Model Evaluation", use_container_width=True):
        set_user_session('current_page', "Model Evaluation")
    
    if st.sidebar.button("🔮 Predict New Data", use_container_width=True):
        set_user_session('current_page', "Predict New Data")
    
    if st.sidebar.button("⚙️ Parameters", use_container_width=True):
        set_user_session('current_page', "Parameter Settings")
    
    st.sidebar.markdown("---")
    
    # Cross-validation
    st.sidebar.subheader("🔧 Cross-validation")
    cv_folds = st.sidebar.radio(
        "CV Folds",
        [5, 10],
        index=0 if st.session_state.cv_folds == 5 else 1,
        key="cv_folds_sidebar"
    )
    st.session_state.cv_folds = cv_folds
    st.sidebar.info(f"Current: {cv_folds}-fold")
    
    st.sidebar.markdown("---")
    
    # Load user models
    st.sidebar.subheader("📁 Your Models")
    saved_models = load_user_saved_models()
    if saved_models:
        set_user_session('trained_models', saved_models)
        st.sidebar.success(f"Loaded: {len(saved_models)} models")
    
    # User-specific clear data
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear My Data", type="secondary", help="Clear all data for current user only"):
        if clear_user_data():
            # Clear session state
            for key in ['trained_models', 'data_loaded', 'X_train', 'X_test', 
                       'y_train', 'y_test', 'feature_names', 'uploaded_file',
                       'training_complete', 'evaluation_results', 'data_preprocessed',
                       'models_trained', 'evaluation_done']:
                set_user_session(key, None if key != 'trained_models' else {})
            
            set_user_session('current_page', "Home")
            st.sidebar.success("Your data cleared!")
            st.rerun()
    
    # Status
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Your Status")
    
    status_col1, status_col2 = st.sidebar.columns(2)
    with status_col1:
        data_status = "✅" if get_user_session('data_loaded') else "❌"
        st.metric("Data", data_status)
    with status_col2:
        trained_count = len(get_user_session('trained_models') or {})
        st.metric("Models", f"{trained_count}/5")
    
    # Page content
    current_page = get_user_session('current_page') or "Home"
    
    # Home page
    if current_page == "Home":
        st.header("Machine learning Mineralization Potential discrimination System")
        st.markdown("""
        ### System Features
        
        **📊 Data Upload** - Upload and preprocess zircon data
        **🤖 Model Training** - Train machine learning models
        **📈 Model Evaluation** - Evaluate and visualize model performance
        **🔮 Predict New Data** - Predict on new datasets
        **⚙️ Parameters** - Customize model parameters
        
        ### 🛡️ User Isolation System
        - Each user has **completely isolated workspace**
        - Your data, models, and results are **private to you**
        - No interference between different users
        - Automatic cleanup when you clear your data
        
        ### Usage Steps
        1. Upload data in "Data Upload"
        2. Train models in "Model Training" 
        3. Evaluate in "Model Evaluation"
        4. Predict in "Predict New Data"
        
        ### Tips
        - Results are preserved between pages
        - Use "Clear My Data" to restart
        - Models are auto-saved in your private workspace
        - Results can be downloaded
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            status = "✅ Loaded" if get_user_session('data_loaded') else "❌ Not Loaded"
            st.metric("Your Data Status", status)
        with col2:
            trained_count = len(get_user_session('trained_models') or {})
            st.metric("Your Models Trained", f"{trained_count}/5")
        with col3:
            status = "✅ Done" if get_user_session('training_complete') else "⏳ Waiting"
            st.metric("Training Status", status)
        
        st.info(f"Cross-validation: **{st.session_state.cv_folds}-fold**")
        st.info(f"Test set size: **{st.session_state.test_size*100}%**")
        st.info(f"👤 Your User ID: **{user_id[:8]}...**")
    
    # Data Upload page
    elif current_page == "Data Upload":
        st.header("📊 Data Upload")
        st.markdown("Upload your data (Excel/CSV)")
        
        if get_user_session('data_loaded'):
            st.success("✅ Your data loaded")
            st.write(f"- Train: {get_user_session('X_train').shape[0]}")
            st.write(f"- Test: {get_user_session('X_test').shape[0]}")
            st.write(f"- Features: {len(get_user_session('feature_names'))}")
            st.info(f"Test set size: **{st.session_state.test_size*100}%**")
        
        # Data split settings
        st.subheader("Data Split Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider(
                "Test Set Size (%)", 
                min_value=10, 
                max_value=50, 
                value=int(st.session_state.test_size * 100),
                step=5,
                help="Percentage of data to use for testing"
            )
            st.session_state.test_size = test_size / 100
        
        with col2:
            random_state = st.number_input(
                "Random State", 
                min_value=0, 
                max_value=100, 
                value=st.session_state.random_state,
                step=1,
                help="Random seed for reproducible splits"
            )
            st.session_state.random_state = random_state
        
        st.info(f"Training set: **{(1-st.session_state.test_size)*100}%** | Test set: **{st.session_state.test_size*100}%**")
        
        uploaded_file = st.file_uploader("Select Your File", type=['xlsx', 'csv'])
        
        if uploaded_file is not None:
            set_user_session('uploaded_file', uploaded_file)
            
            try:
                if uploaded_file.name.endswith('.xlsx'):
                    data = pd.read_excel(uploaded_file)
                else:
                    data = pd.read_csv(uploaded_file)
                
                st.write("**Data Preview:**")
                st.dataframe(data.head())
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Data Info:**")
                    st.write(f"- Shape: {data.shape}")
                    st.write(f"- Features: {data.shape[1]-1}")
                    st.write(f"- Samples: {data.shape[0]}")
                
                with col2:
                    if 'Label' in data.columns:
                        st.write("**Label Count:**")
                        label_counts = data['Label'].value_counts()
                        st.write(label_counts)
                
                if 'Label' in data.columns:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    label_counts = data['Label'].value_counts()
                    label_counts.plot(kind='bar', ax=ax)
                    ax.set_title('Label Distribution', fontsize=16)
                    ax.set_xlabel('Label', fontsize=12)
                    ax.set_ylabel('Count', fontsize=12)
                    ax.set_xticklabels(['Class 0', 'Class 1'], rotation=0)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                if st.button("Preprocess Your Data", type="primary"):
                    with st.spinner("Processing your data..."):
                        X_train, X_test, y_train, y_test, feature_names, data_shape = load_and_preprocess_data(
                            uploaded_file, 
                            test_size=st.session_state.test_size,
                            random_state=st.session_state.random_state
                        )
                        
                        if X_train is not None:
                            set_user_session('X_train', X_train)
                            set_user_session('X_test', X_test)
                            set_user_session('y_train', y_train)
                            set_user_session('y_test', y_test)
                            set_user_session('feature_names', feature_names)
                            set_user_session('data_loaded', True)
                            set_user_session('data_preprocessed', True)
                            
                            st.success("Your data processed!")
                            st.write(f"- Training set: {X_train.shape[0]} samples ({(1-st.session_state.test_size)*100:.1f}%)")
                            st.write(f"- Test set: {X_test.shape[0]} samples ({st.session_state.test_size*100:.1f}%)")
                            st.write(f"- Features: {len(feature_names)}")
                            
                            # Show label distribution in train/test sets
                            train_label_counts = pd.Series(y_train).value_counts()
                            test_label_counts = pd.Series(y_test).value_counts()
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Training Set Labels:**")
                                st.write(train_label_counts)
                            with col2:
                                st.write("**Test Set Labels:**")
                                st.write(test_label_counts)
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Model Training page
    elif current_page == "Model Training":
        st.header("🤖 Model Training")
        
        st.info(f"CV: **{st.session_state.cv_folds}-fold**")
        st.info(f"Test set size: **{st.session_state.test_size*100}%**")
        
        if not get_user_session('data_loaded'):
            st.warning("Upload your data first!")
            return
        
        st.write("**Select models:**")
        
        model_options = ['XGBoost', 'Random Forest', 'SVM', 'Logistic Regression', 'Neural Network']
        selected_models = st.multiselect(
            "Models",
            model_options,
            default=model_options
        )
        
        search_method = st.radio(
            "Search Method",
            ["Random Search", "Grid Search", "Custom Parameters"]
        )
        
        user_trained_models = get_user_session('trained_models') or {}
        if user_trained_models:
            st.info(f"📁 Your loaded models: {len(user_trained_models)}")
            trained_list = list(user_trained_models.keys())
            st.write(f"Your trained models: {', '.join(trained_list)}")
        
        col1, col2 = st.columns(2)
        with col1:
            use_existing = st.checkbox("Use your existing models", value=True)
        with col2:
            retrain = st.checkbox("Retrain selected", value=False)
        
        if st.button("Train Your Models", type="primary"):
            if not selected_models:
                st.error("Select at least one model!")
                return
            
            with st.spinner("Training your models..."):
                models_to_train = selected_models
                if use_existing and not retrain:
                    existing_models = list(user_trained_models.keys())
                    models_to_train = [model for model in selected_models if model not in existing_models]
                    
                    if not models_to_train:
                        st.info("Using your existing models.")
                    else:
                        st.info(f"New models to train: {', '.join(models_to_train)}")
                
                if models_to_train or retrain:
                    custom_params = get_user_session('custom_params') or {}
                    if search_method == "Custom Parameters":
                        custom_params = custom_params  # Already loaded from user session
                    
                    trained_models = train_models(
                        get_user_session('X_train'), 
                        get_user_session('y_train'), 
                        models_to_train if not retrain else selected_models, 
                        search_method,
                        st.session_state.cv_folds,
                        custom_params
                    )
                    
                    # Update user's trained models
                    current_models = get_user_session('trained_models') or {}
                    current_models.update(trained_models)
                    set_user_session('trained_models', current_models)
                    st.success(f"Trained {len(trained_models)} models for you")
                else:
                    st.success("Using your existing models")
    
    # Model Evaluation page
    elif current_page == "Model Evaluation":
        st.header("📈 Model Evaluation")
        
        st.info(f"CV: **{st.session_state.cv_folds}-fold**")
        st.info(f"Test set size: **{st.session_state.test_size*100}%**")
        
        user_trained_models = get_user_session('trained_models') or {}
        if not user_trained_models:
            st.warning("Train your models first!")
            return
        
        if get_user_session('evaluation_done'):
            st.success("✅ Your evaluation done")
            st.write("Your previous results:")
            
            evaluate_and_visualize(
                user_trained_models,
                get_user_session('X_train'),
                get_user_session('X_test'),
                get_user_session('y_train'),
                get_user_session('y_test'),
                get_user_session('feature_names'),
                st.session_state.cv_folds
            )
            
            user_eval_results = get_user_session('evaluation_results')
            if user_eval_results:
                download_evaluation_results(
                    user_eval_results,
                    get_user_session('feature_names')
                )
        else:
            if st.button("Evaluate Your Models", type="primary"):
                with st.spinner("Evaluating your models..."):
                    metrics, rank_df, evaluation_results = evaluate_and_visualize(
                        user_trained_models,
                        get_user_session('X_train'),
                        get_user_session('X_test'),
                        get_user_session('y_train'),
                        get_user_session('y_test'),
                        get_user_session('feature_names'),
                        st.session_state.cv_folds
                    )
                    
                    set_user_session('evaluation_results', evaluation_results)
                    st.success("Your evaluation done!")
            
            user_eval_results = get_user_session('evaluation_results')
            if user_eval_results:
                download_evaluation_results(
                    user_eval_results,
                    get_user_session('feature_names')
                )
    
    # Predict New Data page
    elif current_page == "Predict New Data":
        st.header("🔮 Predict New Data")
        
        user_trained_models = get_user_session('trained_models') or {}
        if not user_trained_models:
            st.warning("Train your models first!")
            return
        
        st.write("Upload new data for prediction")
        new_data_file = st.file_uploader("Select Your File", type=['xlsx', 'csv'], key="new_data")
        
        if new_data_file is not None:
            trained_model_names = list(user_trained_models.keys())
            selected_models_for_prediction = st.multiselect(
                "Select Your Models",
                trained_model_names,
                default=trained_model_names
            )
            
            if st.button("Predict with Your Models", type="primary"):
                if not selected_models_for_prediction:
                    st.error("Select at least one model!")
                    return
                
                with st.spinner("Predicting with your models..."):
                    predict_new_dataset(
                        user_trained_models,
                        new_data_file,
                        selected_models_for_prediction
                    )
    
    # Parameter Settings page
    elif current_page == "Parameter Settings":
        st.header("⚙️ Your Parameters")
        
        st.info("Set your custom parameters")
        
        model_options = ['XGBoost', 'Random Forest', 'SVM', 'Logistic Regression', 'Neural Network']
        
        custom_params = get_user_session('custom_params') or {}
        
        for model in model_options:
            with st.expander(f"{model}"):
                if model == 'XGBoost':
                    n_estimators = st.slider("n_estimators", 50, 300, 100, key=f"xgb_n_est_{user_id}")
                    max_depth = st.slider("max_depth", 3, 10, 6, key=f"xgb_depth_{user_id}")
                    learning_rate = st.slider("learning_rate", 0.01, 0.3, 0.1, key=f"xgb_lr_{user_id}")
                    subsample = st.slider("subsample", 0.6, 1.0, 0.8, key=f"xgb_sub_{user_id}")
                    colsample_bytree = st.slider("colsample_bytree", 0.6, 1.0, 0.8, key=f"xgb_col_{user_id}")
                    
                    custom_params[model] = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'learning_rate': learning_rate,
                        'subsample': subsample,
                        'colsample_bytree': colsample_bytree
                    }
                
                elif model == 'Random Forest':
                    n_estimators = st.slider("n_estimators", 50, 300, 100, key=f"rf_n_est_{user_id}")
                    max_depth = st.slider("max_depth", 3, 20, 10, key=f"rf_depth_{user_id}")
                    min_samples_split = st.slider("min_samples_split", 2, 10, 2, key=f"rf_split_{user_id}")
                    min_samples_leaf = st.slider("min_samples_leaf", 1, 5, 1, key=f"rf_leaf_{user_id}")
                    
                    custom_params[model] = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf
                    }
                
                elif model == 'SVM':
                    C = st.slider("C", 0.1, 10.0, 1.0, key=f"svm_c_{user_id}")
                    gamma = st.slider("gamma", 0.01, 1.0, 0.1, key=f"svm_gamma_{user_id}")
                    kernel = st.selectbox("kernel", ['linear', 'rbf', 'poly'], key=f"svm_kernel_{user_id}")
                    
                    custom_params[model] = {
                        'C': C,
                        'gamma': gamma,
                        'kernel': kernel
                    }
                
                elif model == 'Logistic Regression':
                    C = st.slider("C", 0.1, 10.0, 1.0, key=f"lr_c_{user_id}")
                    penalty = st.selectbox("penalty", ['l1', 'l2'], key=f"lr_penalty_{user_id}")
                    solver = st.selectbox("solver", 
                                         ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
                                         index=0,
                                         key=f"lr_solver_{user_id}")
                    
                    custom_params[model] = {
                        'C': C,
                        'penalty': penalty,
                        'solver': solver
                    }
                
                elif model == 'Neural Network':
                    st.write("**Hidden Layer Sizes**")
                    st.info("Enter comma-separated values (e.g., '100,50,25' for 3 layers with 100, 50, and 25 neurons)")
                    
                    hidden_layer_input = st.text_input(
                        "Hidden Layer Sizes",
                        value="100,50",
                        key=f"nn_layers_input_{user_id}",
                        help="Format: comma-separated integers (e.g., '100,50' for two layers)"
                    )
                    
                    # Parse the input
                    hidden_layer_sizes = parse_hidden_layer_sizes(hidden_layer_input)
                    
                    st.write(f"Parsed layer structure: {hidden_layer_sizes}")
                    
                    alpha = st.slider("alpha", 0.0001, 0.1, 0.001, key=f"nn_alpha_{user_id}")
                    activation = st.selectbox("activation", ['relu', 'tanh', 'logistic'], key=f"nn_act_{user_id}")
                    learning_rate_init = st.slider("learning_rate_init", 0.001, 0.01, 0.001, key=f"nn_lr_{user_id}")
                    solver = st.selectbox("solver", ['lbfgs', 'sgd', 'adam'], key=f"nn_solver_{user_id}")
                    batch_size = st.selectbox("batch_size", ['auto', 32, 64, 128], key=f"nn_batch_{user_id}")
                    
                    custom_params[model] = {
                        'hidden_layer_sizes': hidden_layer_sizes,
                        'alpha': alpha,
                        'activation': activation,
                        'learning_rate_init': learning_rate_init,
                        'solver': solver,
                        'batch_size': batch_size
                    }
        
        set_user_session('custom_params', custom_params)
        
        if st.button("Save Your Parameters", type="primary"):
            st.success("Your parameters saved!")

if __name__ == "__main__":
    main()