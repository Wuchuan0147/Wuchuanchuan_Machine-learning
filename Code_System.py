import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.font_manager as fm
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
import time
import warnings
import joblib
import os
import io
import base64
from datetime import datetime

# Remove Chinese font settings and use default fonts
plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 14, 'xtick.labelsize': 13, 
    'ytick.labelsize': 13, 'legend.fontsize': 12, 'figure.figsize': (10, 8),
    'figure.titlesize': 16, 'axes.titlesize': 16, 'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Zircon Mineralization Prediction System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get current directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Initialize session state
def initialize_session_state():
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'custom_params' not in st.session_state:
        st.session_state.custom_params = {}
    if 'training_complete' not in st.session_state:
        st.session_state.training_complete = False
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    if 'cv_folds' not in st.session_state:
        st.session_state.cv_folds = 5  # Default 5-fold cross-validation
    if 'data_preprocessed' not in st.session_state:
        st.session_state.data_preprocessed = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'evaluation_done' not in st.session_state:
        st.session_state.evaluation_done = False

# Helper functions: Create download links
def get_table_download_link(df, filename, link_text):
    """Generate table download link"""
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def get_image_download_link(fig, filename, link_text):
    """Generate image download link"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">{link_text}</a>'
    return href

def create_evaluation_report(metrics_df, rank_df, models, feature_names, cv_folds):
    """Create evaluation report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_content = []
    
    # Report title
    report_content.append(f"Zircon Mineralization Prediction Model Evaluation Report")
    report_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"Cross-validation folds: {cv_folds}")
    report_content.append("="*50)
    report_content.append("")
    
    # Performance metrics
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
    
    # Ranking results
    report_content.append("Model Performance Ranking (1=Best):")
    for model_name in rank_df.index:
        report_content.append(f"{model_name}: Average Rank {rank_df.loc[model_name, 'Average_Rank']:.2f}")
    
    # Best model
    best_model = rank_df['Average_Rank'].idxmin()
    report_content.append("")
    report_content.append(f"Best Model: {best_model}")
    report_content.append(f"Best Parameters: {models[best_model]['best_params']}")
    
    return "\n".join(report_content), timestamp

# Data loading and preprocessing function
def load_and_preprocess_data(uploaded_file):
    try:
        # Read data
        if uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        else:
            data = pd.read_csv(uploaded_file)
        
        # Handle missing values
        if data.isnull().sum().any():
            data = data.dropna()
        
        # Separate features and target
        X = data.iloc[:, :-1]
        feature_names = X.columns.tolist()
        y = data['Label']
        
        # Split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Save scaler
        joblib.dump(scaler, 'scaler.pkl')
        
        return X_train, X_test, y_train, y_test, feature_names, data.shape
    except Exception as e:
        st.error(f"Data loading and preprocessing error: {e}")
        return None, None, None, None, None, None

# Model training function
def train_models(X_train, y_train, selected_models, search_method, cv_folds, custom_params=None):
    # Define model configurations
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
                'solver': ['liblinear']
            },
            'grid_params': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        },
        'Neural Network': {
            'model': MLPClassifier(max_iter=1000, random_state=42),
            'random_params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 30), (100, 50)],
                'alpha': uniform(0.0001, 0.1),
                'activation': ['relu', 'tanh'],
                'learning_rate_init': uniform(0.001, 0.01)
            },
            'grid_params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 30), (100, 50)],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'activation': ['relu', 'tanh'],
                'learning_rate_init': [0.001, 0.01, 0.1]
            }
        }
    }
    
    # Define evaluation metrics
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'f1': 'f1_macro',
        'roc_auc': 'roc_auc_ovo'
    }
    
    # Only train selected models
    models = {name: config for name, config in base_models.items() if name in selected_models}
    
    # Train models
    trained_models = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, config) in enumerate(models.items()):
        status_text.text(f"Training {name}... (using {cv_folds}-fold cross-validation)")
        
        # Use custom parameters or default parameters
        if custom_params and name in custom_params:
            params = custom_params[name]
            model = config['model']
            model.set_params(**params)
            model.fit(X_train, y_train)
            
            model_dict = {
                'model': model,
                'best_params': params,
                'cv_metrics': {},
                'train_metrics': {},
                'cv_folds': cv_folds
            }
        else:
            # Use search method
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
            else:  # Grid Search
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
            
            # Extract cross-validation performance metrics
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
            
            # Calculate training set performance metrics
            model = search.best_estimator_
            y_train_pred = model.predict(X_train)
            train_metrics = {
                'accuracy': accuracy_score(y_train, y_train_pred),
                'precision': precision_score(y_train, y_train_pred, average='weighted'),
                'recall': recall_score(y_train, y_train_pred, average='weighted'),
                'f1': f1_score(y_train, y_train_pred, average='weighted')
            }
            
            model_dict = {
                'model': model,
                'best_params': search.best_params_,
                'cv_metrics': cv_metrics,
                'train_metrics': train_metrics,
                'cv_folds': cv_folds
            }
        
        # Save model
        model_filename = f"{name.replace(' ', '_')}_model.pkl"
        joblib.dump(model_dict, model_filename)
        
        trained_models[name] = model_dict
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(models))
    
    status_text.text("Training completed!")
    st.session_state.training_complete = True
    st.session_state.models_trained = True
    return trained_models

# Plot confusion matrices
def plot_confusion_matrices(models, X_train, X_test, y_train, y_test):
    figures = {}
    for model_name, model_dict in models.items():
        model = model_dict['model']
        
        # Training set confusion matrix
        y_train_pred = model.predict(X_train)
        cm_train = confusion_matrix(y_train, y_train_pred)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-mineralized', 'Mineralized'],
                    yticklabels=['Non-mineralized', 'Mineralized'],
                    annot_kws={"size": 14},
                    cbar=False, ax=ax1)
        ax1.set_title(f'{model_name} - Training Set Confusion Matrix', fontsize=14)
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        
        # Test set confusion matrix
        y_test_pred = model.predict(X_test)
        cm_test = confusion_matrix(y_test, y_test_pred)
        
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-mineralized', 'Mineralized'],
                    yticklabels=['Non-mineralized', 'Mineralized'],
                    annot_kws={"size": 14},
                    cbar=False, ax=ax2)
        ax2.set_title(f'{model_name} - Test Set Confusion Matrix', fontsize=14)
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
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Random Guessing')
    
    for model_name, model_dict in models.items():
        model = model_dict['model']
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
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
    
    # Add random baseline
    random_precision = sum(y_test) / len(y_test)
    plt.plot([0, 1], [random_precision, random_precision], 'k--', alpha=0.7, label='Random Guessing')
    
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
        
        # Skip logistic regression model
        if model_name == 'Logistic Regression':
            continue
            
        # Handle different model feature importance
        if hasattr(model, 'feature_importances_'):
            # Tree models
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title(f"{model_name} - Feature Importance", fontsize=16)
            bars = plt.bar(range(len(importances)), importances[indices], align="center")
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90, fontsize=12)
            plt.xlim([-1, len(importances)])
            plt.ylabel("Importance Score", fontsize=12)
            
            # Add value labels
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
            # Linear models (except logistic regression)
            coef = model.coef_[0]
            indices = np.argsort(np.abs(coef))[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title(f"{model_name} - Feature Coefficients", fontsize=16)
            bars = plt.bar(range(len(coef)), np.abs(coef)[indices], align="center", color='salmon')
            plt.xticks(range(len(coef)), [feature_names[i] for i in indices], rotation=90, fontsize=12)
            plt.xlim([-1, len(coef)])
            plt.ylabel("Coefficient Absolute Value", fontsize=12)
            
            # Add value labels
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

# Performance metric ranking system
def calculate_model_ranks(metrics_df):
    # Create ranking copy
    rank_df = metrics_df.copy()
    
    # Calculate ranking for each metric (1=best)
    for column in rank_df.columns:
        # All metrics are better when larger
        rank_df[column] = rank_df[column].rank(ascending=False)
    
    # Calculate average ranking
    rank_df['Average_Rank'] = rank_df.mean(axis=1)
    
    # Sort by average ranking
    rank_df = rank_df.sort_values(by='Average_Rank')
    
    return rank_df

# Evaluation function
def evaluate_and_visualize(models, X_train, X_test, y_train, y_test, feature_names, cv_folds):
    # Store evaluation metrics
    metrics = []
    
    st.subheader("Model Best Parameters and Performance Summary")
    
    for model_name, model_dict in models.items():
        with st.expander(f"{model_name} Details"):
            st.write(f"**Best Parameters:** {model_dict['best_params']}")
            st.write(f"**Cross-validation Folds:** {model_dict.get('cv_folds', cv_folds)}")
            
            # Print cross-validation metrics
            if 'cv_metrics' in model_dict and model_dict['cv_metrics']:
                st.write("**Cross-validation Performance Metrics:**")
                cv_data = []
                for metric, value in model_dict['cv_metrics'].items():
                    cv_data.append({
                        'Metric': metric,
                        'Mean': f"{value['mean']:.4f}",
                        'Std': f"± {value['std']:.4f}"
                    })
                st.table(pd.DataFrame(cv_data))
            
            # Print training set performance metrics
            if 'train_metrics' in model_dict and model_dict['train_metrics']:
                st.write("**Training Set Performance Metrics:**")
                train_data = []
                for metric, value in model_dict['train_metrics'].items():
                    train_data.append({
                        'Metric': metric,
                        'Value': f"{value:.4f}"
                    })
                st.table(pd.DataFrame(train_data))
    
    st.subheader("Confusion Matrices")
    confusion_figures = plot_confusion_matrices(models, X_train, X_test, y_train, y_test)
    
    st.subheader("Test Set Performance Metrics")
    for model_name, model_dict in models.items():
        model = model_dict['model']
        
        # Test set predictions
        y_test_pred = model.predict(X_test)
        
        # Ensure model supports probability prediction
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            # For models that don't support probability prediction, use decision function
            y_prob = model.decision_function(X_test)
        
        # Calculate test set metrics
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        # Calculate ROC AUC
        if hasattr(model, "predict_proba") or hasattr(model, "decision_function"):
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            roc_auc = 0.5  # Cannot calculate ROC AUC
        
        # Calculate PR AUC
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
        
        # Display test set classification report
        with st.expander(f"{model_name} Test Set Classification Report"):
            st.text(classification_report(y_test, y_test_pred, target_names=['Non-mineralized', 'Mineralized']))
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame(metrics).set_index('Model')
    
    # Calculate rankings
    rank_df = calculate_model_ranks(metrics_df)
    
    # Display metrics and rankings
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Test Set Performance Metrics:**")
        st.dataframe(metrics_df.style.format("{:.4f}"))
    
    with col2:
        st.write("**Model Performance Ranking (1=Best):**")
        st.dataframe(rank_df.style.format("{:.2f}"))
    
    # Select best model (lowest average rank)
    best_model_name = rank_df['Average_Rank'].idxmin()
    best_model = models[best_model_name]['model']
    st.success(f"**Best Model: {best_model_name}** (Average Rank {rank_df.loc[best_model_name, 'Average_Rank']:.2f})")
    
    # Save best model
    joblib.dump(best_model, 'best_model.pkl')
    
    # Visualization
    st.subheader("ROC Curves")
    roc_fig = plot_roc_curves(models, X_test, y_test)
    
    st.subheader("Precision-Recall Curves")
    pr_fig = plot_pr_curves(models, X_test, y_test)
    
    st.subheader("Feature Importance")
    feature_figures = plot_feature_importance(models, feature_names)
    
    # Save evaluation results
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
    
    st.session_state.evaluation_results = evaluation_results
    st.session_state.evaluation_done = True
    
    return metrics_df, rank_df, evaluation_results

# Prediction function
def predict_new_dataset(models, uploaded_file, selected_models_for_prediction):
    st.subheader("New Dataset Prediction Results")
    
    # Load new dataset
    try:
        if uploaded_file.name.endswith('.xlsx'):
            new_data = pd.read_excel(uploaded_file)
        else:
            new_data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"File reading error: {e}")
        return
    
    # Check and handle missing values
    if new_data.isnull().sum().any():
        st.warning("Missing values found in new dataset, removing rows with missing values...")
        new_data = new_data.dropna()
    
    # Separate features and labels (if labels exist)
    if 'Label' in new_data.columns:
        X_new = new_data.drop('Label', axis=1)
        y_new = new_data['Label']
        has_labels = True
        st.info("Label column found, will calculate performance metrics.")
    else:
        X_new = new_data
        has_labels = False
        st.info("No 'Label' column found, only prediction will be performed.")
    
    # Load previously saved scaler
    try:
        scaler = joblib.load('scaler.pkl')
        X_new_scaled = scaler.transform(X_new)
        st.success("Using saved scaler to standardize data.")
    except FileNotFoundError:
        st.error("Error: Scaler file 'scaler.pkl' not found, please train models first.")
        return
    
    # Predict for each selected model
    for model_name in selected_models_for_prediction:
        if model_name not in models:
            st.warning(f"Model {model_name} not trained, skipping prediction.")
            continue
            
        st.subheader(f"{model_name} Prediction Results")
        model_dict = models[model_name]
        model = model_dict['model']
        
        try:
            # Make predictions
            y_pred = model.predict(X_new_scaled)
            
            # Try to get prediction confidence
            confidence = np.ones(len(y_pred))  # Default value
            if hasattr(model, "predict_proba"):
                confidence = np.max(model.predict_proba(X_new_scaled), axis=1)
            elif hasattr(model, "decision_function"):
                decision_values = model.decision_function(X_new_scaled)
                confidence = 1 / (1 + np.exp(-decision_values))  # Convert to probability
            
            # Create prediction results DataFrame
            prediction_df = pd.DataFrame({
                'Predicted_Label': y_pred,
                'Prediction_Confidence': confidence
            })
            
            # Add original features
            prediction_df = pd.concat([X_new.reset_index(drop=True), prediction_df], axis=1)
            
            # Save prediction results
            prediction_filename = f"{model_name.replace(' ', '_')}_predictions.csv"
            prediction_df.to_csv(prediction_filename, index=False)
            
            # Display prediction results
            st.write(f"**Prediction Results Sample:**")
            st.dataframe(prediction_df.head())
            
            # Download prediction results
            csv = prediction_df.to_csv(index=False)
            st.download_button(
                label=f"Download {model_name} Prediction Results",
                data=csv,
                file_name=prediction_filename,
                mime="text/csv"
            )
            
            # If true labels exist, calculate performance metrics
            if has_labels:
                # Calculate performance metrics
                accuracy = accuracy_score(y_new, y_pred)
                precision = precision_score(y_new, y_pred, average='weighted')
                recall = recall_score(y_new, y_pred, average='weighted')
                f1 = f1_score(y_new, y_pred, average='weighted')
                
                # Display performance metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{accuracy:.4f}")
                col2.metric("Precision", f"{precision:.4f}")
                col3.metric("Recall", f"{recall:.4f}")
                col4.metric("F1 Score", f"{f1:.4f}")
                
                # Plot confusion matrix
                cm = confusion_matrix(y_new, y_pred)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Non-mineralized', 'Mineralized'],
                            yticklabels=['Non-mineralized', 'Mineralized'],
                            annot_kws={"size": 14}, ax=ax)
                ax.set_title(f'{model_name} - New Dataset Confusion Matrix', fontsize=14)
                ax.set_ylabel('True Label', fontsize=12)
                ax.set_xlabel('Predicted Label', fontsize=12)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Display classification report
                with st.expander(f"{model_name} New Dataset Classification Report"):
                    st.text(classification_report(y_new, y_pred, target_names=['Non-mineralized', 'Mineralized']))
            else:
                # Visualize prediction results distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.countplot(x='Predicted_Label', data=prediction_df, ax=ax)
                ax.set_title(f'{model_name} - Prediction Results Distribution', fontsize=16)
                ax.set_xlabel('Predicted Label', fontsize=12)
                ax.set_ylabel('Sample Count', fontsize=12) 
                ax.set_xticklabels(['Non-mineralized', 'Mineralized'])
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
        except Exception as e:
            st.error(f"Error processing {model_name}: {str(e)}")

# Load saved models
def load_saved_models():
    saved_models = {}
    model_files = {
        'XGBoost': 'XGBoost_model.pkl',
        'Random Forest': 'Random_Forest_model.pkl', 
        'SVM': 'SVM_model.pkl',
        'Logistic Regression': 'Logistic_Regression_model.pkl',
        'Neural Network': 'Neural_Network_model.pkl'
    }
    
    for model_name, filename in model_files.items():
        if os.path.exists(filename):
            try:
                model_dict = joblib.load(filename)
                saved_models[model_name] = model_dict
                st.sidebar.success(f"✅ {model_name} loaded")
            except Exception as e:
                st.sidebar.warning(f"⚠️ {model_name} loading failed: {e}")
        else:
            st.sidebar.info(f"📝 {model_name} not trained")
    
    return saved_models

# Download evaluation results function
def download_evaluation_results(evaluation_results, feature_names):
    if not evaluation_results:
        st.warning("No evaluation results to download, please perform model evaluation first.")
        return
    
    st.subheader("📥 Download Evaluation Results")
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create evaluation report
    report_content, report_timestamp = create_evaluation_report(
        evaluation_results['metrics_df'],
        evaluation_results['rank_df'],
        evaluation_results['models'],
        feature_names,
        evaluation_results['cv_folds']
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download performance metrics tables
        st.markdown("### Performance Metrics Tables")
        st.markdown(get_table_download_link(
            evaluation_results['metrics_df'], 
            f"model_metrics_{timestamp}.csv", 
            "📊 Download Performance Metrics Table"
        ), unsafe_allow_html=True)
        
        st.markdown(get_table_download_link(
            evaluation_results['rank_df'], 
            f"model_ranks_{timestamp}.csv", 
            "🏆 Download Model Ranking Table"
        ), unsafe_allow_html=True)
    
    with col2:
        # Download visualization charts
        st.markdown("### Visualization Charts")
        
        # ROC curve
        if 'roc_fig' in evaluation_results:
            st.markdown(get_image_download_link(
                evaluation_results['roc_fig'],
                f"roc_curves_{timestamp}.png",
                "📈 Download ROC Curves"
            ), unsafe_allow_html=True)
        
        # PR curve
        if 'pr_fig' in evaluation_results:
            st.markdown(get_image_download_link(
                evaluation_results['pr_fig'],
                f"pr_curves_{timestamp}.png",
                "📊 Download PR Curves"
            ), unsafe_allow_html=True)
    
    with col3:
        # Download confusion matrices and feature importance
        st.markdown("### Model Detailed Charts")
        
        # Confusion matrices
        if 'confusion_figures' in evaluation_results:
            for model_name, fig in evaluation_results['confusion_figures'].items():
                st.markdown(get_image_download_link(
                    fig,
                    f"confusion_matrix_{model_name}_{timestamp}.png",
                    f"🎯 Download {model_name} Confusion Matrix"
                ), unsafe_allow_html=True)
        
        # Feature importance
        if 'feature_figures' in evaluation_results:
            for model_name, fig in evaluation_results['feature_figures'].items():
                st.markdown(get_image_download_link(
                    fig,
                    f"feature_importance_{model_name}_{timestamp}.png",
                    f"🔍 Download {model_name} Feature Importance"
                ), unsafe_allow_html=True)
    
    # Download complete evaluation report
    st.markdown("---")
    st.markdown("### Complete Evaluation Report")
    st.download_button(
        label="📄 Download Complete Evaluation Report (TXT)",
        data=report_content,
        file_name=f"model_evaluation_report_{timestamp}.txt",
        mime="text/plain"
    )
    
    # Display report preview
    with st.expander("Preview Evaluation Report"):
        st.text(report_content)

# Main application
def main():
    initialize_session_state()
    
    st.title("🔬 Zircon Mineralization Prediction System")
    st.markdown("---")
    
    # Sidebar navigation - directly display five functional areas
    st.sidebar.title("🚀 Functional Area Navigation")
    
    # Functional area buttons
    if st.sidebar.button("🏠 Home", use_container_width=True):
        st.session_state.current_page = "Home"
    
    if st.sidebar.button("📊 Data Upload", use_container_width=True):
        st.session_state.current_page = "Data Upload"
    
    if st.sidebar.button("🤖 Model Training", use_container_width=True):
        st.session_state.current_page = "Model Training"
    
    if st.sidebar.button("📈 Model Evaluation", use_container_width=True):
        st.session_state.current_page = "Model Evaluation"
    
    if st.sidebar.button("🔮 Predict New Data", use_container_width=True):
        st.session_state.current_page = "Predict New Data"
    
    if st.sidebar.button("⚙️ Parameter Settings", use_container_width=True):
        st.session_state.current_page = "Parameter Settings"
    
    st.sidebar.markdown("---")
    
    # Cross-validation settings
    st.sidebar.subheader("🔧 Cross-validation Settings")
    cv_folds = st.sidebar.radio(
        "Select Cross-validation Folds",
        [5, 10],
        index=0 if st.session_state.cv_folds == 5 else 1,
        key="cv_folds_sidebar"
    )
    st.session_state.cv_folds = cv_folds
    st.sidebar.info(f"Current: {cv_folds}-fold cross-validation")
    
    st.sidebar.markdown("---")
    
    # Load saved models
    st.sidebar.subheader("📁 Saved Models")
    saved_models = load_saved_models()
    if saved_models:
        st.session_state.trained_models.update(saved_models)
        st.sidebar.success(f"Loaded {len(saved_models)} models")
    
    # Clear data button
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear All Data", type="secondary"):
        # Reset all states
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        # Reinitialize
        initialize_session_state()
        st.rerun()
    
    # Display current status
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Current Status")
    
    status_col1, status_col2 = st.sidebar.columns(2)
    with status_col1:
        st.metric("Data Status", "✅" if st.session_state.data_loaded else "❌")
    with status_col2:
        trained_count = len(st.session_state.trained_models)
        st.metric("Model Count", f"{trained_count}/5")
    
    # Display content based on current page
    current_page = st.session_state.current_page
    
    # Home page
    if current_page == "Home":
        st.header("Welcome to Zircon Mineralization Prediction System")
        st.markdown("""
        ### 🎯 System Functions
        
        **📊 Data Upload** - Upload zircon data files and perform preprocessing
        
        **🤖 Model Training** - Select and train machine learning models
        - XGBoost
        - Random Forest  
        - SVM
        - Logistic Regression
        - Neural Network
        
        **📈 Model Evaluation** - Evaluate model performance and visualize results
        
        **🔮 Predict New Data** - Use trained models to predict new data
        
        **⚙️ Parameter Settings** - Customize model parameters
        
        ### 🚀 Usage Process
        1. Upload your data in "Data Upload" page
        2. Select models to train in "Model Training" page
        3. View model performance in "Model Evaluation" page
        4. Use models for prediction in "Predict New Data" page
        
        ### 💡 Tips
        - Each functional area's operation results are preserved for comparison
        - You can use "Clear All Data" in sidebar to start over anytime
        - Trained models are automatically saved for future use
        - Model evaluation results can be downloaded and saved
        - You can select 5-fold or 10-fold cross-validation in sidebar
        """)
        
        # Display current status
        col1, col2, col3 = st.columns(3)
        with col1:
            status = "✅ Loaded" if st.session_state.data_loaded else "❌ Not Loaded"
            st.metric("Data Status", status)
        with col2:
            trained_count = len(st.session_state.trained_models)
            st.metric("Trained Models", f"{trained_count}/5")
        with col3:
            status = "✅ Completed" if st.session_state.training_complete else "⏳ Pending Training"
            st.metric("Training Status", status)
        
        # Display cross-validation settings
        st.info(f"Current Cross-validation Setting: **{st.session_state.cv_folds}-fold cross-validation**")
    
    # Data Upload page
    elif current_page == "Data Upload":
        st.header("📊 Data Upload")
        st.markdown("Upload your zircon data files (supports Excel and CSV formats)")
        
        # Display current status
        if st.session_state.data_loaded:
            st.success("✅ Data loaded and preprocessing completed")
            st.write(f"- Training set size: {st.session_state.X_train.shape[0]}")
            st.write(f"- Test set size: {st.session_state.X_test.shape[0]}")
            st.write(f"- Feature count: {len(st.session_state.feature_names)}")
        
        uploaded_file = st.file_uploader("Select File", type=['xlsx', 'csv'])
        
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            
            # Display data information
            try:
                if uploaded_file.name.endswith('.xlsx'):
                    data = pd.read_excel(uploaded_file)
                else:
                    data = pd.read_csv(uploaded_file)
                
                st.write("**Data Preview:**")
                st.dataframe(data.head())
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Data Information:**")
                    st.write(f"- Data shape: {data.shape}")
                    st.write(f"- Feature count: {data.shape[1]-1}")
                    st.write(f"- Sample count: {data.shape[0]}")
                
                with col2:
                    if 'Label' in data.columns:
                        st.write("**Label Distribution:**")
                        label_counts = data['Label'].value_counts()
                        st.write(label_counts)
                
                if 'Label' in data.columns:
                    # Visualize label distribution
                    fig, ax = plt.subplots(figsize=(8, 6))
                    label_counts = data['Label'].value_counts()
                    label_counts.plot(kind='bar', ax=ax)
                    ax.set_title('Label Distribution', fontsize=16)
                    ax.set_xlabel('Label', fontsize=12)
                    ax.set_ylabel('Count', fontsize=12)
                    ax.set_xticklabels(['Non-mineralized', 'Mineralized'], rotation=0)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                # Preprocess data
                if st.button("Start Data Preprocessing", type="primary"):
                    with st.spinner("Preprocessing data..."):
                        X_train, X_test, y_train, y_test, feature_names, data_shape = load_and_preprocess_data(uploaded_file)
                        
                        if X_train is not None:
                            st.session_state.X_train = X_train
                            st.session_state.X_test = X_test
                            st.session_state.y_train = y_train
                            st.session_state.y_test = y_test
                            st.session_state.feature_names = feature_names
                            st.session_state.data_loaded = True
                            st.session_state.data_preprocessed = True
                            
                            st.success("Data preprocessing completed!")
                            st.write(f"- Training set size: {X_train.shape[0]}")
                            st.write(f"- Test set size: {X_test.shape[0]}")
                            st.write(f"- Feature count: {len(feature_names)}")
                            st.write(f"- Feature list: {', '.join(feature_names)}")
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # Model Training page
    elif current_page == "Model Training":
        st.header("🤖 Model Training")
        
        # Display cross-validation settings
        st.info(f"Current Cross-validation Setting: **{st.session_state.cv_folds}-fold cross-validation**")
        
        if not st.session_state.data_loaded:
            st.warning("Please upload and preprocess data first!")
            return
        
        st.write("**Select models to train:**")
        
        # Model selection
        model_options = ['XGBoost', 'Random Forest', 'SVM', 'Logistic Regression', 'Neural Network']
        selected_models = st.multiselect(
            "Select Models",
            model_options,
            default=model_options
        )
        
        # Search method selection
        search_method = st.radio(
            "Select Parameter Search Method",
            ["Random Search", "Grid Search", "Custom Parameters"]
        )
        
        # Display saved models
        if st.session_state.trained_models:
            st.info(f"📁 Loaded {len(st.session_state.trained_models)} trained models")
            trained_list = list(st.session_state.trained_models.keys())
            st.write(f"Trained models: {', '.join(trained_list)}")
        
        # Training options
        col1, col2 = st.columns(2)
        with col1:
            use_existing = st.checkbox("Use saved models (if available)", value=True)
        with col2:
            retrain = st.checkbox("Retrain selected models", value=False)
        
        # Training button
        if st.button("Start Model Training", type="primary"):
            if not selected_models:
                st.error("Please select at least one model!")
                return
            
            with st.spinner("Training models..."):
                # Determine models to train
                models_to_train = selected_models
                if use_existing and not retrain:
                    # Exclude existing models
                    existing_models = list(st.session_state.trained_models.keys())
                    models_to_train = [model for model in selected_models if model not in existing_models]
                    
                    if not models_to_train:
                        st.info("All selected models are already trained, using existing models.")
                    else:
                        st.info(f"Will train new models: {', '.join(models_to_train)}")
                
                if models_to_train or retrain:
                    # Get custom parameters
                    custom_params = {}
                    if search_method == "Custom Parameters":
                        custom_params = st.session_state.get('custom_params', {})
                    
                    trained_models = train_models(
                        st.session_state.X_train, 
                        st.session_state.y_train, 
                        models_to_train if not retrain else selected_models, 
                        search_method,
                        st.session_state.cv_folds,
                        custom_params
                    )
                    
                    # Update session state
                    st.session_state.trained_models.update(trained_models)
                    st.success(f"Model training completed! Trained {len(trained_models)} models")
                else:
                    st.success("Using existing trained models")
    
    # Model Evaluation page
    elif current_page == "Model Evaluation":
        st.header("📈 Model Evaluation")
        
        # Display cross-validation settings
        st.info(f"Current Cross-validation Setting: **{st.session_state.cv_folds}-fold cross-validation**")
        
        if not st.session_state.trained_models:
            st.warning("Please train models first!")
            return
        
        # Display completed evaluation
        if st.session_state.evaluation_done:
            st.success("✅ Model evaluation completed")
            st.write("Previous evaluation results:")
            
            # Directly display previous evaluation results
            evaluate_and_visualize(
                st.session_state.trained_models,
                st.session_state.X_train,
                st.session_state.X_test,
                st.session_state.y_train,
                st.session_state.y_test,
                st.session_state.feature_names,
                st.session_state.cv_folds
            )
            
            # Download evaluation results
            if st.session_state.evaluation_results:
                download_evaluation_results(
                    st.session_state.evaluation_results,
                    st.session_state.feature_names
                )
        else:
            # Evaluation button
            if st.button("Start Evaluation", type="primary"):
                with st.spinner("Evaluating models..."):
                    metrics, rank_df, evaluation_results = evaluate_and_visualize(
                        st.session_state.trained_models,
                        st.session_state.X_train,
                        st.session_state.X_test,
                        st.session_state.y_train,
                        st.session_state.y_test,
                        st.session_state.feature_names,
                        st.session_state.cv_folds
                    )
                    
                    st.session_state.evaluation_results = evaluation_results
                    st.success("Model evaluation completed!")
            
            # Download evaluation results
            if st.session_state.evaluation_results:
                download_evaluation_results(
                    st.session_state.evaluation_results,
                    st.session_state.feature_names
                )
    
    # Predict New Data page
    elif current_page == "Predict New Data":
        st.header("🔮 Predict New Data")
        
        if not st.session_state.trained_models:
            st.warning("Please train models first!")
            return
        
        st.write("Upload new data for prediction")
        new_data_file = st.file_uploader("Select New Data File", type=['xlsx', 'csv'], key="new_data")
        
        if new_data_file is not None:
            # Select models for prediction
            trained_model_names = list(st.session_state.trained_models.keys())
            selected_models_for_prediction = st.multiselect(
                "Select Models for Prediction",
                trained_model_names,
                default=trained_model_names
            )
            
            if st.button("Start Prediction", type="primary"):
                if not selected_models_for_prediction:
                    st.error("Please select at least one model!")
                    return
                
                with st.spinner("Performing prediction..."):
                    predict_new_dataset(
                        st.session_state.trained_models,
                        new_data_file,
                        selected_models_for_prediction
                    )
    
    # Parameter Settings page
    elif current_page == "Parameter Settings":
        st.header("⚙️ Parameter Settings")
        
        st.info("Set custom model parameters here")
        
        # Set parameters for each model
        model_options = ['XGBoost', 'Random Forest', 'SVM', 'Logistic Regression', 'Neural Network']
        
        custom_params = st.session_state.get('custom_params', {})
        
        for model in model_options:
            with st.expander(f"{model} Parameters"):
                if model == 'XGBoost':
                    n_estimators = st.slider("n_estimators", 50, 300, 100, key=f"xgb_n_est")
                    max_depth = st.slider("max_depth", 3, 10, 6, key=f"xgb_depth")
                    learning_rate = st.slider("learning_rate", 0.01, 0.3, 0.1, key=f"xgb_lr")
                    subsample = st.slider("subsample", 0.6, 1.0, 0.8, key=f"xgb_sub")
                    colsample_bytree = st.slider("colsample_bytree", 0.6, 1.0, 0.8, key=f"xgb_col")
                    
                    custom_params[model] = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'learning_rate': learning_rate,
                        'subsample': subsample,
                        'colsample_bytree': colsample_bytree
                    }
                
                elif model == 'Random Forest':
                    n_estimators = st.slider("n_estimators", 50, 300, 100, key=f"rf_n_est")
                    max_depth = st.slider("max_depth", 3, 20, 10, key=f"rf_depth")
                    min_samples_split = st.slider("min_samples_split", 2, 10, 2, key=f"rf_split")
                    min_samples_leaf = st.slider("min_samples_leaf", 1, 5, 1, key=f"rf_leaf")
                    
                    custom_params[model] = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf
                    }
                
                elif model == 'SVM':
                    C = st.slider("C", 0.1, 10.0, 1.0, key=f"svm_c")
                    gamma = st.slider("gamma", 0.01, 1.0, 0.1, key=f"svm_gamma")
                    kernel = st.selectbox("kernel", ['linear', 'rbf', 'poly'], key=f"svm_kernel")
                    
                    custom_params[model] = {
                        'C': C,
                        'gamma': gamma,
                        'kernel': kernel
                    }
                
                elif model == 'Logistic Regression':
                    C = st.slider("C", 0.1, 10.0, 1.0, key=f"lr_c")
                    penalty = st.selectbox("penalty", ['l1', 'l2'], key=f"lr_penalty")
                    
                    custom_params[model] = {
                        'C': C,
                        'penalty': penalty,
                        'solver': 'liblinear'
                    }
                
                elif model == 'Neural Network':
                    hidden_layer_sizes = st.selectbox(
                        "hidden_layer_sizes", 
                        [(50,), (100,), (50, 30), (100, 50)],
                        format_func=lambda x: f"{x}",
                        key=f"nn_layers"
                    )
                    alpha = st.slider("alpha", 0.0001, 0.1, 0.001, key=f"nn_alpha")
                    activation = st.selectbox("activation", ['relu', 'tanh'], key=f"nn_act")
                    learning_rate_init = st.slider("learning_rate_init", 0.001, 0.01, 0.001, key=f"nn_lr")
                    
                    custom_params[model] = {
                        'hidden_layer_sizes': hidden_layer_sizes,
                        'alpha': alpha,
                        'activation': activation,
                        'learning_rate_init': learning_rate_init
                    }
        
        # Save custom parameters to session state
        st.session_state.custom_params = custom_params
        
        if st.button("Save Parameters", type="primary"):
            st.success("Parameters saved! You can now use custom parameters in the Model Training page.")

if __name__ == "__main__":
    main()