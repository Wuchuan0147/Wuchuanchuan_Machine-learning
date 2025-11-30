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

# 设置中文字体
try:
    font_list = [f.name for f in fm.fontManager.ttflist]
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'STSong', 'SimSun', 'Arial Unicode MS', 'Heiti SC']
    
    available_font = None
    for font in chinese_fonts:
        if font in font_list:
            available_font = font
            break
    
    if available_font:
        plt.rcParams['font.sans-serif'] = [available_font]
        mpl.rcParams['font.sans-serif'] = [available_font]
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    plt.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['axes.unicode_minus'] = False
except Exception as e:
    st.warning(f"字体设置错误: {e}")

# 全局绘图参数设置
plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 14, 'xtick.labelsize': 13, 
    'ytick.labelsize': 13, 'legend.fontsize': 12, 'figure.figsize': (10, 8),
    'figure.titlesize': 16, 'axes.titlesize': 16, 'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

warnings.filterwarnings('ignore')

# 设置页面
st.set_page_config(
    page_title="机器学习判别分类模型系统",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 获取当前目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 初始化session state
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
        st.session_state.current_page = "首页"
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'custom_params' not in st.session_state:
        st.session_state.custom_params = {}
    if 'training_complete' not in st.session_state:
        st.session_state.training_complete = False
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    if 'cv_folds' not in st.session_state:
        st.session_state.cv_folds = 5  # 默认5折交叉验证
    if 'data_preprocessed' not in st.session_state:
        st.session_state.data_preprocessed = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'evaluation_done' not in st.session_state:
        st.session_state.evaluation_done = False

# 辅助函数：创建下载链接
def get_table_download_link(df, filename, link_text):
    """生成表格下载链接"""
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def get_image_download_link(fig, filename, link_text):
    """生成图片下载链接"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">{link_text}</a>'
    return href

def create_evaluation_report(metrics_df, rank_df, models, feature_names, cv_folds):
    """创建评估报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_content = []
    
    # 报告标题
    report_content.append(f"机器学习判别分类模型评估报告")
    report_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"交叉验证折数: {cv_folds}")
    report_content.append("="*50)
    report_content.append("")
    
    # 性能指标
    report_content.append("模型性能指标汇总:")
    report_content.append("")
    for model_name in metrics_df.index:
        report_content.append(f"{model_name}:")
        report_content.append(f"  准确率: {metrics_df.loc[model_name, 'Accuracy']:.4f}")
        report_content.append(f"  精确率: {metrics_df.loc[model_name, 'Precision']:.4f}")
        report_content.append(f"  召回率: {metrics_df.loc[model_name, 'Recall']:.4f}")
        report_content.append(f"  F1分数: {metrics_df.loc[model_name, 'F1']:.4f}")
        report_content.append(f"  ROC AUC: {metrics_df.loc[model_name, 'ROC_AUC']:.4f}")
        report_content.append(f"  PR AUC: {metrics_df.loc[model_name, 'PR_AUC']:.4f}")
        report_content.append("")
    
    # 排名结果
    report_content.append("模型性能排名 (1=最佳):")
    for model_name in rank_df.index:
        report_content.append(f"{model_name}: 平均排名 {rank_df.loc[model_name, 'Average_Rank']:.2f}")
    
    # 最佳模型
    best_model = rank_df['Average_Rank'].idxmin()
    report_content.append("")
    report_content.append(f"最佳模型: {best_model}")
    report_content.append(f"最佳参数: {models[best_model]['best_params']}")
    
    return "\n".join(report_content), timestamp

# 数据加载和预处理函数
def load_and_preprocess_data(uploaded_file):
    try:
        # 读取数据
        if uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        else:
            data = pd.read_csv(uploaded_file)
        
        # 处理缺失值
        if data.isnull().sum().any():
            data = data.dropna()
        
        # 分离特征和目标
        X = data.iloc[:, :-1]
        feature_names = X.columns.tolist()
        y = data['Label']
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)
        
        # 标准化特征
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # 保存标准化器
        joblib.dump(scaler, 'scaler.pkl')
        
        return X_train, X_test, y_train, y_test, feature_names, data.shape
    except Exception as e:
        st.error(f"数据加载和预处理错误: {e}")
        return None, None, None, None, None, None

# 模型训练函数
def train_models(X_train, y_train, selected_models, search_method, cv_folds, custom_params=None):
    # 定义模型配置
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
    
    # 定义评估指标
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'f1': 'f1_macro',
        'roc_auc': 'roc_auc_ovo'
    }
    
    # 只训练选中的模型
    models = {name: config for name, config in base_models.items() if name in selected_models}
    
    # 训练模型
    trained_models = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, config) in enumerate(models.items()):
        status_text.text(f"训练 {name}... (使用{cv_folds}折交叉验证)")
        
        # 使用自定义参数或默认参数
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
            # 使用搜索方法
            if search_method == "随机搜索":
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
            else:  # 网格搜索
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
            
            # 提取交叉验证性能指标
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
            
            # 计算训练集性能指标
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
        
        # 保存模型
        model_filename = f"{name.replace(' ', '_')}_model.pkl"
        joblib.dump(model_dict, model_filename)
        
        trained_models[name] = model_dict
        
        # 更新进度条
        progress_bar.progress((i + 1) / len(models))
    
    status_text.text("训练完成!")
    st.session_state.training_complete = True
    st.session_state.models_trained = True
    return trained_models

# 绘制混淆矩阵
def plot_confusion_matrices(models, X_train, X_test, y_train, y_test):
    figures = {}
    for model_name, model_dict in models.items():
        model = model_dict['model']
        
        # 训练集混淆矩阵
        y_train_pred = model.predict(X_train)
        cm_train = confusion_matrix(y_train, y_train_pred)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['非成矿', '成矿'],
                    yticklabels=['非成矿', '成矿'],
                    annot_kws={"size": 14},
                    cbar=False, ax=ax1)
        ax1.set_title(f'{model_name} - 训练集混淆矩阵', fontsize=14)
        ax1.set_ylabel('实际类别', fontsize=12)
        ax1.set_xlabel('预测类别', fontsize=12)
        
        # 测试集混淆矩阵
        y_test_pred = model.predict(X_test)
        cm_test = confusion_matrix(y_test, y_test_pred)
        
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['非成矿', '成矿'],
                    yticklabels=['非成矿', '成矿'],
                    annot_kws={"size": 14},
                    cbar=False, ax=ax2)
        ax2.set_title(f'{model_name} - 测试集混淆矩阵', fontsize=14)
        ax2.set_ylabel('实际类别', fontsize=12)
        ax2.set_xlabel('预测类别', fontsize=12)
        
        plt.tight_layout()
        figures[model_name] = fig
        st.pyplot(fig)
        plt.close()
    
    return figures

# 绘制ROC曲线
def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='随机猜测')
    
    for model_name, model_dict in models.items():
        model = model_dict['model']
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('假正例率 (FPR)', fontsize=12)
    plt.ylabel('真正例率 (TPR)', fontsize=12)
    plt.title('ROC曲线', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close()
    return fig

# 绘制PR曲线
def plot_pr_curves(models, X_test, y_test):
    plt.figure(figsize=(10, 8))
    
    # 添加随机基准线
    random_precision = sum(y_test) / len(y_test)
    plt.plot([0, 1], [random_precision, random_precision], 'k--', alpha=0.7, label='随机猜测')
    
    for model_name, model_dict in models.items():
        model = model_dict['model']
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            avg_precision = average_precision_score(y_test, y_prob)
            plt.plot(recall, precision, lw=2, label=f'{model_name} (AP = {avg_precision:.3f})')
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('召回率 (Recall)', fontsize=12)
    plt.ylabel('精确率 (Precision)', fontsize=12)
    plt.title('PR曲线', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close()
    return fig

# 绘制特征重要性
def plot_feature_importance(models, feature_names):
    figures = {}
    for model_name, model_dict in models.items():
        model = model_dict['model']
        
        # 跳过逻辑回归模型
        if model_name == 'Logistic Regression':
            continue
            
        # 处理不同模型的特征重要性
        if hasattr(model, 'feature_importances_'):
            # 树模型
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title(f"{model_name} - 特征重要性", fontsize=16)
            bars = plt.bar(range(len(importances)), importances[indices], align="center")
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90, fontsize=12)
            plt.xlim([-1, len(importances)])
            plt.ylabel("重要性得分", fontsize=12)
            
            # 添加数值标签
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
            # 线性模型 (除逻辑回归外)
            coef = model.coef_[0]
            indices = np.argsort(np.abs(coef))[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title(f"{model_name} - 特征系数", fontsize=16)
            bars = plt.bar(range(len(coef)), np.abs(coef)[indices], align="center", color='salmon')
            plt.xticks(range(len(coef)), [feature_names[i] for i in indices], rotation=90, fontsize=12)
            plt.xlim([-1, len(coef)])
            plt.ylabel("系数绝对值", fontsize=12)
            
            # 添加数值标签
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

# 性能指标排名系统
def calculate_model_ranks(metrics_df):
    # 创建排名副本
    rank_df = metrics_df.copy()
    
    # 为每个指标计算排名 (1=最好)
    for column in rank_df.columns:
        # 所有指标都是越大越好
        rank_df[column] = rank_df[column].rank(ascending=False)
    
    # 计算平均排名
    rank_df['Average_Rank'] = rank_df.mean(axis=1)
    
    # 按平均排名排序
    rank_df = rank_df.sort_values(by='Average_Rank')
    
    return rank_df

# 评估函数
def evaluate_and_visualize(models, X_train, X_test, y_train, y_test, feature_names, cv_folds):
    # 存储评估指标
    metrics = []
    
    st.subheader("模型最佳参数和性能汇总")
    
    for model_name, model_dict in models.items():
        with st.expander(f"{model_name} 详细信息"):
            st.write(f"**最佳参数:** {model_dict['best_params']}")
            st.write(f"**交叉验证折数:** {model_dict.get('cv_folds', cv_folds)}")
            
            # 打印交叉验证指标
            if 'cv_metrics' in model_dict and model_dict['cv_metrics']:
                st.write("**交叉验证性能指标:**")
                cv_data = []
                for metric, value in model_dict['cv_metrics'].items():
                    cv_data.append({
                        '指标': metric,
                        '平均值': f"{value['mean']:.4f}",
                        '标准差': f"± {value['std']:.4f}"
                    })
                st.table(pd.DataFrame(cv_data))
            
            # 打印训练集性能指标
            if 'train_metrics' in model_dict and model_dict['train_metrics']:
                st.write("**训练集性能指标:**")
                train_data = []
                for metric, value in model_dict['train_metrics'].items():
                    train_data.append({
                        '指标': metric,
                        '值': f"{value:.4f}"
                    })
                st.table(pd.DataFrame(train_data))
    
    st.subheader("混淆矩阵")
    confusion_figures = plot_confusion_matrices(models, X_train, X_test, y_train, y_test)
    
    st.subheader("测试集性能指标")
    for model_name, model_dict in models.items():
        model = model_dict['model']
        
        # 测试集预测
        y_test_pred = model.predict(X_test)
        
        # 确保模型支持概率预测
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            # 对于不支持概率预测的模型，使用决策函数
            y_prob = model.decision_function(X_test)
        
        # 计算测试集指标
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        # 计算ROC AUC
        if hasattr(model, "predict_proba") or hasattr(model, "decision_function"):
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            roc_auc = 0.5  # 无法计算ROC AUC
        
        # 计算PR AUC
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
        
        # 显示测试集分类报告
        with st.expander(f"{model_name} 测试集分类报告"):
            st.text(classification_report(y_test, y_test_pred, target_names=['非成矿', '成矿']))
    
    # 创建指标数据框
    metrics_df = pd.DataFrame(metrics).set_index('Model')
    
    # 计算排名
    rank_df = calculate_model_ranks(metrics_df)
    
    # 显示指标和排名
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**模型测试集性能指标:**")
        st.dataframe(metrics_df.style.format("{:.4f}"))
    
    with col2:
        st.write("**模型性能排名 (1=最佳):**")
        st.dataframe(rank_df.style.format("{:.2f}"))
    
    # 选择最佳模型 (平均排名最高)
    best_model_name = rank_df['Average_Rank'].idxmin()
    best_model = models[best_model_name]['model']
    st.success(f"**最佳模型: {best_model_name}** (平均排名 {rank_df.loc[best_model_name, 'Average_Rank']:.2f})")
    
    # 保存最佳模型
    joblib.dump(best_model, 'best_model.pkl')
    
    # 可视化
    st.subheader("ROC曲线")
    roc_fig = plot_roc_curves(models, X_test, y_test)
    
    st.subheader("PR曲线")
    pr_fig = plot_pr_curves(models, X_test, y_test)
    
    st.subheader("特征重要性")
    feature_figures = plot_feature_importance(models, feature_names)
    
    # 保存评估结果
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

# 预测函数
def predict_new_dataset(models, uploaded_file, selected_models_for_prediction):
    st.subheader("新数据集预测结果")
    
    # 加载新数据集
    try:
        if uploaded_file.name.endswith('.xlsx'):
            new_data = pd.read_excel(uploaded_file)
        else:
            new_data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"读取文件错误: {e}")
        return
    
    # 检查并处理缺失值
    if new_data.isnull().sum().any():
        st.warning("新数据集中存在缺失值，删除包含缺失值的行...")
        new_data = new_data.dropna()
    
    # 分离特征和标签（如果存在标签）
    if 'Label' in new_data.columns:
        X_new = new_data.drop('Label', axis=1)
        y_new = new_data['Label']
        has_labels = True
        st.info("找到标签列，将计算性能指标。")
    else:
        X_new = new_data
        has_labels = False
        st.info("未找到'Label'列，仅进行预测。")
    
    # 加载之前保存的标准化器
    try:
        scaler = joblib.load('scaler.pkl')
        X_new_scaled = scaler.transform(X_new)
        st.success("使用保存的标准化器对数据进行标准化。")
    except FileNotFoundError:
        st.error("错误: 未找到标准化器文件'scaler.pkl'，请先训练模型。")
        return
    
    # 对每个选中的模型进行预测
    for model_name in selected_models_for_prediction:
        if model_name not in models:
            st.warning(f"模型 {model_name} 未训练，跳过预测。")
            continue
            
        st.subheader(f"{model_name} 预测结果")
        model_dict = models[model_name]
        model = model_dict['model']
        
        try:
            # 进行预测
            y_pred = model.predict(X_new_scaled)
            
            # 尝试获取预测置信度
            confidence = np.ones(len(y_pred))  # 默认值
            if hasattr(model, "predict_proba"):
                confidence = np.max(model.predict_proba(X_new_scaled), axis=1)
            elif hasattr(model, "decision_function"):
                decision_values = model.decision_function(X_new_scaled)
                confidence = 1 / (1 + np.exp(-decision_values))  # 转换为概率
            
            # 创建预测结果DataFrame
            prediction_df = pd.DataFrame({
                'Predicted_Label': y_pred,
                'Prediction_Confidence': confidence
            })
            
            # 添加原始特征
            prediction_df = pd.concat([X_new.reset_index(drop=True), prediction_df], axis=1)
            
            # 保存预测结果
            prediction_filename = f"{model_name.replace(' ', '_')}_predictions.csv"
            prediction_df.to_csv(prediction_filename, index=False)
            
            # 显示预测结果
            st.write(f"**预测结果示例:**")
            st.dataframe(prediction_df.head())
            
            # 下载预测结果
            csv = prediction_df.to_csv(index=False)
            st.download_button(
                label=f"下载 {model_name} 预测结果",
                data=csv,
                file_name=prediction_filename,
                mime="text/csv"
            )
            
            # 如果有真实标签，计算性能指标
            if has_labels:
                # 计算性能指标
                accuracy = accuracy_score(y_new, y_pred)
                precision = precision_score(y_new, y_pred, average='weighted')
                recall = recall_score(y_new, y_pred, average='weighted')
                f1 = f1_score(y_new, y_pred, average='weighted')
                
                # 显示性能指标
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("准确率", f"{accuracy:.4f}")
                col2.metric("精确率", f"{precision:.4f}")
                col3.metric("召回率", f"{recall:.4f}")
                col4.metric("F1分数", f"{f1:.4f}")
                
                # 绘制混淆矩阵
                cm = confusion_matrix(y_new, y_pred)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['非成矿', '成矿'],
                            yticklabels=['非成矿', '成矿'],
                            annot_kws={"size": 14}, ax=ax)
                ax.set_title(f'{model_name} - 新数据集混淆矩阵', fontsize=14)
                ax.set_ylabel('实际类别', fontsize=12)
                ax.set_xlabel('预测类别', fontsize=12)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # 显示分类报告
                with st.expander(f"{model_name} 新数据集分类报告"):
                    st.text(classification_report(y_new, y_pred, target_names=['非成矿', '成矿']))
            else:
                # 可视化预测结果分布
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.countplot(x='Predicted_Label', data=prediction_df, ax=ax)
                ax.set_title(f'{model_name} - 预测结果分布', fontsize=16)
                ax.set_xlabel('预测类别', fontsize=12)
                ax.set_ylabel('样本数量', fontsize=12) 
                ax.set_xticklabels(['非成矿', '成矿'])
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
        except Exception as e:
            st.error(f"处理 {model_name} 时出错: {str(e)}")

# 加载已保存的模型
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
                st.sidebar.success(f"✅ {model_name} 已加载")
            except Exception as e:
                st.sidebar.warning(f"⚠️ {model_name} 加载失败: {e}")
        else:
            st.sidebar.info(f"📝 {model_name} 未训练")
    
    return saved_models

# 下载评估结果功能
def download_evaluation_results(evaluation_results, feature_names):
    if not evaluation_results:
        st.warning("没有可下载的评估结果，请先进行模型评估。")
        return
    
    st.subheader("📥 下载评估结果")
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建评估报告
    report_content, report_timestamp = create_evaluation_report(
        evaluation_results['metrics_df'],
        evaluation_results['rank_df'],
        evaluation_results['models'],
        feature_names,
        evaluation_results['cv_folds']
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 下载性能指标表格
        st.markdown("### 性能指标表格")
        st.markdown(get_table_download_link(
            evaluation_results['metrics_df'], 
            f"model_metrics_{timestamp}.csv", 
            "📊 下载性能指标表格"
        ), unsafe_allow_html=True)
        
        st.markdown(get_table_download_link(
            evaluation_results['rank_df'], 
            f"model_ranks_{timestamp}.csv", 
            "🏆 下载模型排名表格"
        ), unsafe_allow_html=True)
    
    with col2:
        # 下载可视化图表
        st.markdown("### 可视化图表")
        
        # ROC曲线
        if 'roc_fig' in evaluation_results:
            st.markdown(get_image_download_link(
                evaluation_results['roc_fig'],
                f"roc_curves_{timestamp}.png",
                "📈 下载ROC曲线"
            ), unsafe_allow_html=True)
        
        # PR曲线
        if 'pr_fig' in evaluation_results:
            st.markdown(get_image_download_link(
                evaluation_results['pr_fig'],
                f"pr_curves_{timestamp}.png",
                "📊 下载PR曲线"
            ), unsafe_allow_html=True)
    
    with col3:
        # 下载混淆矩阵和特征重要性
        st.markdown("### 模型详细图表")
        
        # 混淆矩阵
        if 'confusion_figures' in evaluation_results:
            for model_name, fig in evaluation_results['confusion_figures'].items():
                st.markdown(get_image_download_link(
                    fig,
                    f"confusion_matrix_{model_name}_{timestamp}.png",
                    f"🎯 下载{model_name}混淆矩阵"
                ), unsafe_allow_html=True)
        
        # 特征重要性
        if 'feature_figures' in evaluation_results:
            for model_name, fig in evaluation_results['feature_figures'].items():
                st.markdown(get_image_download_link(
                    fig,
                    f"feature_importance_{model_name}_{timestamp}.png",
                    f"🔍 下载{model_name}特征重要性"
                ), unsafe_allow_html=True)
    
    # 下载完整评估报告
    st.markdown("---")
    st.markdown("### 完整评估报告")
    st.download_button(
        label="📄 下载完整评估报告 (TXT)",
        data=report_content,
        file_name=f"model_evaluation_report_{timestamp}.txt",
        mime="text/plain"
    )
    
    # 显示报告预览
    with st.expander("预览评估报告"):
        st.text(report_content)

# 主应用
def main():
    initialize_session_state()
    
    st.title("🔬 机器学习判别分类模型系统")
    st.markdown("---")
    
    # 侧边栏导航 - 直接显示五个功能区
    st.sidebar.title("🚀 功能区导航")
    
    # 功能区按钮
    if st.sidebar.button("🏠 首页", use_container_width=True):
        st.session_state.current_page = "首页"
    
    if st.sidebar.button("📊 数据上传", use_container_width=True):
        st.session_state.current_page = "数据上传"
    
    if st.sidebar.button("🤖 模型训练", use_container_width=True):
        st.session_state.current_page = "模型训练"
    
    if st.sidebar.button("📈 模型评估", use_container_width=True):
        st.session_state.current_page = "模型评估"
    
    if st.sidebar.button("🔮 预测新数据", use_container_width=True):
        st.session_state.current_page = "预测新数据"
    
    if st.sidebar.button("⚙️ 参数设置", use_container_width=True):
        st.session_state.current_page = "参数设置"
    
    st.sidebar.markdown("---")
    
    # 交叉验证设置
    st.sidebar.subheader("🔧 交叉验证设置")
    cv_folds = st.sidebar.radio(
        "选择交叉验证折数",
        [5, 10],
        index=0 if st.session_state.cv_folds == 5 else 1,
        key="cv_folds_sidebar"
    )
    st.session_state.cv_folds = cv_folds
    st.sidebar.info(f"当前使用: {cv_folds}折交叉验证")
    
    st.sidebar.markdown("---")
    
    # 加载已保存的模型
    st.sidebar.subheader("📁 已保存模型")
    saved_models = load_saved_models()
    if saved_models:
        st.session_state.trained_models.update(saved_models)
        st.sidebar.success(f"已加载 {len(saved_models)} 个模型")
    
    # 清空数据按钮
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ 清空所有数据", type="secondary"):
        # 重置所有状态
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        # 重新初始化
        initialize_session_state()
        st.rerun()
    
    # 显示当前状态
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 当前状态")
    
    status_col1, status_col2 = st.sidebar.columns(2)
    with status_col1:
        st.metric("数据状态", "✅" if st.session_state.data_loaded else "❌")
    with status_col2:
        trained_count = len(st.session_state.trained_models)
        st.metric("模型数量", f"{trained_count}/5")
    
    # 根据当前页面显示内容
    current_page = st.session_state.current_page
    
    # 首页
    if current_page == "首页":
        st.header("欢迎使用机器学习判别分类模型系统")
        st.markdown("""
        ### 🎯 系统功能
        
        **📊 数据上传** - 上传锆石数据文件并进行预处理
        
        **🤖 模型训练** - 选择并训练机器学习模型
        - XGBoost
        - Random Forest  
        - SVM
        - Logistic Regression
        - Neural Network
        
        **📈 模型评估** - 评估模型性能并可视化结果
        
        **🔮 预测新数据** - 使用训练好的模型对新数据进行预测
        
        **⚙️ 参数设置** - 自定义模型参数
        
        ### 🚀 使用流程
        1. 在"数据上传"页面上传您的数据
        2. 在"模型训练"页面选择要训练的模型
        3. 在"模型评估"页面查看模型性能
        4. 在"预测新数据"页面使用模型进行预测
        
        ### 💡 温馨提示
        - 每个功能区的操作结果都会保留，方便您对比查看
        - 可以随时使用侧边栏的"清空所有数据"重新开始
        - 训练好的模型会自动保存，下次可直接使用
        - 模型评估结果可以下载保存
        - 可在侧边栏选择5折或10折交叉验证
        """)
        
        # 显示当前状态
        col1, col2, col3 = st.columns(3)
        with col1:
            status = "✅ 已加载" if st.session_state.data_loaded else "❌ 未加载"
            st.metric("数据状态", status)
        with col2:
            trained_count = len(st.session_state.trained_models)
            st.metric("已训练模型", f"{trained_count}/5")
        with col3:
            status = "✅ 完成" if st.session_state.training_complete else "⏳ 待训练"
            st.metric("训练状态", status)
        
        # 显示交叉验证设置
        st.info(f"当前交叉验证设置: **{st.session_state.cv_folds}折交叉验证**")
    
    # 数据上传页面
    elif current_page == "数据上传":
        st.header("📊 数据上传")
        st.markdown("上传您的锆石数据文件（支持Excel和CSV格式）")
        
        # 显示当前状态
        if st.session_state.data_loaded:
            st.success("✅ 数据已加载并预处理完成")
            st.write(f"- 训练集大小: {st.session_state.X_train.shape[0]}")
            st.write(f"- 测试集大小: {st.session_state.X_test.shape[0]}")
            st.write(f"- 特征数量: {len(st.session_state.feature_names)}")
        
        uploaded_file = st.file_uploader("选择文件", type=['xlsx', 'csv'])
        
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            
            # 显示数据信息
            try:
                if uploaded_file.name.endswith('.xlsx'):
                    data = pd.read_excel(uploaded_file)
                else:
                    data = pd.read_csv(uploaded_file)
                
                st.write("**数据预览:**")
                st.dataframe(data.head())
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**数据信息:**")
                    st.write(f"- 数据形状: {data.shape}")
                    st.write(f"- 特征数量: {data.shape[1]-1}")
                    st.write(f"- 样本数量: {data.shape[0]}")
                
                with col2:
                    if 'Label' in data.columns:
                        st.write("**标签分布:**")
                        label_counts = data['Label'].value_counts()
                        st.write(label_counts)
                
                if 'Label' in data.columns:
                    # 可视化标签分布
                    fig, ax = plt.subplots(figsize=(8, 6))
                    label_counts = data['Label'].value_counts()
                    label_counts.plot(kind='bar', ax=ax)
                    ax.set_title('标签分布', fontsize=16)
                    ax.set_xlabel('标签', fontsize=12)
                    ax.set_ylabel('数量', fontsize=12)
                    ax.set_xticklabels(['非成矿', '成矿'], rotation=0)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                # 预处理数据
                if st.button("开始预处理数据", type="primary"):
                    with st.spinner("正在预处理数据..."):
                        X_train, X_test, y_train, y_test, feature_names, data_shape = load_and_preprocess_data(uploaded_file)
                        
                        if X_train is not None:
                            st.session_state.X_train = X_train
                            st.session_state.X_test = X_test
                            st.session_state.y_train = y_train
                            st.session_state.y_test = y_test
                            st.session_state.feature_names = feature_names
                            st.session_state.data_loaded = True
                            st.session_state.data_preprocessed = True
                            
                            st.success("数据预处理完成!")
                            st.write(f"- 训练集大小: {X_train.shape[0]}")
                            st.write(f"- 测试集大小: {X_test.shape[0]}")
                            st.write(f"- 特征数量: {len(feature_names)}")
                            st.write(f"- 特征列表: {', '.join(feature_names)}")
            except Exception as e:
                st.error(f"处理文件时出错: {e}")
    
    # 模型训练页面
    elif current_page == "模型训练":
        st.header("🤖 模型训练")
        
        # 显示交叉验证设置
        st.info(f"当前交叉验证设置: **{st.session_state.cv_folds}折交叉验证**")
        
        if not st.session_state.data_loaded:
            st.warning("请先上传并预处理数据!")
            return
        
        st.write("**选择要训练的模型:**")
        
        # 模型选择
        model_options = ['XGBoost', 'Random Forest', 'SVM', 'Logistic Regression', 'Neural Network']
        selected_models = st.multiselect(
            "选择模型",
            model_options,
            default=model_options
        )
        
        # 搜索方法选择
        search_method = st.radio(
            "选择参数搜索方法",
            ["随机搜索", "网格搜索", "自定义参数"]
        )
        
        # 显示已保存的模型
        if st.session_state.trained_models:
            st.info(f"📁 已加载 {len(st.session_state.trained_models)} 个训练好的模型")
            trained_list = list(st.session_state.trained_models.keys())
            st.write(f"已训练模型: {', '.join(trained_list)}")
        
        # 训练选项
        col1, col2 = st.columns(2)
        with col1:
            use_existing = st.checkbox("使用已保存的模型（如存在）", value=True)
        with col2:
            retrain = st.checkbox("重新训练选中的模型", value=False)
        
        # 训练按钮
        if st.button("开始训练模型", type="primary"):
            if not selected_models:
                st.error("请至少选择一个模型!")
                return
            
            with st.spinner("正在训练模型..."):
                # 确定要训练的模型
                models_to_train = selected_models
                if use_existing and not retrain:
                    # 排除已存在的模型
                    existing_models = list(st.session_state.trained_models.keys())
                    models_to_train = [model for model in selected_models if model not in existing_models]
                    
                    if not models_to_train:
                        st.info("所有选中的模型都已训练完成，使用现有模型。")
                    else:
                        st.info(f"将训练新模型: {', '.join(models_to_train)}")
                
                if models_to_train or retrain:
                    # 获取自定义参数
                    custom_params = {}
                    if search_method == "自定义参数":
                        custom_params = st.session_state.get('custom_params', {})
                    
                    trained_models = train_models(
                        st.session_state.X_train, 
                        st.session_state.y_train, 
                        models_to_train if not retrain else selected_models, 
                        search_method,
                        st.session_state.cv_folds,
                        custom_params
                    )
                    
                    # 更新session state
                    st.session_state.trained_models.update(trained_models)
                    st.success(f"模型训练完成! 共训练 {len(trained_models)} 个模型")
                else:
                    st.success("使用现有训练好的模型")
    
    # 模型评估页面
    elif current_page == "模型评估":
        st.header("📈 模型评估")
        
        # 显示交叉验证设置
        st.info(f"当前交叉验证设置: **{st.session_state.cv_folds}折交叉验证**")
        
        if not st.session_state.trained_models:
            st.warning("请先训练模型!")
            return
        
        # 显示已完成的评估
        if st.session_state.evaluation_done:
            st.success("✅ 模型评估已完成")
            st.write("以下是之前的评估结果:")
            
            # 直接显示之前的评估结果
            evaluate_and_visualize(
                st.session_state.trained_models,
                st.session_state.X_train,
                st.session_state.X_test,
                st.session_state.y_train,
                st.session_state.y_test,
                st.session_state.feature_names,
                st.session_state.cv_folds
            )
            
            # 下载评估结果
            if st.session_state.evaluation_results:
                download_evaluation_results(
                    st.session_state.evaluation_results,
                    st.session_state.feature_names
                )
        else:
            # 评估按钮
            if st.button("开始评估", type="primary"):
                with st.spinner("正在评估模型..."):
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
                    st.success("模型评估完成!")
            
            # 下载评估结果
            if st.session_state.evaluation_results:
                download_evaluation_results(
                    st.session_state.evaluation_results,
                    st.session_state.feature_names
                )
    
    # 预测新数据页面
    elif current_page == "预测新数据":
        st.header("🔮 预测新数据")
        
        if not st.session_state.trained_models:
            st.warning("请先训练模型!")
            return
        
        st.write("上传新数据进行预测")
        new_data_file = st.file_uploader("选择新数据文件", type=['xlsx', 'csv'], key="new_data")
        
        if new_data_file is not None:
            # 选择用于预测的模型
            trained_model_names = list(st.session_state.trained_models.keys())
            selected_models_for_prediction = st.multiselect(
                "选择用于预测的模型",
                trained_model_names,
                default=trained_model_names
            )
            
            if st.button("开始预测", type="primary"):
                if not selected_models_for_prediction:
                    st.error("请至少选择一个模型!")
                    return
                
                with st.spinner("正在进行预测..."):
                    predict_new_dataset(
                        st.session_state.trained_models,
                        new_data_file,
                        selected_models_for_prediction
                    )
    
    # 参数设置页面
    elif current_page == "参数设置":
        st.header("⚙️ 参数设置")
        
        st.info("在这里设置自定义模型参数")
        
        # 为每个模型设置参数
        model_options = ['XGBoost', 'Random Forest', 'SVM', 'Logistic Regression', 'Neural Network']
        
        custom_params = st.session_state.get('custom_params', {})
        
        for model in model_options:
            with st.expander(f"{model} 参数"):
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
        
        # 保存自定义参数到session state
        st.session_state.custom_params = custom_params
        
        if st.button("保存参数", type="primary"):
            st.success("参数已保存! 现在可以在模型训练页面使用自定义参数了。")

if __name__ == "__main__":
    main()