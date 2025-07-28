import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter
from sklearn.model_selection import ParameterGrid
import time


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_dense_if_sparse(X):
    return X.toarray() if issparse(X) else X

def compare_models_metrics(models, X_test, y_test, average="macro", labels=None):
    if isinstance(models, list):
        models_dict = {m.name: m for m in models}
    else:
        models_dict = models

    rows = []
    for name, model in models_dict.items():
        metrics = model.evaluate(X_test, y_test, average=average, labels=labels, output_dict=True)
        metrics_flat = {k: v for k, v in metrics.items() if not isinstance(v, (list, pd.DataFrame, pd.Series, tuple))}
        metrics_flat["model"] = name
        rows.append(metrics_flat)

    df = pd.DataFrame(rows).set_index("model")
    return df

def check_naive_predictions(model, X_test, y_test, y_train=None, model_name=None, show_plot=True):
    if model_name is None:
        model_name = model.__class__.__name__
    
    y_pred = model.predict(X_test)
    
    if y_train is not None:
        majority_class = Counter(y_train).most_common(1)[0][0]
    else:
        majority_class = Counter(y_test).most_common(1)[0][0]
    
    unique_classes = np.unique(y_test)
    pred_counts = Counter(y_pred)
    true_counts = Counter(y_test)
    
    total_predictions = len(y_pred)
    majority_pred_ratio = pred_counts.get(majority_class, 0) / total_predictions
    unique_pred_classes = len(set(y_pred))
    
    dummy_majority = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy_majority.fit(X_test, y_test)
    y_dummy_maj = dummy_majority.predict(X_test)
    
    model_f1 = f1_score(y_test, y_pred, average='macro')
    dummy_maj_f1 = f1_score(y_test, y_dummy_maj, average='macro')
    
    is_naive_majority = (majority_pred_ratio > 0.95)
    is_single_class = (unique_pred_classes == 1)
    improvement_over_dummy = model_f1 - dummy_maj_f1
    
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Diagnostic - {model_name}', fontsize=16, fontweight='bold')
        
        ax1 = axes[0, 0]
        classes, counts = zip(*sorted(true_counts.items()))
        ax1.bar([str(c) for c in classes], counts, alpha=0.7)
        ax1.set_title('True Classes')
        ax1.set_ylabel('Number')
        
        ax2 = axes[0, 1]
        pred_values = [pred_counts.get(c, 0) for c in classes]
        bars2 = ax2.bar([str(c) for c in classes], pred_values, alpha=0.7)
        maj_idx = list(classes).index(majority_class) if majority_class in classes else 0
        if maj_idx < len(bars2):
            bars2[maj_idx].set_color('red')
        ax2.set_title('Predictions')
        ax2.set_ylabel('Number')
        
        ax3 = axes[1, 0]
        methods = ['Model', 'Dummy']
        f1_scores = [model_f1, dummy_maj_f1]
        ax3.bar(methods, f1_scores, alpha=0.7)
        ax3.set_title('F1-Score')
        ax3.set_ylim(0, 1)
        
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        diagnostic_text = f"""
{model_name}

Predicted classes: {unique_pred_classes}/{len(unique_classes)}
% majority: {majority_pred_ratio:.1%}
F1 vs Dummy: +{improvement_over_dummy:.3f}
"""
        
        ax4.text(0.05, 0.95, diagnostic_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    return {
        'model_name': model_name,
        'unique_classes_predicted': unique_pred_classes,
        'majority_prediction_ratio': majority_pred_ratio,
        'improvement_over_dummy': improvement_over_dummy,
        'is_naive_majority': is_naive_majority,
        'is_single_class_predictor': is_single_class
    }

def batch_check_naive_predictions(models_dict, X_test, y_test, y_train=None, show_individual_plots=False):
    results = []
    
    for name, model in models_dict.items():
        stats = check_naive_predictions(model, X_test, y_test, y_train, name, show_plot=show_individual_plots)
        results.append(stats)
    
    df = pd.DataFrame(results)
    
    problematic = df[(df['is_naive_majority']) | (df['is_single_class_predictor']) | (df['improvement_over_dummy'] < 0)]
    
    if len(problematic) > 0:
        for _, row in problematic.iterrows():
            issues = []
            if row['is_single_class_predictor']:
                issues.append("unique class")
            if row['is_naive_majority']:
                issues.append("naive majority")
            if row['improvement_over_dummy'] < 0:
                issues.append("worse than dummy")
    return df

def get_default_param_grid(model_type):
    
    grids = {
        'LogReg': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'max_iter': [1000, 2000]
        },
        
        'RandomForest': {
            'n_estimators': [100, 300, 500],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        },
        
        'XGBoostClassifier': {
            'n_estimators': [100, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        },
        
        'KNNClassifier': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'cosine']
        },
        
        'NaiveBayesClassifier': {
            'alpha': [0.01, 0.1, 1.0, 10.0]
        },
        
        'LightGBMClassifier': {
            'n_estimators': [100, 300],
            'num_leaves': [20, 31, 50],
            'learning_rate': [0.05, 0.1, 0.2],
            'feature_fraction': [0.8, 0.9, 1.0]
        },
        
        'ExtraTreesClassifierWrapper': {
            'n_estimators': [100, 300],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        },
        
        'AdaBoostClassifierWrapper': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.5, 1.0, 1.5]
        },
        
        'RidgeClassifierWrapper': {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        },
        
        'SGDClassifierWrapper': {
            'alpha': [0.0001, 0.001, 0.01],
            'max_iter': [1000, 2000],
            'learning_rate': ['constant', 'adaptive']
        },
        
        'MLPClassifier': {
            'hidden_dims': [(256,), (512,), (256, 128), (512, 256)],
            'lr': [0.001, 0.01],
            'epochs': [20, 30],
            'batch_size': [64, 128]
        }
    }
    
    return grids.get(model_type, {})

def generalized_gridsearch(model, X_train, y_train, X_val=None, y_val=None, 
                          custom_param_grid=None, scoring='f1_macro', 
                          model_name=None, verbose=True, max_combinations=50):
    
    if model_name is None:
        model_name = model.__class__.__name__
    
    if X_val is None or y_val is None:
        X_val, y_val = X_train, y_train
    
    if custom_param_grid is not None:
        param_grid = custom_param_grid
    else:
        param_grid = get_default_param_grid(model_name)
    
    if not param_grid:
        return {
            'best_model': model,
            'best_params': {},
            'best_score': None,
            'message': f"Pas de grille définie pour {model_name}"
        }
    
    param_combinations = list(ParameterGrid(param_grid))
    if len(param_combinations) > max_combinations:
        np.random.seed(42)
        param_combinations = np.random.choice(param_combinations, max_combinations, replace=False)
        if verbose:
            print(f"Limitation à {max_combinations} combinaisons sur {len(list(ParameterGrid(param_grid)))}")
    
    best_score = -np.inf
    best_params = None
    best_model = None
    results = []
    
    start_time = time.time()
    
    for i, params in enumerate(param_combinations):
        try:
            model_class = model.__class__
            test_model = model_class(**params)
            
            if hasattr(test_model, 'fit'):
                test_model.fit(X_train, y_train)
            else:
                test_model.fit(X_train, y_train, X_val, y_val)
            
            y_pred = test_model.predict(X_val)
            
            if scoring == 'f1_macro':
                score = f1_score(y_val, y_pred, average='macro')
            elif scoring == 'accuracy':
                score = accuracy_score(y_val, y_pred)
            else:
                score = f1_score(y_val, y_pred, average='macro')
            
            results.append({
                'params': params,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_params = params
                best_model = test_model
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(param_combinations)} - Best: {best_score:.4f}")
                
        except Exception as e:
            if verbose:
                print(f"  Erreur avec {params}: {str(e)[:50]}")
            continue
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"\n{model_name} GridSearch terminé en {elapsed_time:.1f}s")
        print(f"Meilleur score ({scoring}): {best_score:.4f}")
        print(f"Meilleurs paramètres: {best_params}")
    
    return {
        'best_model': best_model,
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results,
        'n_combinations_tested': len(results),
        'elapsed_time': elapsed_time,
        'scoring': scoring
    }