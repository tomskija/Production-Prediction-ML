############################################################################################
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import shap
matplotlib.use('Agg')
############################################################################################
def run_shap_analysis(regressor=[], X_train=[], X_test=[], feature_names=[], path_db='', sampling_method='Unknown', max_display=10):
    """
    Run SHAP TreeExplainer analysis on a trained RandomForestRegressor and save plots.
    :param regressor:       Trained sklearn RandomForestRegressor
    :param X_train:         Training feature array (numpy, used to build explainer background)
    :param X_test:          Test feature array (numpy, used for SHAP value computation)
    :param feature_names:   List of feature name strings matching columns of X_train / X_test
    :param path_db:         Output directory path (same as rest of pipeline)
    :param sampling_method: Label string for plot titles (e.g. 'GMM_spherical')
    :param max_display:     Max features to show in summary plot
    :return:                shap_values array (n_test_samples x n_features)
    """
    ########################################################################################
    print(f"\nRunning SHAP analysis for sampling method: {sampling_method}")
    # TreeExplainer is exact for tree-based models — no approximation needed
    explainer   = shap.TreeExplainer(regressor, data=X_train, feature_perturbation='interventional')
    shap_values = explainer(X_test)
    ########################################################################################
    # Plot 1: Summary (beeswarm) — global feature importance across all test samples ---
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values.values, X_test, feature_names=feature_names, max_display=max_display, show=False, plot_type='dot')
    plt.title(f'SHAP Feature Importance — {sampling_method} Sampling', fontsize=13)
    plt.tight_layout()
    summary_path = os.path.join(path_db, f'SHAP_Summary_{sampling_method}.png')
    plt.savefig(summary_path, bbox_inches='tight', dpi=150)
    ########################################################################################
    # Plot 2: Bar plot — mean absolute SHAP value per feature (clean, easy to read) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values.values, X_test, feature_names=feature_names, max_display=max_display, show=False, plot_type='bar')
    plt.title(f'SHAP Mean |Value| by Feature — {sampling_method} Sampling', fontsize=13)
    plt.tight_layout()
    bar_path = os.path.join(path_db, f'SHAP_Bar_{sampling_method}.png')
    plt.savefig(bar_path, bbox_inches='tight', dpi=150)
    ########################################################################################
    # Plot 3: Waterfall — single prediction breakdown for the highest-production well ---
    best_idx = int(np.argmax(X_test[:, feature_names.index('Production')] if 'Production' in feature_names else np.arange(len(X_test))))
    # Fall back to index 0 if Production column not in X_test (it shouldn't be)
    best_idx = 0 if best_idx >= len(shap_values) else best_idx
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(shap_values[best_idx], max_display=max_display, show=False)
    plt.title(f'SHAP Waterfall — Single Prediction (idx {best_idx}) — {sampling_method}', fontsize=11)
    plt.tight_layout()
    waterfall_path = os.path.join(path_db, f'SHAP_Waterfall_{sampling_method}.png')
    plt.savefig(waterfall_path, bbox_inches='tight', dpi=150)
    ########################################################################################
    # Numeric summary: mean absolute SHAP value per feature, printed to console ---
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    ranked   = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)
    print("SHAP feature ranking ("+sampling_method+"):")
    for name, val in ranked: print(f"    {name:<20} mean|SHAP| = {val:.5f}")
    ########################################################################################
    return shap_values
############################################################################################