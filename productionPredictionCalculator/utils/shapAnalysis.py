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
def run_bnn_shap_analysis(bnn=None, X_train=[], X_test=[], feature_names=[], path_db='', sampling_method='Unknown', bnn_library='pytorch', max_display=10):
    """
    Run SHAP GradientExplainer analysis on a trained BNN and save plots.
    Works with both PyTorch and TensorFlow BNN implementations via bnn_library flag.
    GradientExplainer uses integrated gradients — faster than KernelExplainer,
    more stable than DeepExplainer with sigmoid activations, directly comparable
    to TreeExplainer output.

    :param bnn:             Trained BNN instance (BNN_PT or BNN_TF)
    :param X_train:         Training feature array (numpy, background dataset)
    :param X_test:          Test feature array (numpy, samples to explain)
    :param feature_names:   List of feature name strings
    :param path_db:         Output directory path
    :param sampling_method: Label string for plot titles
    :param bnn_library:     'pytorch' or 'tensorflow'
    :param max_display:     Max features to show in plots
    :return:                shap_values array (n_test_samples x n_features)
    """
    ########################################################################################
    print(f"\nRunning BNN SHAP analysis ({bnn_library}) for sampling method: {sampling_method}")
    ########################################################################################
    if bnn_library == 'pytorch':
        import torch
        # GradientExplainer needs the raw nn.Module and torch tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
        bnn.eval()
        explainer   = shap.GradientExplainer(model=bnn.network, data=X_train_t)
        shap_values = explainer.shap_values(X=X_test_t)
        # GradientExplainer returns list for multi-output — take first output
        if isinstance(shap_values, list): shap_values = shap_values[0]
        shap_values = shap_values.numpy() if hasattr(shap_values, 'numpy') else np.array(shap_values)
    else:
        import tensorflow as tf
        # GradientExplainer works with Keras model directly
        X_train_t = tf.constant(X_train, dtype=tf.float32)
        X_test_t  = tf.constant(X_test,  dtype=tf.float32)
        explainer   = shap.GradientExplainer(model=bnn.model, data=X_train_t)
        shap_values = explainer.shap_values(X=X_test_t)
        if isinstance(shap_values, list): shap_values = shap_values[0]
        shap_values = np.array(shap_values)
    ########################################################################################
    label = f'BNN_{bnn_library.upper()}'
    ########################################################################################
    # Plot 1: Summary (beeswarm)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, max_display=max_display, show=False, plot_type='dot')
    plt.title(f'BNN SHAP Feature Importance — {sampling_method} ({bnn_library})', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(path_db, f'BNN_{label}_SHAP_Summary_{sampling_method}.png'), bbox_inches='tight', dpi=150)
    plt.close()
    ########################################################################################
    # Plot 2: Bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, max_display=max_display, show=False, plot_type='bar')
    plt.title(f'BNN SHAP Mean |Value| — {sampling_method} ({bnn_library})', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(path_db, f'BNN_{label}_SHAP_Bar_{sampling_method}.png'), bbox_inches='tight', dpi=150)
    plt.close()
    ########################################################################################
    # Numeric summary
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked   = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)
    print(f"BNN SHAP feature ranking ({sampling_method} — {bnn_library}):")
    for name, val in ranked: print(f"    {name:<20} mean|SHAP| = {val:.5f}")
    ########################################################################################
    return shap_values
############################################################################################