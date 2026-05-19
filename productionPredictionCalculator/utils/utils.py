############################################################################################
import numpy as np
import pandas as pd
import os
import warnings
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn import mixture, preprocessing
from sklearn.metrics import silhouette_score, r2_score, mean_squared_error, explained_variance_score, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from tqdm import tqdm
############################################################################################
warnings.filterwarnings('ignore')
sns.set(font_scale=0.8)

############################################################################################
def checkData(inJson=[]):
    ########################################################################################
    p = {item['name']: item['value'] for item in inJson['data']['params']}
    t = {item['name']: item['values'] for item in inJson['data']['tables']}
    ########################################################################################
    inputData = {
        'name':                      p['name'],
        'problemfolder_db':          p['problemfolder_db'],
        'min_pred_norm':             p['min_pred_norm'],
        'max_pred_norm':             p['max_pred_norm'],
        'min_target_norm':           p['min_target_norm'],
        'max_target_norm':           p['max_target_norm'],
        'plot':                      bool(p['plot']),
        'fi_random_state':           p['fi_random_state'],
        'fi_n_estimators':           p['fi_n_estimators'],
        'fi_max_depth':              p['fi_max_depth'],
        'fi_max_features':           p['fi_max_features'],
        'n_range':                   range(p['n_range_start'],                   p['n_range_end']),
        'kmeans_random_state_range': range(p['kmeans_random_state_range_start'], p['kmeans_random_state_range_end']),
        'gmm_random_state_range':    range(p['gmm_random_state_range_start'],    p['gmm_random_state_range_end']),
        'predictive_features':       t['predictive_features'],
        'target_feature':            p['target_feature'],
        'split_seed_range':          range(p['split_seed_range_start'],          p['split_seed_range_end']),
        'rf_seed_range':             range(p['rf_seed_range_start'],             p['rf_seed_range_end']),
        'run_sampling_split':        bool(p['run_sampling_split']),
        'run_test':                  bool(p['run_test']),
        'min_plot':                  p['min_plot'],
        'max_plot':                  p['max_plot'],
    }
    ########################################################################################
    return inputData

############################################################################################
def setup_output_directory(name='Production_Results', problemfolder_db='z_Feature_Selection'):
    """
    :param name:             Name suffix for the output directory
    :param problemfolder_db: Base folder name
    :return:                 path_db string
    """
    path_db = problemfolder_db #+ '_' + name
    print(path_db)
    if not os.path.exists(path_db):
        os.makedirs(path_db)
    else:
        shutil.rmtree(path_db)
        os.makedirs(path_db)
    return path_db

############################################################################################
def load_and_clean_data(df=pd.DataFrame()):
    """
    :param df: Raw input dataframe
    :return:   Cleaned dataframe with WellIndex dropped
    """
    df = df.drop(columns=['WellIndex'])
    df = df.reset_index(drop=True)
    print(df.shape)
    return df

############################################################################################
def feature_engineering(df_data1=pd.DataFrame(), min_pred_norm=-1.3875, max_pred_norm=1.3875, min_target_norm=0.005, max_target_norm=0.995, path_db='', plot=True):
    """
    :param df_data1:        Raw dataset containing production data
    :param min_pred_norm:   Min predictive feature normalization
    :param max_pred_norm:   Max predictive feature normalization
    :param min_target_norm: Min target feature normalization
    :param max_target_norm: Max target feature normalization
    :param path_db:         Path to save histograms
    :param plot:            Whether to plot before/after histograms
    :return:                Engineered and normalized dataset
    """
    ########################################################################################
    if plot:
        df_data1.hist(figsize=(10, 10))
        plt.tight_layout()
        plt.savefig(path_db + '/Original_Histograms.png', bbox_inches='tight')
    ########################################################################################
    df_org        = pd.DataFrame(df_data1, index=df_data1.index, columns=df_data1.columns)
    y_target      = df_org['Production']
    df_data2_norm = df_org.iloc[:, :-1]
    ########################################################################################
    scaler        = preprocessing.StandardScaler().fit(df_data2_norm)
    df_data2_norm = pd.DataFrame(scaler.fit_transform(df_data2_norm), index=df_data2_norm.index, columns=df_data2_norm.columns)
    y_target      = np.array([y_target]).transpose()
    scaler        = preprocessing.StandardScaler().fit(y_target)
    y_target      = pd.DataFrame(scaler.fit_transform(y_target), columns=['Production'])
    ########################################################################################
    scaler        = preprocessing.RobustScaler().fit(df_data2_norm)
    df_data2_norm = pd.DataFrame(scaler.transform(df_data2_norm), index=df_data2_norm.index, columns=df_data2_norm.columns)
    scaler        = preprocessing.RobustScaler().fit(y_target)
    y_target      = pd.DataFrame(scaler.fit_transform(y_target), columns=['Production'])
    ########################################################################################
    scaler        = preprocessing.PowerTransformer().fit(df_data2_norm)
    df_data2_norm = pd.DataFrame(scaler.transform(df_data2_norm), index=df_data2_norm.index, columns=df_data2_norm.columns)
    ########################################################################################
    scaler        = preprocessing.MinMaxScaler(feature_range=(min_pred_norm, max_pred_norm)).fit(df_data2_norm)
    df_data2_norm = pd.DataFrame(scaler.transform(df_data2_norm), index=df_data2_norm.index, columns=df_data2_norm.columns)
    scaler        = preprocessing.MinMaxScaler(feature_range=(min_target_norm, max_target_norm)).fit(y_target)
    y_target_norm = pd.DataFrame(scaler.transform(y_target), columns=['Production'])
    y_target_norm = y_target_norm.replace(0, 0.0001)
    ########################################################################################
    pred_cols     = df_org.columns[:-1].tolist()
    df_out        = pd.concat([pd.DataFrame(df_data2_norm, columns=pred_cols), pd.DataFrame(y_target_norm, columns=['Production'])], axis=1)
    ########################################################################################
    print('Production min: ', df_out['Production'].min())
    print('Production max: ', df_out['Production'].max())
    ########################################################################################
    if plot:
        df_out.hist(figsize=(10, 10))
        plt.tight_layout()
        plt.savefig(path_db + '/Engineered_Histograms.png', bbox_inches='tight')
    ########################################################################################
    return df_out

############################################################################################
def correlation_analysis(df=pd.DataFrame(), path_db=''):
    """
    :param df:      Engineered dataset
    :param path_db: Path to save heatmap figures
    :return:        correlation, rank_correlation, rank_correlation_pval, rank_correlation_scatter
    """
    ########################################################################################
    correlation                      = df.corr().iloc[-1, :-1]
    rank_correlation_scatter, pval   = stats.spearmanr(df)
    rank_correlation                 = rank_correlation_scatter[:, -1][:-1]
    rank_correlation_pval            = pval[:, -1][:-1]
    ########################################################################################
    plt.figure(figsize=(16, 10))
    sns.heatmap(df.corr(), annot=True, linewidth=0, vmin=-1, square=True)
    plt.title("Correlation Heatmap: Features and Response", size=18)
    plt.tight_layout()
    plt.savefig(path_db + '/Correlation_Heatmap.png', bbox_inches='tight')
    ########################################################################################
    plt.figure(figsize=(16, 10))
    tick_labels = df.columns.tolist()
    sns.heatmap(rank_correlation_scatter, annot=True, linewidth=0, vmin=-1, square=True, xticklabels=tick_labels, yticklabels=tick_labels)
    plt.title("Rank Correlation Heatmap: Features and Response", size=18)
    plt.tight_layout()
    plt.savefig(path_db + '/Rank_Correlation_Heatmap.png', bbox_inches='tight')
    ########################################################################################
    plt.close('all')
    ########################################################################################
    return correlation, rank_correlation, rank_correlation_pval, rank_correlation_scatter

############################################################################################
def feature_information(df=pd.DataFrame(), path_db='', random_state=5195, n_estimators=100, max_depth=25, max_features=3):
    """
    :param df:           Engineered dataset with predictive and target features
    :param path_db:      Path to save the output figure
    :param random_state: Random state for the random forest
    :param n_estimators: Number of trees in the random forest
    :param max_depth:    Max depth of the random forest
    :param max_features: Max features to consider per split
    :return: None
    """
    ########################################################################################
    x         = df.iloc[:, :-1]
    y         = df.iloc[:, [-1]]
    lab_enc   = preprocessing.LabelEncoder()
    y_encoded = lab_enc.fit_transform(y)
    mi        = mutual_info_regression(x, np.ravel(y_encoded))
    mi       /= np.max(mi)
    indices1  = np.argsort(mi)[::-1]
    rf        = RandomForestRegressor(oob_score=True, max_depth=max_depth, random_state=random_state, n_estimators=n_estimators, max_features=max_features)
    rf.fit(x, np.ravel(y_encoded))
    importances = rf.feature_importances_
    std         = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    indices2    = np.argsort(importances)[::-1]
    ########################################################################################
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Mutual Information", size=12)
    plt.bar(range(x.shape[1]), mi[indices1], color="g", yerr=std[indices1], align="center")
    plt.xticks(range(x.shape[1]), x.columns[indices1], rotation=90)
    plt.xlim([-1, x.shape[1]])
    plt.subplot(1, 2, 2)
    plt.title("Feature Importances", size=12)
    plt.bar(range(x.shape[1]), importances[indices2], color="g", yerr=std[indices2], align="center")
    plt.xticks(range(x.shape[1]), x.columns[indices2], rotation=90)
    plt.xlim([-1, x.shape[1]])
    plt.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=1., wspace=0.2, hspace=0.2)
    plt.savefig(path_db + '/Mutual_Info_and_Feature_Import.png', bbox_inches='tight')

############################################################################################
def plot_well_data(df=pd.DataFrame(), path_db=''):
    """
    :param df:      Engineered dataset containing Latitude, Longitude, Production columns
    :param path_db: Path to save the output figure
    :return: None
    """
    ########################################################################################
    fig, ax     = plt.subplots(figsize=(16, 7))
    lat         = df['Latitude'];  long = df['Longitude'];  cum_oil_365 = df['Production']
    cum_oil_p05 = np.percentile(cum_oil_365.values, 5)
    cum_oil_p95 = np.percentile(cum_oil_365.values, 95)
    cax         = ax.scatter(lat, long, c=cum_oil_365, cmap=plt.cm.plasma, vmin=cum_oil_p05, vmax=cum_oil_p95, alpha=0.8, linewidths=0.8, edgecolors="black")
    cbar        = fig.colorbar(cax)
    cbar.set_label('Production', rotation=270, size=12, labelpad=20)
    cbar.ax.tick_params(labelsize=10)
    ax.set_ylabel('Longitude', size=12);  ax.set_xlabel('Latitude', size=12)
    ax.set_title('Well Data - Production', size=14)
    ax.set_xlim([lat.min(), lat.max()]);  ax.set_ylim([long.min(), long.max()])
    plt.tight_layout()
    fig.savefig(path_db + '/Well_Data_ScatterPlot.png', bbox_inches='tight')

############################################################################################
def dynamicallyPickClustering(df=pd.DataFrame(), path_db='', n_range=range(2, 30), kmeans_random_state_range=range(0, 50), gmm_random_state_range=range(0, 50)):
    """
    :param df:                        Engineered dataset containing Latitude and Longitude columns
    :param path_db:                   Path to save the output figure
    :param n_range:                   Range of cluster counts to evaluate
    :param kmeans_random_state_range: Range of random states to sweep for K-means
    :param gmm_random_state_range:    Range of random states to sweep for GMM
    :return:                          df, ArrayVals, best_method, best_params
    """
    ########################################################################################
    gmm_methods = ['full', 'spherical', 'diag', 'tied']
    ########################################################################################
    lat_min  = df['Latitude'].min();   lat_max  = df['Latitude'].max()
    long_min = df['Longitude'].min();  long_max = df['Longitude'].max()
    df['Norm_Lat']  = (df['Latitude']  - lat_min)  / (lat_max  - lat_min)
    df['Norm_Long'] = (df['Longitude'] - long_min) / (long_max - long_min)
    data_to_fit = df[['Norm_Lat', 'Norm_Long']].values
    ########################################################################################
    def eval_gmm(n=0, method='full', random_state=0):
        try:
            gmm    = mixture.GaussianMixture(n_components=n, covariance_type=method, n_init=5, max_iter=500, random_state=random_state)
            labels = gmm.fit_predict(data_to_fit)
            if len(np.unique(labels)) < 2: return -1, n, method, random_state
            return silhouette_score(data_to_fit, labels), n, method, random_state
        except Exception:
            return -1, n, method, random_state
    ########################################################################################
    def eval_kmeans(n=0, random_state=0):
        try:
            km     = KMeans(n_clusters=n, n_init=10, init='k-means++', max_iter=5000, tol=0.000001, random_state=random_state)
            labels = km.fit_predict(data_to_fit)
            if len(np.unique(labels)) < 2: return -1, n, random_state
            return silhouette_score(data_to_fit, labels), n, random_state
        except Exception:
            return -1, n, random_state
    ########################################################################################
    print("Sweeping GMM methods and random states...")
    gmm_results = Parallel(n_jobs=-1)(delayed(eval_gmm)(n=n, method=method, random_state=rs) for method in gmm_methods for n in n_range for rs in gmm_random_state_range)
    print("Sweeping K-means random states...")
    kmeans_results = Parallel(n_jobs=-1)(delayed(eval_kmeans)(n=n, random_state=rs) for n in n_range for rs in kmeans_random_state_range)
    best_score  = -1;  best_method = None;  best_params = {}
    ########################################################################################
    for score, n, method, random_state in gmm_results:
        if score > best_score:
            best_score  = score
            best_method = f'GMM {method}'
            best_params = {'type': 'gmm', 'n': n, 'covariance_type': method, 'random_state': random_state}
    ########################################################################################
    for score, n, random_state in kmeans_results:
        if score > best_score:
            best_score  = score
            best_method = f'K-means random state {random_state}'
            best_params = {'type': 'kmeans', 'n': n, 'random_state': random_state}
    ########################################################################################
    print(f"\nBest method: {best_method} | n_clusters: {best_params['n']} | silhouette score: {round(best_score, 4)}")
    ########################################################################################
    if best_params['type'] == 'gmm':
        model = mixture.GaussianMixture(n_components=best_params['n'], covariance_type=best_params['covariance_type'], n_init=5, max_iter=500, random_state=best_params['random_state'])
    else:
        model = KMeans(n_clusters=best_params['n'], n_init=10, init='k-means++', max_iter=5000, tol=0.000001, random_state=best_params['random_state'])
    ########################################################################################
    model.fit(data_to_fit)
    ########################################################################################
    df['cluster'] = model.predict(data_to_fit)
    df            = df.sort_values('cluster', ascending=True).reset_index(drop=True)
    categories    = np.unique(df['cluster'])
    colordict     = dict(zip(categories, np.linspace(0, 1, len(categories))))
    df["color"]   = df['cluster'].apply(lambda x: colordict[x])
    print(df['cluster'].value_counts(sort=True))
    ########################################################################################
    if best_params['type'] == 'gmm':
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(df['Norm_Lat'], df['Norm_Long'], c=df['color'])
        ax.set_title(f'Best GMM: {best_method} | n={best_params["n"]} | rs={best_params["random_state"]} | silhouette={round(best_score, 4)}')
    else:
        cluster_centers = model.cluster_centers_
        points          = np.vstack([cluster_centers[:, 0], cluster_centers[:, 1]]).transpose()
        vor             = Voronoi(points)
        fig, ax         = plt.subplots(figsize=(12, 6))
        voronoi_plot_2d(vor, ax=ax, line_alpha=0.6, line_colors='green', show_vertices=False, line_width=2)
        ax.scatter(df['Norm_Lat'], df['Norm_Long'], c=df['color'])
        for i, center in enumerate(cluster_centers):
            ax.scatter(center[0], center[1], s=150, marker='o', linewidths=1.0, edgecolors="black")
            ax.text(center[0], center[1], str(i+1), horizontalalignment='center', verticalalignment='center', size=7.5)
        ax.set_title(f'Best K-means: {best_method} | n={best_params["n"]} | silhouette={round(best_score, 4)}')

    ########################################################################################
    ax.set_xlabel('Latitude (Normalized)');  ax.set_ylabel('Longitude (Normalized)')
    ax.set_xlim(-0.01, 1.01);               ax.set_ylim(-0.01, 1.01)
    fig.savefig(path_db + '/Sp_Clustering.png', bbox_inches='tight')
    ########################################################################################
    n_clusters = df['cluster'].nunique()
    ArrayVals  = [df[df['cluster'] == i] for i in range(n_clusters)]
    for i, cluster_df in enumerate(ArrayVals): print(f"Cluster {i:02d}: {cluster_df.shape}")
    ########################################################################################
    return df, ArrayVals, best_method, best_params

############################################################################################
def training_testing_datasets1(rnd_state=0, shuffle_bool=True, test_size=0.3, ArrayVals=[], sample_method='Random'):
    """
    :param rnd_state:     Train/test random state split
    :param shuffle_bool:  Whether to shuffle the data
    :param test_size:     Train/test split size
    :param ArrayVals:     Array containing data belonging to each cluster
    :param sample_method: Chosen method to derive the clusters
    :return:              Finalized training and testing datasets
    """
    ########################################################################################
    if sample_method == 'Random':
        train, test  = train_test_split(ArrayVals, test_size=test_size, random_state=rnd_state, shuffle=shuffle_bool)
        train_arrays = [np.asarray(train.to_numpy())]
        test_arrays  = [np.asarray(test.to_numpy())]
        columns      = ArrayVals.columns.tolist()
    else:
        train_arrays, test_arrays = [], []
        for cluster_df in ArrayVals:
            tr, te = train_test_split(cluster_df, test_size=test_size, random_state=rnd_state, shuffle=shuffle_bool)
            train_arrays.append(np.asarray(tr.to_numpy()))
            test_arrays.append(np.asarray(te.to_numpy()))
        columns = ArrayVals[0].columns.tolist()
    ########################################################################################
    drop_cols   = [c for c in columns if c in ['Norm_Lat', 'Norm_Long', 'cluster', 'color']]
    training_df = pd.DataFrame(np.vstack(train_arrays), columns=columns).drop(columns=drop_cols)
    testing_df  = pd.DataFrame(np.vstack(test_arrays),  columns=columns).drop(columns=drop_cols)
    ########################################################################################
    return training_df, testing_df

############################################################################################
def random_forest_prod_prediction(split_seed=0, random_seed=0, max_depth=10, num_trees_rf=50, max_features=3, plot_print=False, plot_min=0, plot_max=1, print_str='', path_db='', ArrayVals=[], sampling_method='Random', parameters_rf=[]):
    ########################################################################################
    """
    :param split_seed:      Training/testing random seed split
    :param random_seed:     Random forest regressor random seed
    :param max_depth:       Maximum depth of the random forest
    :param num_trees_rf:    Number of trees in the random forest
    :param max_features:    Max features to consider per split
    :param plot_print:      Whether to plot results
    :param plot_min:        Minimum x/y axis value
    :param plot_max:        Maximum x/y axis value
    :param print_str:       Sampling method label for plot titles
    :param path_db:         Path to save output figures
    :param ArrayVals:       Cluster arrays
    :param sampling_method: Sampling method string
    :param parameters_rf:   Array of feature column names including target
    :return:                Model performance metrics
    """
    ########################################################################################
    training_data_new, testing_data_new = training_testing_datasets1(rnd_state=split_seed, shuffle_bool=True, test_size=0.3, ArrayVals=ArrayVals, sample_method=sampling_method)
    traindata_rf     = np.asarray(training_data_new[parameters_rf].to_numpy())
    testdata_rf      = np.asarray(testing_data_new[parameters_rf].to_numpy())
    X_train, y_train = traindata_rf[:, :-1], traindata_rf[:, -1]
    X_test,  y_test  = testdata_rf[:, :-1],  testdata_rf[:, -1]
    ########################################################################################
    regressor        = RandomForestRegressor(oob_score=True, max_depth=max_depth, random_state=random_seed, n_estimators=num_trees_rf, max_features=max_features)
    regressor.fit(X=X_train, y=y_train)
    y_pred_train = regressor.predict(X_train)
    y_pred_test  = regressor.predict(X_test)
    ########################################################################################
    rf_RMSE_train          = float(np.sqrt(mean_squared_error(y_train, y_pred_train)) * 100)
    rf_mape_train          = float(np.mean(mean_absolute_error(y_train, y_pred_train) / y_train) * 100)
    rf_Var_Explained_train = float(explained_variance_score(y_train, y_pred_train) * 100)
    rf_r2_train            = float(r2_score(y_train, y_pred_train) * 100)
    rf_RMSE_test           = float(np.sqrt(mean_squared_error(y_test, y_pred_test)) * 100)
    rf_mape_test           = float(np.mean(mean_absolute_error(y_test, y_pred_test) / y_test) * 100)
    rf_Var_Explained_test  = float(explained_variance_score(y_test, y_pred_test) * 100)
    rf_r2_test             = float(r2_score(y_test, y_pred_test) * 100)
    ########################################################################################
    str07 = f"Training RMSE: {round(rf_RMSE_train, 3)}% , MAPE: {round(rf_mape_train, 3)}% , Exp_var: {round(rf_Var_Explained_train, 3)}% , R^2: {round(rf_r2_train, 3)}%"
    str08 = f"Testing  RMSE: {round(rf_RMSE_test, 3)}%  , MAPE: {round(rf_mape_test, 3)}% , Exp_var: {round(rf_Var_Explained_test, 3)}% , R^2: {round(rf_r2_test, 3)}%"
    ########################################################################################
    if plot_print:
        print(str07); print(str08)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, y_true, y_pred, split in zip(axes, [y_train, y_test], [y_pred_train, y_pred_test], ['Training', 'Testing']):
            ax.scatter(y_true, y_pred, s=None, c='red', alpha=0.2, linewidths=0.3, edgecolors="black")
            ax.set_title(f'Random Forest Model Fit with {split} Dataset given {print_str} Sampling', size=10)
            ax.set_xlabel('Actual Production', size=10);  ax.set_ylabel('Estimated Production', size=10)
            ax.set_xlim(plot_min, plot_max);              ax.set_ylim(plot_min, plot_max)
            ax.arrow(plot_min, plot_min, plot_max, plot_max, width=0.001, color='black', head_length=5.0, head_width=0.0)
        plt.tight_layout()
        plt.savefig(path_db + '/RF_Final_Fit.png', bbox_inches='tight')
    ########################################################################################
    return rf_r2_test, rf_Var_Explained_test, rf_RMSE_test, rf_mape_test, str07, str08

############################################################################################
def find_best_rf_seeds(predictive_features=[], target_feature='Production', ArrayVals=[], sampling_method='Random', parameters_rf=[], path_db='', split_seed_range=range(0, 100), rf_seed_range=range(0, 100)):
    ########################################################################################
    """
    :param predictive_features: List of predictive feature column names
    :param target_feature:      Target feature column name
    :param ArrayVals:           Cluster arrays
    :param sampling_method:     Sampling method string
    :param parameters_rf:       Array of feature column names including target
    :param path_db:             Path to save output figures
    :param split_seed_range:    Range of train/test split seeds to sweep
    :param rf_seed_range:       Range of RF random seeds to sweep
    :return:                    p dict with best seeds and hyperparameters
    """
    ########################################################################################
    print_str = sampling_method
    def eval_rf_seeds(split_seed=0, rf_seed=0):
        try:
            r2, _, _, _, _, _ = random_forest_prod_prediction(
                split_seed=split_seed, random_seed=rf_seed,
                max_depth=10, num_trees_rf=50, max_features=len(predictive_features),
                plot_print=False, plot_min=0, plot_max=1, print_str=print_str,
                path_db=path_db, ArrayVals=ArrayVals, sampling_method=sampling_method,
                parameters_rf=parameters_rf)
            return r2, split_seed, rf_seed
        except Exception:
            return -1, split_seed, rf_seed
    ########################################################################################
    print("Sweeping RF split and random seeds...")
    seed_results = Parallel(n_jobs=-1)(delayed(eval_rf_seeds)(split_seed=ss, rf_seed=rs) for ss in tqdm(split_seed_range, desc='Sweeping split seeds') for rs in rf_seed_range)
    best_r2 = -1;  best_split_seed = 0;  best_rf_seed = 0
    for r2, ss, rs in seed_results:
        if r2 > best_r2:
            best_r2 = r2;  best_split_seed = ss;  best_rf_seed = rs
    ########################################################################################
    print(f"Best split_seed: {best_split_seed} | Best rf_seed: {best_rf_seed} | Best R2: {round(best_r2, 3)}%")
    ########################################################################################
    p = dict(split_seed=best_split_seed, rf_seed=best_rf_seed, max_depth=10, num_trees=50, max_features=len(predictive_features), print_str=print_str)
    ########################################################################################
    str01 = f"sampling_method:  {sampling_method}"
    str02 = f"best_split_seed:  {p['split_seed']}"
    str03 = f"best_rf_seed:     {p['rf_seed']}"
    str04 = f"max_depth:        {p['max_depth']}"
    str05 = f"num_trees:        {p['num_trees']}"
    str06 = f"max_features:     {p['max_features']}"
    for s in [str01, str02, str03, str04, str05, str06]: print(s)
    ########################################################################################
    _, _, _, _, str07, str08 = random_forest_prod_prediction(split_seed=p['split_seed'], random_seed=p['rf_seed'], max_depth=p['max_depth'], num_trees_rf=p['num_trees'], max_features=p['max_features'], plot_print=True, plot_min=0, plot_max=1, print_str=p['print_str'], path_db=path_db, ArrayVals=ArrayVals, sampling_method=sampling_method, parameters_rf=parameters_rf)
    ########################################################################################
    output = pd.DataFrame([str01, str02, str03, str04, str05, str06, str07, str08], columns=["Results"])
    output.to_csv(path_db + '/RF_Results.csv', sep='\t')
    ########################################################################################
    return p

############################################################################################
def run_single_seed(rnd00=0, run_sampling_split=True, p={}, path_db='', ArrayVals=[], sampling_method='Random', parameters_rf=[]):
    ########################################################################################
    """
    :param rnd00:               Random seed to evaluate
    :param run_sampling_split:  Whether to sweep the split seed or the rf seed
    :param p:                   Method params dict containing seeds and hyperparameters
    :param path_db:             Path to save output figures
    :param ArrayVals:           Cluster arrays
    :param sampling_method:     Sampling method string
    :param parameters_rf:       Array of feature column names including target
    :return:                    Tuple of (r2, var_exp, rmse, mape)
    """
    ########################################################################################
    if run_sampling_split:
        r2, var_exp, rmse, mape, _, _ = random_forest_prod_prediction(
            split_seed=rnd00,          random_seed=p['rf_seed'],
            max_depth=p['max_depth'],  num_trees_rf=p['num_trees'],
            max_features=p['max_features'], plot_print=False,
            plot_min=0, plot_max=1,    print_str=p['print_str'],
            path_db=path_db, ArrayVals=ArrayVals,
            sampling_method=sampling_method, parameters_rf=parameters_rf)
    else:
        r2, var_exp, rmse, mape, _, _ = random_forest_prod_prediction(
            split_seed=p['split_seed'], random_seed=rnd00,
            max_depth=p['max_depth'],   num_trees_rf=p['num_trees'],
            max_features=p['max_features'], plot_print=False,
            plot_min=0, plot_max=1,     print_str=p['print_str'],
            path_db=path_db, ArrayVals=ArrayVals,
            sampling_method=sampling_method, parameters_rf=parameters_rf)
    ########################################################################################
    return r2, var_exp, rmse, mape
############################################################################################
def sweep_hyperparameter(param_values=[], fixed_depth=[], fixed_trees=[], mode='depth', desc='', p={}, path_db='', ArrayVals=[], sampling_method='', parameters_rf=[]):
    ########################################################################################
    """
    :param param_values:    Array of values to sweep over
    :param fixed_depth:     Fixed max_depth when sweeping trees
    :param fixed_trees:     Fixed num_trees when sweeping depth
    :param mode:            'depth' or 'trees'
    :param desc:            tqdm progress bar label
    :param p:               Method params dict containing seeds and hyperparameters
    :param path_db:         Path to save output figures
    :param ArrayVals:       Cluster arrays
    :param sampling_method: Sampling method string
    :param parameters_rf:   Array of feature column names including target
    :return:                Tuple of (ve_array, rmse_array)
    """
    ########################################################################################
    def run(val=0):
        depth = int(val) if mode == 'depth' else fixed_depth
        trees = int(val) if mode == 'trees' else fixed_trees
        _, ve, rmse, _, _, _ = random_forest_prod_prediction(
            split_seed=p['split_seed'], random_seed=p['rf_seed'],
            max_depth=depth,            num_trees_rf=trees,
            max_features=p['max_features'], plot_print=False,
            plot_min=0, plot_max=1,     print_str=p['print_str'],
            path_db=path_db, ArrayVals=ArrayVals,
            sampling_method=sampling_method, parameters_rf=parameters_rf)
        return ve, rmse
    ########################################################################################
    return zip(*Parallel(n_jobs=-1)(delayed(run)(v) for v in tqdm(param_values, desc=desc)))

############################################################################################
def hyperparameter_tuning(p={}, path_db='', ArrayVals=[], sampling_method='Random', parameters_rf=[], run_sampling_split=True, run_test=False):
    ########################################################################################
    """
    :param p:                   Method params dict containing seeds and hyperparameters
    :param path_db:             Path to save output figures
    :param ArrayVals:           Cluster arrays
    :param sampling_method:     Sampling method string
    :param parameters_rf:       Array of feature column names including target
    :param run_sampling_split:  Whether to sweep the split seed or the rf seed
    :param run_test:            Whether to run a short test sweep or the full sweep
    :return:                    df_hpt_rf, max_depth_save_state, num_trees_save_state
    """
    ########################################################################################
    # Seed sweep
    rnd_seed_iter = np.arange(0, 51, 1) if run_test else np.arange(0, 10001, 1)
    results       = Parallel(n_jobs=-1)(delayed(run_single_seed)(rnd00=rnd, run_sampling_split=run_sampling_split, p=p, path_db=path_db, ArrayVals=ArrayVals, sampling_method=sampling_method, parameters_rf=parameters_rf) for rnd in tqdm(rnd_seed_iter, desc='Sweeping random seeds'))
    results_array = np.array(results)
    fit_r2_ary   = results_array[:, 0];  fit_ve_ary   = results_array[:, 1]
    fit_rmse_ary = results_array[:, 2];  fit_mape_ary = results_array[:, 3]
    ########################################################################################
    df_hpt_rf = pd.DataFrame({
        'optimal random state': [rnd_seed_iter[np.argmax(fit_r2_ary)],   rnd_seed_iter[np.argmax(fit_ve_ary)],
                                 rnd_seed_iter[np.argmin(fit_rmse_ary)],  rnd_seed_iter[np.argmin(fit_mape_ary)]],
        'metric value':         [np.round(np.max(fit_r2_ary), 3),        np.round(np.max(fit_ve_ary), 3),
                                 np.round(np.min(fit_rmse_ary), 3),       np.round(np.min(fit_mape_ary), 3)]},
        index=['R$^2$ Value', 'Exp_Var Value', 'RMSE Value', 'MAPE Value'])
    ########################################################################################
    # Depth and tree sweep
    max_depths     = np.arange(1, 46)
    ve_depths, rmse_depths = sweep_hyperparameter(param_values=max_depths, fixed_depth=[], fixed_trees=p['num_trees'], mode='depth', desc='Sweeping max_depth', p=p, path_db=path_db, ArrayVals=ArrayVals, sampling_method=sampling_method, parameters_rf=parameters_rf)
    ve_depths, rmse_depths = np.array(list(ve_depths)), np.array(list(rmse_depths))
    optimal_depth_ve       = np.argmax(ve_depths);    optimal_depth_rmse   = np.argmin(rmse_depths)
    max_depth_save_state   = int(max_depths[optimal_depth_ve])
    exp_var_max_max_depths = np.round(np.max(ve_depths), 3);  rmse_min_max_depths = np.round(np.min(rmse_depths), 3)
    ########################################################################################
    num_trees      = np.arange(1, 201)
    ve_trees, rmse_trees = sweep_hyperparameter(param_values=num_trees, fixed_depth=max_depth_save_state, fixed_trees=[], mode='trees', desc='Sweeping num_trees', p=p, path_db=path_db, ArrayVals=ArrayVals, sampling_method=sampling_method, parameters_rf=parameters_rf)
    ve_trees, rmse_trees = np.array(list(ve_trees)), np.array(list(rmse_trees))
    optimal_trees_ve     = np.argmax(ve_trees);       optimal_trees_rmse   = np.argmin(rmse_trees)
    num_trees_save_state = int(num_trees[optimal_trees_ve])
    exp_var_max_num_trees = np.round(np.max(ve_trees), 3);    rmse_min_num_trees = np.round(np.min(rmse_trees), 3)
    ########################################################################################
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    ########################################################################################
    for ax_main, x_vals, ve_vals, rmse_vals, opt_ve, opt_rmse, xlabel, xlim, title_suffix in [
        (axes[0], max_depths, ve_depths, rmse_depths, optimal_depth_ve,  optimal_depth_rmse,
         'Maximum Tree Depth',       (0, 45),  'Maximum Tree Depth'),
        (axes[1], num_trees,  ve_trees,  rmse_trees,  optimal_trees_ve, optimal_trees_rmse,
         'Number of Trees Averaged', (0, 200), 'Number of Trees'),
    ]:
        ####################################################################################
        ax_twin = ax_main.twinx()
        ax_main.arrow(x_vals[opt_ve],   0, 0, 100.1, width=0.02, color='red',  head_length=0, head_width=0)
        ax_twin.arrow(x_vals[opt_rmse], 0, 0, 100.1, width=0.02, color='blue', head_length=0, head_width=0)
        ax_main.scatter(x_vals, ve_vals,   c='red',  alpha=0.2, linewidths=0.3, edgecolors='black')
        ax_twin.scatter(x_vals, rmse_vals, c='blue', alpha=0.2, linewidths=0.3, edgecolors='black')
        ax_main.set_title(f"{p['print_str']} Sampling: Testing Model Performance vs. {title_suffix}")
        ax_main.set_xlabel(xlabel);  ax_main.set_ylabel('Testing Variance Explained')
        ax_twin.set_ylabel('Testing RMSE');  ax_main.set_xlim(xlim)
        ax_main.set_ylim(50, 100);  ax_twin.set_ylim(0, 10)
    ########################################################################################
    fig.tight_layout()
    fig.savefig(path_db + '/RF_Hyperparameter_Tuning.png', bbox_inches='tight')
    ########################################################################################
    print(f"Results achieved with max_depth of:                  {max_depth_save_state}")
    print(f"Results achieved with random forest random_seed of:  {p['rf_seed']}")
    print(f"Results achieved with train-test random_seed of:     {p['split_seed']}")
    print(f"Results achieved with num_trees_rf of:               {num_trees_save_state}")
    print(f"Results achieved with max_features of:               {p['max_features']}")
    print(f"\nFor max tree depth     - max exp variance: {exp_var_max_max_depths}%  | min RMSE: {rmse_min_max_depths}%")
    print(f"For num trees averaged - max exp variance: {exp_var_max_num_trees}% | min RMSE: {rmse_min_num_trees}%")
    ########################################################################################
    return df_hpt_rf, max_depth_save_state, num_trees_save_state

############################################################################################
def plot_split_histogram(ax=None, ax_twin=None, train_data=[], test_data=[], title='', xlim=(), colors=[]):
    ########################################################################################
    """
    :param ax:         Main axis
    :param ax_twin:    Twin axis
    :param train_data: Training data array
    :param test_data:  Testing data array
    :param title:      Plot title
    :param xlim:       X-axis limits
    :param colors:     List of two colors [train, test]
    :return: None
    """
    ########################################################################################
    n, bins, _ = ax.hist([train_data, test_data])
    ax.cla()
    width     = (bins[1] - bins[0]) * 0.4
    train_bar = ax.bar(bins[:-1],              n[0], width, align='edge', color=colors[0], label='Training Data')
    test_bar  = ax_twin.bar(bins[:-1] + width, n[1], width, align='edge', color=colors[1], label='Testing Data')
    ax.set_ylabel("Frequency", color=colors[0]);       ax_twin.set_ylabel("Frequency", color=colors[1])
    ax.tick_params('y', colors=colors[0]);             ax_twin.tick_params('y', colors=colors[1])
    ax.set_title(title);                               ax.set_xlim(xlim)
    ax.legend((train_bar, test_bar), ('Training Data', 'Testing Data'), loc='upper right', fontsize=9)

############################################################################################
def plot_rf_results(df=pd.DataFrame(), path_db='', ArrayVals=[], sampling_method='Random', parameters_rf=[], p={}, min_plot=-1.5, max_plot=1.5):
    ########################################################################################
    """
    :param df:              Engineered dataset
    :param path_db:         Path to save output figures
    :param ArrayVals:       Cluster arrays
    :param sampling_method: Sampling method string
    :param parameters_rf:   Array of feature column names including target
    :param p:               Method params dict
    :param min_plot:        Minimum x-axis value for predictive feature histograms
    :param max_plot:        Maximum x-axis value for predictive feature histograms
    :return:                X_train20, X_test20, y_train20, y_test20
    """
    ########################################################################################
    if sampling_method == 'Random':
        pred_norm     = df[[c for c in df.columns if c != 'Production']].copy()
        response_norm = df[['Production']].copy()
        X_train2, X_test2, y_train2, y_test2 = train_test_split(pred_norm, response_norm, test_size=0.3, random_state=p['split_seed'])
        X_train20 = np.asarray(X_train2.to_numpy());  X_test20  = np.asarray(X_test2.to_numpy())
        y_train20 = np.asarray(y_train2.to_numpy());  y_test20  = np.asarray(y_test2.to_numpy())
    else:
        training_data_new0, testing_data_new0 = training_testing_datasets1(rnd_state=p['split_seed'], shuffle_bool=False, test_size=0.3, ArrayVals=ArrayVals, sample_method=sampling_method)
        traindata_rf0        = np.asarray(training_data_new0[parameters_rf].to_numpy())
        testdata_rf0         = np.asarray(testing_data_new0[parameters_rf].to_numpy())
        X_train20, y_train20 = traindata_rf0[:, :-1], traindata_rf0[:, -1]
        X_test20,  y_test20  = testdata_rf0[:, :-1],  testdata_rf0[:, -1]
    ########################################################################################
    print(X_train20.shape, X_test20.shape)
    ########################################################################################
    # Training/testing histograms
    pred_cols  = list(parameters_rf[:-1])
    n_features = len(pred_cols)
    n_cols     = 3
    n_rows     = int(np.ceil((n_features + 1) / n_cols))
    fig, axes  = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes_flat  = axes.flatten()
    colors     = ['b', 'r']
    ########################################################################################
    for i, col in enumerate(pred_cols):
        ax = axes_flat[i];  ax_twin = ax.twinx()
        plot_split_histogram(ax=ax, ax_twin=ax_twin, train_data=X_train20[:, i], test_data=X_test20[:, i], title=col, xlim=(min_plot, max_plot), colors=colors)
    ########################################################################################
    ax = axes_flat[n_features];  ax_twin = ax.twinx()
    y_train_flat = y_train20.flatten() if y_train20.ndim > 1 else y_train20
    y_test_flat  = y_test20.flatten()  if y_test20.ndim  > 1 else y_test20
    plot_split_histogram(ax=ax, ax_twin=ax_twin, train_data=y_train_flat, test_data=y_test_flat, title='Production', xlim=(0.0, 1.0), colors=colors)
    for j in range(n_features + 1, len(axes_flat)): axes_flat[j].set_visible(False)
    fig.suptitle(f'Random Forest Model - {p["print_str"]} Sampling - Training and Testing Dataset Distributions', fontsize=18, y=1.0125)
    fig.tight_layout()
    fig.savefig(path_db + '/RF_Histograms.png', bbox_inches='tight')
    ########################################################################################
    # Spatial scatter plots
    lat         = df['Latitude'];  long = df['Longitude'];  cum_oil_365 = df['Production']
    lat_col_idx  = list(parameters_rf[:-1]).index('Latitude')
    long_col_idx = list(parameters_rf[:-1]).index('Longitude')
    cum_oil_p05  = np.percentile(cum_oil_365.values, 5)
    cum_oil_p95  = np.percentile(cum_oil_365.values, 95)
    scatter_kwargs = dict(cmap=plt.cm.plasma, vmin=cum_oil_p05, vmax=cum_oil_p95, alpha=0.8, linewidths=0.8, edgecolors="black")
    ########################################################################################
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, lat_vals, long_vals, prod_vals, split in zip(
        axes,
        [X_train20[:, lat_col_idx],          X_test20[:, lat_col_idx]],
        [X_train20[:, long_col_idx],         X_test20[:, long_col_idx]],
        [y_train20.ravel(),                  y_test20.ravel()],
        ['Training',                         'Testing']
    ):
        ####################################################################################
        cax  = ax.scatter(lat_vals, long_vals, c=prod_vals, **scatter_kwargs)
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label('Production', rotation=270, size=12, labelpad=20)
        cbar.ax.tick_params(labelsize=10)
        ax.set_xlabel('Latitude', size=10);  ax.set_ylabel('Longitude', size=10)
        ax.set_title(f"RF - {split} Production Given {p['print_str']} Sampling", size=10)
        ax.set_xlim([lat.min(), lat.max()]);  ax.set_ylim([long.min(), long.max()])
    ########################################################################################
    fig.tight_layout()
    fig.savefig(path_db + '/RF_Production_ScatterPlot.png', bbox_inches='tight')
    ########################################################################################
    return X_train20, X_test20, y_train20, y_test20

############################################################################################