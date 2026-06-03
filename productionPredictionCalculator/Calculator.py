###############################################################################
import json
import os
import mlflow
from os.path import dirname, join
from utils.inputWrapper import standardizedInputWrapper
from utils.mlFlowConfig import setupMLFlow
from utils.utils import (
    checkData,
    feature_engineering,
    loadCleanDataAndSetOutputDirectory,
    analyze_and_select_features,
    dynamicallyPickClustering,
    find_best_rf_seeds,
    hyperparameter_tuning,
    plot_rf_results,
    callSharpAnalysis,
)

###############################################################################
def calculate(inJson={}, localTesing=False):
    ###########################################################################
    setupMLFlow()
    ###########################################################################
    try:
        inputData   = standardizedInputWrapper(inJson=inJson)
        inputData   = checkData(inputData=inputData)
        df, path_db = loadCleanDataAndSetOutputDirectory(localTesing=localTesing)
        #######################################################################
        with mlflow.start_run(run_name=f"pipeline_{inputData['name']}"):
            mlflow.set_tag('sampling_method', 'TBD')
            mlflow.log_params({
                'name':                 inputData['name'],
                'target_feature':       inputData['target_feature'],
                'run_test':             inputData['run_test'],
                'run_sampling_split':   inputData['run_sampling_split'],
                'auto_select_features': inputData['auto_select_features'],
                'mi_threshold':         inputData['mi_threshold'],
                'variance_threshold':   inputData['variance_threshold'],
            })
            df = feature_engineering(df_data1=df, min_pred_norm=inputData['min_pred_norm'], max_pred_norm=inputData['max_pred_norm'], min_target_norm=inputData['min_target_norm'], max_target_norm=inputData['max_target_norm'], path_db=path_db, plot=inputData['plot'])
            df, selected_features, selection_mode     = analyze_and_select_features(df=df, path_db=path_db, target_feature=inputData['target_feature'], predictive_features=inputData['predictive_features'], auto_select_features=inputData['auto_select_features'], mi_threshold=inputData['mi_threshold'], variance_threshold=inputData['variance_threshold'], random_state=inputData['fi_random_state'], n_estimators=inputData['fi_n_estimators'], max_depth=inputData['fi_max_depth'], max_features=inputData['fi_max_features'], plot=inputData['plot'])
            mlflow.log_param('feature_selection_mode', selection_mode)
            mlflow.log_param('selected_features', str(selected_features))
            parameters_rf = selected_features + [inputData['target_feature']]
            df, ArrayVals, best_method  = dynamicallyPickClustering(df=df, path_db=path_db, n_range=inputData['n_range'], kmeans_random_state_range=inputData['kmeans_random_state_range'], gmm_random_state_range=inputData['gmm_random_state_range'])
            mlflow.set_tag('sampling_method', best_method)
            p = find_best_rf_seeds(predictive_features=selected_features, target_feature=inputData['target_feature'], ArrayVals=ArrayVals, sampling_method=best_method, parameters_rf=parameters_rf, path_db=path_db, split_seed_range=inputData['split_seed_range'], rf_seed_range=inputData['rf_seed_range'])
            _, max_depth_save_state, num_trees_save_state = hyperparameter_tuning(p=p, path_db=path_db, ArrayVals=ArrayVals, sampling_method=best_method, parameters_rf=parameters_rf, run_sampling_split=inputData['run_sampling_split'], run_test=inputData['run_test'])
            p['max_depth'] = max_depth_save_state
            p['num_trees'] = num_trees_save_state
            X_train20, X_test20, y_train20, y_test20 = plot_rf_results(df=df, path_db=path_db, ArrayVals=ArrayVals, sampling_method=best_method, parameters_rf=parameters_rf, p=p, min_plot=inputData['min_plot'], max_plot=inputData['max_plot'])
            callSharpAnalysis(inputData={**inputData, 'predictive_features': selected_features}, p=p, X_train20=X_train20, X_test20=X_test20, y_train20=y_train20, path_db=path_db, best_method=best_method)
            if os.path.exists(path_db): mlflow.log_artifacts(path_db, artifact_path='figures')
            print("Completed Calculation")
    except Exception as e:
        print("\nError Message: " + str(e) + "\n")
        return str(e)
    ###########################################################################
    return "Success"

###############################################################################
def main():
    with open(join(dirname(__file__), "tests/testOrg.json")) as json_file:
        inJson = json.load(json_file)
    response = calculate(inJson=inJson, localTesing=True)
    print(response)

###############################################################################
if __name__ == "__main__": main()
###############################################################################