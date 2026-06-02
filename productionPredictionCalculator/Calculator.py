##########################################################
import json
import os
import mlflow
import pandas as pd
from utils.mlFlowConfig import setupMLFlow
from utils.utils import (
    checkData,
    load_and_clean_data,
    setup_output_directory,
    feature_engineering,
    correlation_analysis,
    feature_information,
    dynamicallyPickClustering,
    find_best_rf_seeds,
    hyperparameter_tuning,
    plot_rf_results,
    callSharpAnalysis,
)
from os.path import dirname, join

##########################################################
def calculate(df=[], inJson={}):
    ######################################################
    setupMLFlow()
    ######################################################
    # try:
    inputData, parameters_rf = checkData(inJson=inJson)
    path_db = setup_output_directory(name=inputData['name'], problemfolder_db=inputData['problemfolder_db'])
    df_data1 = load_and_clean_data(df=df)
    ff
    ######################################################
    with mlflow.start_run(run_name=f"pipeline_{inputData['name']}"):
        mlflow.set_tag('sampling_method', 'TBD')
        mlflow.log_params({
            'name':               inputData['name'],
            'target_feature':     inputData['target_feature'],
            'pred_features':      str(inputData['predictive_features']),
            'run_test':           inputData['run_test'],
            'run_sampling_split': inputData['run_sampling_split'],
        })
        df = feature_engineering(df_data1=df, min_pred_norm=inputData['min_pred_norm'], max_pred_norm=inputData['max_pred_norm'], min_target_norm=inputData['min_target_norm'], max_target_norm=inputData['max_target_norm'], path_db=path_db, plot=inputData['plot'])
        correlation, rank_correlation, rank_correlation_pval, rank_correlation_scatter = correlation_analysis(df=df, path_db=path_db)
        featureImportanceArray, mutalImformationArray = feature_information(df=df, path_db=path_db, random_state=inputData['fi_random_state'], n_estimators=inputData['fi_n_estimators'], max_depth=inputData['fi_max_depth'], max_features=inputData['fi_max_features'])
        print()
        print("NEED TO SEE HOW TO USE THIS TO VERIFY MY INPUT FOR PRED FEATURES ARE SOLID")
        print()
        df, ArrayVals, best_method, _ = dynamicallyPickClustering(df=df, path_db=path_db, n_range=inputData['n_range'], kmeans_random_state_range=inputData['kmeans_random_state_range'], gmm_random_state_range=inputData['gmm_random_state_range'])
        mlflow.set_tag('sampling_method', best_method)
        p = find_best_rf_seeds(predictive_features=inputData['predictive_features'], target_feature=inputData['target_feature'], ArrayVals=ArrayVals, sampling_method=best_method, parameters_rf=parameters_rf, path_db=path_db, split_seed_range=inputData['split_seed_range'], rf_seed_range=inputData['rf_seed_range'])
        _, max_depth_save_state, num_trees_save_state = hyperparameter_tuning(p=p, path_db=path_db, ArrayVals=ArrayVals, sampling_method=best_method, parameters_rf=parameters_rf, run_sampling_split=inputData['run_sampling_split'], run_test=inputData['run_test'])
        p['max_depth'] = max_depth_save_state
        p['num_trees'] = num_trees_save_state
        X_train20, X_test20, y_train20, y_test20 = plot_rf_results(df=df, path_db=path_db, ArrayVals=ArrayVals, sampling_method=best_method, parameters_rf=parameters_rf, p=p, min_plot=inputData['min_plot'], max_plot=inputData['max_plot'])
        callSharpAnalysis(inputData=inputData, p=p, X_train20=X_train20, X_test20=X_test20, y_train20=y_train20, path_db=path_db, best_method=best_method)
        if os.path.exists(path_db):
            mlflow.log_artifacts(path_db, artifact_path='figures')
        print("Completed Calculation")
    # except Exception as e:
    #     print("\nError Message: " + str(e) + "\n")
    #     return str(e)
    ######################################################
    return "Success"

##########################################################
def main():
    thisDirName = dirname(__file__)
    fileName    = "Data/Unconventional_Synthetic_Dataset.csv"
    with open(join(thisDirName, "tests/testOrg.json")) as json_file:
        inJson = json.load(json_file)
    response = calculate(df=pd.read_csv(join(thisDirName, fileName)), inJson=inJson)
    print(response)

##########################################################
if __name__ == "__main__": main()
##########################################################