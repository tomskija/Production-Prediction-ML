##########################################################
import asyncio
import pandas as pd
from utils.utils import (
    checkData,
    setup_output_directory,
    load_and_clean_data,
    feature_engineering,
    correlation_analysis,
    feature_information,
    plot_well_data,
    dynamicallyPickClustering,
    find_best_rf_seeds,
    hyperparameter_tuning,
    plot_rf_results,
)
from os.path import dirname, join

##########################################################
async def calculate(df=[]):
    ######################################################
    try:
        inputData = checkData()
        path_db = setup_output_directory(name=inputData['name'], problemfolder_db=inputData['problemfolder_db'])
        df = load_and_clean_data(df=df)
        df = feature_engineering(df_data1=df, min_pred_norm=inputData['min_pred_norm'], max_pred_norm=inputData['max_pred_norm'], min_target_norm=inputData['min_target_norm'], max_target_norm=inputData['max_target_norm'], path_db=path_db, plot=inputData['plot'])
        _, _, _, _ = correlation_analysis(df=df, path_db=path_db)
        feature_information(df=df, path_db=path_db, random_state=inputData['fi_random_state'], n_estimators=inputData['fi_n_estimators'], max_depth=inputData['fi_max_depth'], max_features=inputData['fi_max_features'])
        plot_well_data(df=df, path_db=path_db)
        df, ArrayVals, best_method, best_params = dynamicallyPickClustering(df=df, path_db=path_db, n_range=inputData['n_range'], kmeans_random_state_range=inputData['kmeans_random_state_range'], gmm_random_state_range=inputData['gmm_random_state_range'])
        parameters_rf = inputData['predictive_features'] + [inputData['target_feature']]
        p = find_best_rf_seeds(predictive_features=inputData['predictive_features'], target_feature=inputData['target_feature'], ArrayVals=ArrayVals, sampling_method=best_method, parameters_rf=parameters_rf, path_db=path_db, split_seed_range=inputData['split_seed_range'], rf_seed_range=inputData['rf_seed_range'])
        _, max_depth_save_state, num_trees_save_state = hyperparameter_tuning(p=p, path_db=path_db, ArrayVals=ArrayVals, sampling_method=best_method, parameters_rf=parameters_rf, run_sampling_split=inputData['run_sampling_split'], run_test=inputData['run_test'])
        p['max_depth'] = max_depth_save_state
        p['num_trees'] = num_trees_save_state
        plot_rf_results(df=df, path_db=path_db, ArrayVals=ArrayVals, sampling_method=best_method, parameters_rf=parameters_rf, p=p, min_plot=inputData['min_plot'], max_plot=inputData['max_plot'])
        print("Completed Calculation")
    except Exception as e:
        print("\nError Message: " + str(e) + "\n")
        return str(e)
    ######################################################
    return "Success"

##########################################################
async def main():
    thisDirName = dirname(__file__)
    fileName    = "Data/Unconventional_Synthetic_Dataset.csv"
    response    = await calculate(df=pd.read_csv(join(thisDirName, fileName)))
    print(response)

##########################################################
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    print("Done")
##########################################################