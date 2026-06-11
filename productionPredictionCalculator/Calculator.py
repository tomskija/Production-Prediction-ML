###############################################################################
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings('ignore')
import json
import asyncio
import mlflow
from datetime import datetime
from os.path import dirname, join
from utils.inputWrapper import standardInputWrapper
from utils.outputWrapper import standardOutputWrapper
from utils.mlFlowConfig import setupMLFlow
from utils.dbConfig import DBConfig
from utils.bnn_pt import BNN as BNN_PT, MCMCSampler as MCMCSampler_PT, plot_bnn_results_pt
from utils.bnn_tf import BNN as BNN_TF, MCMCSampler as MCMCSampler_TF, plot_bnn_results_tf
from utils.utils import (
    checkData,
    feature_engineering,
    loadCleanDataAndSetOutputDirectory,
    analyzeAndSelectFeatures,
    dynamicallyPickClustering,
    find_best_rf_seeds,
    find_best_bnn_seeds,
    hyperparameter_tuning,
    plot_rf_results,
    callSharpAnalysis,
    callBNNSharpAnalysis,
    get_train_test_split,
)

###############################################################################
async def run_rf_pipeline(inputData={}, df=None, selected_features=[], ArrayVals=[], best_method='', parameters_rf=[], path_db='', run=None):
    """
    RF pipeline — independent from BNN.
    Finds own optimal split seed, tunes hyperparameters, trains, evaluates, runs SHAP.
    Runs concurrently with run_bnn_pipeline via asyncio.gather.
    """
    ###########################################################################
    loop = asyncio.get_event_loop()
    ###########################################################################
    # Find best RF split seed
    p = await loop.run_in_executor(None, lambda: find_best_rf_seeds(
        predictive_features=selected_features, target_feature=inputData['target_feature'],
        ArrayVals=ArrayVals, sampling_method=best_method, parameters_rf=parameters_rf,
        path_db=path_db, split_seed_range=inputData['split_seed_range'],
        rf_seed_range=inputData['rf_seed_range']
    ))
    ###########################################################################
    # Hyperparameter tuning
    _, p['max_depth'], p['num_trees'] = await loop.run_in_executor(None, lambda: hyperparameter_tuning(
        p=p, path_db=path_db, ArrayVals=ArrayVals, sampling_method=best_method,
        parameters_rf=parameters_rf, run_sampling_split=inputData['run_sampling_split'],
        run_test=inputData['run_test']
    ))
    ###########################################################################
    # Final RF fit + SHAP
    X_train, X_test, y_train, y_test = await loop.run_in_executor(None, lambda: plot_rf_results(
        df=df, path_db=path_db, ArrayVals=ArrayVals, sampling_method=best_method,
        parameters_rf=parameters_rf, p=p, min_plot=inputData['min_plot'], max_plot=inputData['max_plot']
    ))
    await loop.run_in_executor(None, lambda: callSharpAnalysis(
        inputData={**inputData, 'predictive_features': selected_features},
        p=p, X_train20=X_train, X_test20=X_test, y_train20=y_train,
        path_db=path_db, best_method=best_method
    ))
    ###########################################################################
    if mlflow.active_run():
        mlflow.log_params({'rf_split_seed': p['split_seed'], 'rf_rf_seed': p['rf_seed'],
                           'rf_max_depth': p['max_depth'], 'rf_num_trees': p['num_trees']})
    ###########################################################################
    return p, X_train, X_test, y_train, y_test

###############################################################################
async def run_bnn_pipeline(inputData={}, ArrayVals=[], selected_features=[], best_method='', parameters_rf=[], path_db='', run=None):
    """
    BNN pipeline — independent from RF.
    Finds own optimal split seed (full MCMC or fast forward pass gated by run_bnn_split_fast),
    trains BNN on that split, runs SHAP.
    Runs concurrently with run_rf_pipeline via asyncio.gather.
    """
    ###########################################################################
    loop    = asyncio.get_event_loop()
    BNN_cls     = BNN_TF        if inputData['bnn_library'] == 'tensorflow' else BNN_PT
    Sampler_cls = MCMCSampler_TF if inputData['bnn_library'] == 'tensorflow' else MCMCSampler_PT
    plot_fn     = plot_bnn_results_tf if inputData['bnn_library'] == 'tensorflow' else plot_bnn_results_pt
    ###########################################################################
    # Find best BNN split seed independently
    bnn_p = await loop.run_in_executor(None, lambda: find_best_bnn_seeds(
        predictive_features=selected_features, target_feature=inputData['target_feature'],
        ArrayVals=ArrayVals, sampling_method=best_method, parameters_rf=parameters_rf,
        path_db=path_db, split_seed_range=inputData['split_seed_range'],
        bnn_hidden_neurons=inputData['bnn_hidden_neurons'],
        bnn_library=inputData['bnn_library'],
        run_bnn_split_fast=inputData['run_bnn_split_fast']
    ))
    ###########################################################################
    # Get BNN-optimal train/test split
    X_train, X_test, y_train, y_test = await loop.run_in_executor(None, lambda: get_train_test_split(
        ArrayVals=ArrayVals, sampling_method=best_method, parameters_rf=parameters_rf,
        split_seed=bnn_p['split_seed'], predictive_features=selected_features,
        target_feature=inputData['target_feature']
    ))
    ###########################################################################
    # Full BNN training on optimal split
    bnn     = BNN_cls(topology=bnn_p['topology'])
    sampler = Sampler_cls(bnn=bnn, use_langevin=True, langevin_prob=0.5)
    bnn_results = await loop.run_in_executor(None, lambda: sampler.sample(
        x_train=X_train, y_train=y_train.reshape(-1, 1),
        x_test=X_test,   y_test=y_test.reshape(-1, 1),
        n_samples=inputData['bnn_n_samples'], burn_in=inputData['bnn_burn_in'], verbose=True
    ))
    ###########################################################################
    await loop.run_in_executor(None, lambda: plot_fn(
        results=bnn_results, y_train=y_train, y_test=y_test,
        path_db=path_db, sampling_method=best_method
    ))
    await loop.run_in_executor(None, lambda: callBNNSharpAnalysis(
        bnn=bnn, X_train20=X_train, X_test20=X_test, selected_features=selected_features,
        path_db=path_db, best_method=best_method, bnn_library=inputData['bnn_library']
    ))
    ###########################################################################
    if mlflow.active_run():
        mlflow.log_params({'bnn_split_seed': bnn_p['split_seed'], 'bnn_library': inputData['bnn_library']})
        mlflow.log_metrics({
            'bnn_r2_test':   float(bnn_results['r2_test'][bnn_results['burnin_idx']:].mean() * 100),
            'bnn_rmse_test': float(bnn_results['rmse_test'][bnn_results['burnin_idx']:].mean() * 100),
            'bnn_mape_test': float(bnn_results['mape_test'][bnn_results['burnin_idx']:].mean()),
            'bnn_accept':    float(bnn_results['accept_rate']),
        })
    ###########################################################################
    return bnn_p, bnn_results, X_train, X_test, y_train, y_test

###############################################################################
async def calculate_async(inJson={}, localTesing=False):
    ###########################################################################
    setupMLFlow()
    ###########################################################################
    try:
        inputData      = standardInputWrapper(inJson=inJson)
        output_wrapper = standardOutputWrapper()
        inputData      = checkData(inputData=inputData)
        db             = DBConfig(localTesing=localTesing)
        df, path_db    = loadCleanDataAndSetOutputDirectory(localTesing=localTesing, db=db)
        #######################################################################
        with mlflow.start_run(run_name=f"pipeline_{inputData['name']}") as run:
            mlflow.set_tag('sampling_method', 'TBD')
            mlflow.log_params({
                'name': inputData['name'], 'target_feature': inputData['target_feature'],
                'run_test': inputData['run_test'], 'run_sampling_split': inputData['run_sampling_split'],
                'auto_select_features': inputData['auto_select_features'],
                'mi_threshold': inputData['mi_threshold'], 'variance_threshold': inputData['variance_threshold'],
                'run_bnn': inputData['run_bnn'], 'run_bnn_split_fast': inputData['run_bnn_split_fast']
            })
            ###################################################################
            # Shared steps — feature engineering, selection, clustering
            df = feature_engineering(df_data1=df, min_pred_norm=inputData['min_pred_norm'], max_pred_norm=inputData['max_pred_norm'], min_target_norm=inputData['min_target_norm'], max_target_norm=inputData['max_target_norm'], path_db=path_db, plot=inputData['plot'])
            df, selected_features, selection_mode = analyzeAndSelectFeatures(df=df, path_db=path_db, target_feature=inputData['target_feature'], predictive_features=inputData['predictive_features'], auto_select_features=inputData['auto_select_features'], mi_threshold=inputData['mi_threshold'], variance_threshold=inputData['variance_threshold'], random_state=inputData['fi_random_state'], n_estimators=inputData['fi_n_estimators'], max_depth=inputData['fi_max_depth'], max_features=inputData['fi_max_features'], plot=inputData['plot'])
            mlflow.log_param('feature_selection_mode', selection_mode)
            mlflow.log_param('selected_features',      str(selected_features))
            parameters_rf = selected_features + [inputData['target_feature']]
            df, ArrayVals, best_method = dynamicallyPickClustering(df=df, path_db=path_db, n_range=inputData['n_range'], kmeans_random_state_range=inputData['kmeans_random_state_range'], gmm_random_state_range=inputData['gmm_random_state_range'])
            mlflow.set_tag('sampling_method', best_method)
            ###################################################################
            # RF and BNN run independently in parallel from this point
            if inputData['run_bnn']:
                (p, X_train_rf, X_test_rf, y_train_rf, y_test_rf), \
                (bnn_p, bnn_results, X_train_bnn, X_test_bnn, y_train_bnn, y_test_bnn) = \
                    await asyncio.gather(
                        run_rf_pipeline(inputData=inputData, df=df, selected_features=selected_features, ArrayVals=ArrayVals, best_method=best_method, parameters_rf=parameters_rf, path_db=path_db, run=run),
                        run_bnn_pipeline(inputData=inputData, ArrayVals=ArrayVals, selected_features=selected_features, best_method=best_method, parameters_rf=parameters_rf, path_db=path_db, run=run)
                    )
            else:
                p, X_train_rf, X_test_rf, y_train_rf, y_test_rf = await run_rf_pipeline(
                    inputData=inputData, df=df, selected_features=selected_features,
                    ArrayVals=ArrayVals, best_method=best_method,
                    parameters_rf=parameters_rf, path_db=path_db, run=run
                )
            ###################################################################
            if os.path.exists(path_db): mlflow.log_artifacts(path_db, artifact_path='figures')
            ###################################################################
            db.insert_run_results({
                'mlflow_run_id':        run.info.run_id,
                'timestamp':            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'status':               'complete',
                'selection_mode':       selection_mode,
                'selected_features':    str(selected_features),
                'best_sampling_method': best_method,
                'max_depth':            p['max_depth'],
                'num_trees':            p['num_trees'],
                'max_features':         p['max_features'],
                'split_seed':           p['split_seed'],
                'rf_seed':              p['rf_seed'],
            })
            ###################################################################
            output_wrapper.add_param(name='best_sampling_method', value=best_method)
            output_wrapper.add_param(name='mlflow_run_id',        value=run.info.run_id)
            output_wrapper.add_table(name='selected_features',    array=selected_features)
        #######################################################################
        db.close()
        print("Completed Calculation")
        return output_wrapper
    except Exception as e:
        print("\nError Message: " + str(e) + "\n")
        return str(e)

###############################################################################
def calculate(inJson={}, localTesing=False):
    """Sync wrapper around calculate_async — maintains backward compatibility."""
    return asyncio.run(calculate_async(inJson=inJson, localTesing=localTesing))

###############################################################################
def main():
    ###########################################################################
    runLocalPytests = True
    thisDirName = dirname(__file__)
    ###########################################################################
    if runLocalPytests:
        testNum  = 1
        fileName = "../tests/in_jsons/example_test" + str(testNum) + ".json"
    else:
        fileName = "tests/testCase01.json"
    ###########################################################################
    fileDirName = join(thisDirName, fileName)
    ###########################################################################
    with open(fileDirName) as json_file:
        inJson = json.load(json_file)
    output_wrapper = calculate(inJson=inJson, localTesing=True)
    if runLocalPytests:
        try:
            output_wrapper.dump_json(join(thisDirName, "../tests/out_jsons/output_example_test" + str(testNum))+'.json')
        except:
            print("Had Error")

###############################################################################
if __name__ == "__main__": main()
###############################################################################