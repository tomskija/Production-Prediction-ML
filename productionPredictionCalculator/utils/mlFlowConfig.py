############################################################################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from utils.azureConfig import getAzureMLTrackingURI

############################################################################################
def setupMLFlow(experiment_name='production-prediction-rf'):
    """
    Configure MLflow tracking URI and experiment.
    Priority:
        1. Azure ML  — if AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE are set
        2. Docker    — if MLFLOW_TRACKING_URI is set (e.g. http://mlflow:5000 via Docker Compose)
        3. Local     — falls back to http://localhost:5000

    :param experiment_name: MLflow experiment name to log runs under
    """
    ########################################################################################
    azure_uri    = getAzureMLTrackingURI()
    tracking_uri = azure_uri or os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    ########################################################################################
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name
    os.environ['MLFLOW_TRACKING_URI']    = tracking_uri
    ########################################################################################
    backend = 'Azure ML' if azure_uri else ('Docker' if 'mlflow' in tracking_uri else 'Local')
    print(f"MLflow tracking URI : {tracking_uri}")
    print(f"MLflow experiment   : {experiment_name}")
    print(f"MLflow backend      : {backend}")

############################################################################################
def enable_tf_autolog():
    """
    Enable MLflow TensorFlow/Keras autologging.
    Call once before model.fit() in any TensorFlow pipeline.
    Captures: model summary, per-epoch metrics, hyperparameters, saved model artifact.
    Lazy import — TF only loads when this function is explicitly called.
    """
    mlflow.tensorflow.autolog(
        log_models=True,
        log_input_examples=False,
        log_model_signatures=True,
    )
    print("MLflow TensorFlow autologging enabled")

############################################################################################
def enable_sklearn_autolog():
    """
    Enable MLflow sklearn autologging.
    Call once before model.fit() in any sklearn pipeline.
    Captures: params, metrics, feature importances, model artifact.
    """
    mlflow.sklearn.autolog(
        log_models=True,
        log_input_examples=False,
        log_model_signatures=True,
    )
    print("MLflow sklearn autologging enabled")
############################################################################################