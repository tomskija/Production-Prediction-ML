############################################################################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import mlflow
import mlflow.sklearn
import mlflow.tensorflow

############################################################################################
def setupMLFlow(experiment_name='production-prediction-rf'):
    """
    Configure MLflow tracking URI and experiment.
    Reads MLFLOW_TRACKING_URI from environment (set by Docker Compose),
    falls back to localhost for local dev outside the container.

    :param experiment_name: MLflow experiment name to log runs under
    """
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name  # add this
    os.environ['MLFLOW_TRACKING_URI']    = tracking_uri     # add this
    print(f"MLflow tracking URI : {tracking_uri}")
    print(f"MLflow experiment   : {experiment_name}")

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