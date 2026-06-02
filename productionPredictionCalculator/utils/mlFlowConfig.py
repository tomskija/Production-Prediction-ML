############################################################################################
import os
import mlflow
import mlflow.sklearn

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
    import mlflow.tensorflow
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