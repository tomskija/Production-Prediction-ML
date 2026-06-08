############################################################################################
import os

############################################################################################
def getAzureMLTrackingURI():
    """
    Build the Azure ML MLflow tracking URI from environment variables.
    Returns None if Azure ML env vars are not set — falls back to local MLflow.

    Required environment variables when using Azure ML:
        AZURE_SUBSCRIPTION_ID  : Azure subscription ID
        AZURE_RESOURCE_GROUP   : Azure resource group name
        AZURE_ML_WORKSPACE     : Azure ML workspace name

    :return: Azure ML tracking URI string or None
    """
    ########################################################################################
    subscription_id = os.environ.get('AZURE_SUBSCRIPTION_ID', '')
    resource_group  = os.environ.get('AZURE_RESOURCE_GROUP',  '')
    workspace_name  = os.environ.get('AZURE_ML_WORKSPACE',    '')
    if not all([subscription_id, resource_group, workspace_name]): return None
    return f"azureml://subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.MachineLearningServices/workspaces/{workspace_name}"
############################################################################################