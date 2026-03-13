##########################################################
import json
import numpy as np


##########################################################
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


##########################################################
def writeToJson(data="", outFileName=""):
    jsonOut = json.dumps(
        data, cls=NumpyEncoder
    )  # indent=2, ensure_ascii=False, sort_keys=True)
    file1 = open(outFileName, "w")
    file1.write(jsonOut)
    file1.close()
    return


##########################################################
def generateInputFileDict():
    inputData = {}
    return inputData


##########################################################
