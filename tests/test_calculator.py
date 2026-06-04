###############################################################################
import sys
import json
import os
import numpy as np
from os.path import abspath, join, dirname as d
from productionPredictionCalculator.Calculator import calculate
root_dir = d(d(abspath(__file__)))
sys.path.append(root_dir)
np.random.seed(0)

###############################################################################
def getFileString(fileName=''):
    with open(fileName) as f:
        return f.read()

############################################################################################
def dumpOutputWrapper(output_wrapper={}):
    """
    Flatten output_wrapper params and tables into a single comparable dict.
    """
    result = {}
    for p in output_wrapper.params:
        result[p['name']] = p['value']
    for t in output_wrapper.tables:
        result[t['name']] = t['array']
    return result

############################################################################################
def saveOutputFixture(output_dict={}, fileName=''):
    """
    Save output dict as JSON fixture for future comparison.
    """
    serializable = {}
    for k, v in output_dict.items():
        serializable[k] = v if not isinstance(v, (np.ndarray, np.generic)) else v.tolist()
    with open(fileName, 'w') as f:
        json.dump(serializable, f, indent=4)

############################################################################################
inFiles  = ["example_test1.json"]
outFiles = ["output_example_test1.json"]

############################################################################################
def test1():
    """
    Smoke test + output comparison for example_test1.json (run_bnn=0, localTesing=True).
    Runs the full RF pipeline and compares output against saved fixture.
    On first run, saves the fixture automatically.
    """
    ########################################################################################
    fileNameIn  = join(root_dir, "productionPredictionCalculator/tests/in_json/"  + inFiles[0])
    fileNameOut = join(root_dir, "productionPredictionCalculator/tests/out_json/" + outFiles[0])
    ########################################################################################
    with open(fileNameIn) as f: inJson = json.load(f)
    ########################################################################################
    output_wrapper = calculate(inJson=inJson, localTesing=True)
    ########################################################################################
    # Basic smoke assertions
    assert output_wrapper is not None, "calculate() returned None"
    assert not isinstance(output_wrapper, str), f"calculate() returned error: {output_wrapper}"
    assert len(output_wrapper.params) > 0, "output_wrapper has no params"
    assert len(output_wrapper.tables) > 0, "output_wrapper has no tables"
    ########################################################################################
    leftDict = dumpOutputWrapper(output_wrapper=output_wrapper)
    ########################################################################################
    # Save fixture on first run if it doesn't exist
    os.makedirs(d(fileNameOut), exist_ok=True)
    if not os.path.exists(fileNameOut):
        saveOutputFixture(output_dict=leftDict, fileName=fileNameOut)
        print(f"Fixture saved to {fileNameOut} — re-run to compare")
        return
    ########################################################################################
    # Load saved fixture and compare
    with open(fileNameOut) as f: rightDict = json.load(f)
    ########################################################################################
    varsToCheck = {
        'best_sampling_method': 'string',
        'selected_features':    'list',
    }
    ########################################################################################
    for var, varType in varsToCheck.items():
        assert var in leftDict,  f"'{var}' missing from output_wrapper"
        assert var in rightDict, f"'{var}' missing from fixture"
        if varType == 'string':
            assert leftDict[var] == rightDict[var], f"{var}: '{leftDict[var]}' != '{rightDict[var]}'"
        elif varType == 'list':
            assert sorted(leftDict[var]) == sorted(rightDict[var]), f"{var}: {leftDict[var]} != {rightDict[var]}"
    ########################################################################################
    print(f"test1 passed — best_sampling_method: {leftDict['best_sampling_method']}, " f"selected_features: {leftDict['selected_features']}")

############################################################################################