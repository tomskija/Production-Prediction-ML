############################################################################################
def standardInputWrapper(inJson={}):
    """
    Generically unpacks any JSON with the standard params/tables structure.
    :param inJson: Raw JSON input dict
    :return:       Flat inputData dict — access any value via inputData['key']
    """
    ########################################################################################
    inputData = {}
    inputData.update({item['name']: item['value']  for item in inJson['data']['params']})
    inputData.update({item['name']: item['values'] for item in inJson['data']['tables']})
    ########################################################################################
    return inputData
############################################################################################