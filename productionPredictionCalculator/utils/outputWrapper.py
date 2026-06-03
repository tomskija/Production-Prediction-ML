############################################################################################
class standardOutputWrapper:
    ########################################################################################
    def __init__(self):
        self.params = []
        self.tables = []
    ########################################################################################
    def add_param(self, name='', value=None, value_type=2):
        self.params.append({'name': name, 'value': value, 'value_type': value_type})
    ########################################################################################
    def add_table(self, name='', array=[], row_headers=[], column_headers=[], shape=[]):
        self.tables.append({'name': name, 'array': array, 'row_headers': row_headers, 'column_headers': column_headers, 'shape': shape if shape else [len(array)]})
############################################################################################