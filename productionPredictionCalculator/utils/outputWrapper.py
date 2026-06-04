############################################################################################
import json
import os
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
    ########################################################################################
    def dump_json(self, fileName=''):
        """
        Serialize output_wrapper to JSON file — used for test fixture generation.
        """
        os.makedirs(os.path.dirname(fileName), exist_ok=True)
        data = {'params': self.params, 'tables': self.tables}
        with open(fileName, 'w') as f:
            json.dump(data, f, indent=4)
    ########################################################################################
    # def dump_flattened_data(self):
    #     """Flatten params and tables into a single dict — used for test assertions."""
    #     result = {}
    #     for p in self.params:
    #         result[p['name']] = p['value']
    #     for t in self.tables:
    #         result[t['name']] = t['array']
    #     return result
############################################################################################