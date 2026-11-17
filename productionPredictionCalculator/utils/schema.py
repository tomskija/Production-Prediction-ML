############################################################################################
"""
Input form definition for the UI (served at GET /schema).

Defaults match tests/testCase01.json, except run_test, which defaults to the fast sweep
so a click in the UI doesn't kick off the full 10,000+ seed search by accident.
Field types:
    number, integer, boolean, string, select     sent as params
    list_number, list_string                     sent as tables
show_if hides a field unless another field has one of the listed values.
"""
############################################################################################
def field(name='', label='', type='number', default=None, help='', options=None, min=None, max=None, show_if=None):
    f = {'name': name, 'label': label or name, 'type': type, 'default': default,
         'kind': 'table' if type.startswith('list_') else 'param'}
    if help: f['help'] = help
    if options is not None: f['options'] = options
    if min is not None: f['min'] = min
    if max is not None: f['max'] = max
    if show_if: f['show_if'] = show_if
    return f

############################################################################################
def getSchema():
    bnn = {'run_bnn': [True, 1]}
    return {
        'name': 'ppml',
        'title': 'Production Prediction',
        'description': 'Random Forest and Bayesian Neural Network prediction of unconventional well production with spatially aware train/test splits.',
        'groups': [
            {'name': 'Run', 'fields': [
                field('target_feature', 'Target', 'string', 'Production'),
                field('predictive_features', 'Predictive features', 'list_string', ['Por', 'Brittle', 'Latitude', 'Longitude']),
                field('auto_select_features', 'Auto select features', 'boolean', False),
                field('run_test', 'Fast test sweep', 'boolean', True, help='Short sweep for development; turn off for the full production sweep'),
                field('run_sampling_split', 'Sweep sampling splits', 'boolean', True),
                field('plot', 'Save figures', 'boolean', True),
            ]},
            {'name': 'Normalization', 'fields': [
                field('min_pred_norm', 'Feature min', 'number', -1.3875),
                field('max_pred_norm', 'Feature max', 'number', 1.3875),
                field('min_target_norm', 'Target min', 'number', 0.005),
                field('max_target_norm', 'Target max', 'number', 0.995),
                field('min_plot', 'Plot min', 'number', -1.5),
                field('max_plot', 'Plot max', 'number', 1.5),
            ]},
            {'name': 'Feature Selection', 'fields': [
                field('mi_threshold', 'Mutual info threshold', 'number', 0.1, min=0),
                field('variance_threshold', 'PCA variance kept', 'number', 0.95, min=0, max=1),
                field('fi_random_state', 'RF random state', 'integer', 5195),
                field('fi_n_estimators', 'RF trees', 'integer', 100, min=1),
                field('fi_max_depth', 'RF max depth', 'integer', 25, min=1),
                field('fi_max_features', 'RF max features', 'integer', 3, min=1),
            ]},
            {'name': 'Clustering', 'fields': [
                field('n_range_start', 'Clusters from', 'integer', 2, min=2),
                field('n_range_end', 'Clusters to', 'integer', 30, min=3),
                field('kmeans_random_state_range_start', 'K-means seeds from', 'integer', 0),
                field('kmeans_random_state_range_end', 'K-means seeds to', 'integer', 50),
                field('gmm_random_state_range_start', 'GMM seeds from', 'integer', 0),
                field('gmm_random_state_range_end', 'GMM seeds to', 'integer', 50),
            ]},
            {'name': 'Random Forest Sweep', 'fields': [
                field('split_seed_range_start', 'Split seeds from', 'integer', 0),
                field('split_seed_range_end', 'Split seeds to', 'integer', 100),
                field('rf_seed_range_start', 'RF seeds from', 'integer', 0),
                field('rf_seed_range_end', 'RF seeds to', 'integer', 100),
            ]},
            {'name': 'Bayesian Neural Network', 'fields': [
                field('run_bnn', 'Run BNN', 'boolean', False),
                field('run_bnn_split_fast', 'Fast BNN split selection', 'boolean', False, show_if=bnn),
                field('bnn_library', 'Library', 'select', 'pytorch', options=['pytorch', 'tensorflow'], show_if=bnn),
                field('bnn_n_samples', 'MCMC samples', 'integer', 400, min=10, show_if=bnn),
                field('bnn_burn_in', 'Burn-in fraction', 'number', 0.85, min=0, max=0.99, show_if=bnn),
                field('bnn_hidden_neurons', 'Hidden neurons', 'list_number', [9], show_if=bnn),
            ]},
        ],
    }
############################################################################################
