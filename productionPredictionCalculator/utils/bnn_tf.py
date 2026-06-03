############################################################################################
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

############################################################################################
class BNN:
    """
    Generic Bayesian Neural Network — N inputs, M outputs, configurable hidden layers.
    Uses TensorFlow/Keras for forward pass and GradientTape for Langevin gradients.
    """
    ########################################################################################
    def __init__(self, topology=[1, 9, 1]):
        """
        :param topology: List [n_inputs, n_hidden, n_outputs] — extend to more hidden layers if needed
        """
        self.topology = topology
        self.model    = self.build_model()

    ########################################################################################
    def build_model(self):
        """
        Build Keras Sequential model from topology.
        """
        layers = [keras.layers.Input(shape=(self.topology[0],))]
        for i in range(1, len(self.topology) - 1):
            layers.append(keras.layers.Dense(self.topology[i], activation='sigmoid'))
        layers.append(keras.layers.Dense(self.topology[-1], activation='sigmoid'))
        model = keras.Sequential(layers)
        return model

    ########################################################################################
    def encode(self):
        """
        Flatten all weights and biases into a single 1D numpy vector.
        """
        return np.concatenate([w.numpy().ravel() for w in self.model.trainable_weights])

    ########################################################################################
    def decode(self, w=None):
        """
        Load a flat numpy weight vector back into the model weights.
        """
        idx = 0
        for weight in self.model.trainable_weights:
            size  = weight.numpy().size
            shape = weight.numpy().shape
            weight.assign(tf.constant(w[idx:idx + size].reshape(shape), dtype=tf.float32))
            idx  += size

    ########################################################################################
    def predict(self, x_np=None):
        """
        Run forward pass on a numpy array, return numpy array.
        """
        x   = tf.constant(x_np, dtype=tf.float32)
        out = self.model(x, training=False)
        return out.numpy()

    ########################################################################################
    def langevin_gradient(self, x_np=None, y_np=None, w=None, lr=0.01, steps=1):
        """
        Apply Langevin gradient steps using TensorFlow GradientTape.
        Returns updated flat weight vector.

        :param x_np:  Input features numpy array (n_samples, n_inputs)
        :param y_np:  Target values numpy array (n_samples, n_outputs)
        :param w:     Current flat weight vector
        :param lr:    Gradient step size (Langevin learning rate)
        :param steps: Number of gradient update steps
        :return:      Updated flat weight vector
        """
        self.decode(w=w)
        x         = tf.constant(x_np, dtype=tf.float32)
        y         = tf.constant(y_np, dtype=tf.float32)
        optimizer = keras.optimizers.SGD(learning_rate=lr)
        for _ in range(steps):
            with tf.GradientTape() as tape:
                pred = self.model(x, training=True)
                loss = tf.reduce_mean(tf.square(pred - y))
            grads = tape.gradient(loss, self.model.trainable_weights)
            optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return self.encode()

############################################################################################
class MCMCSampler:
    """
    Metropolis-Hastings MCMC sampler with optional Langevin gradient proposals.
    Uses tensorflow_probability.distributions.Normal for log likelihood and log prior.
    """
    ########################################################################################
    def __init__(self, bnn=None, use_langevin=True, langevin_prob=0.5, sigma_sq=25.0):
        """
        :param bnn:           BNN instance
        :param use_langevin:  Whether to use Langevin gradient proposals
        :param langevin_prob: Probability of using Langevin proposal vs random walk
        :param sigma_sq:      Prior variance for weights
        """
        self.bnn          = bnn
        self.use_langevin = use_langevin
        self.l_prob       = langevin_prob
        self.sigma_sq     = sigma_sq

    ########################################################################################
    @staticmethod
    def metrics(predictions=None, targets=None):
        """
        Compute R2, RMSE, MAPE for a set of predictions and targets.
        """
        predictions = predictions.ravel()
        targets     = targets.ravel()
        ss_res      = np.sum((targets - predictions) ** 2)
        ss_tot      = np.sum((targets - np.mean(targets)) ** 2)
        r2          = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        rmse        = float(np.sqrt(np.mean((predictions - targets) ** 2)))
        safe_tgt    = np.where(targets != 0, targets, np.mean(targets))
        mape        = float(np.mean(np.abs((targets - predictions) / safe_tgt)) * 100)
        return r2, rmse, mape

    ########################################################################################
    def log_likelihood(self, x=None, y=None, w=None, tau_sq=None):
        """
        Gaussian log likelihood using tfp.distributions.Normal.
        p(y | w, tau) = Normal(pred, sqrt(tau_sq))
        """
        pred    = self.bnn.predict(x_np=x)
        dist    = tfd.Normal(loc=pred.ravel().astype(np.float32), scale=np.float32(tau_sq ** 0.5))
        log_lik = float(tf.reduce_sum(dist.log_prob(y.ravel().astype(np.float32))).numpy())
        return log_lik, pred

    ########################################################################################
    def log_prior(self, w=None, tau_sq=None):
        """
        Log prior using tfp.distributions.Normal for weights and Jeffreys prior for tau.
        p(w) = Normal(0, sqrt(sigma_sq))
        p(tau) ~ 1/tau  (Jeffreys)
        """
        w_dist    = tfd.Normal(loc=np.float32(0.0), scale=np.float32(self.sigma_sq ** 0.5))
        w_prior   = float(tf.reduce_sum(w_dist.log_prob(w.astype(np.float32))).numpy())
        tau_prior = -float(np.log(tau_sq))
        return w_prior + tau_prior

    ########################################################################################
    def sample(self, x_train=None, y_train=None, x_test=None, y_test=None, n_samples=400, burn_in=0.85, w_step=0.02, tau_step=0.01, lr=0.01, langevin_steps=1, verbose=False):
        """
        Run MCMC sampling.

        :param x_train:        Training features (numpy, n x n_inputs)
        :param y_train:        Training targets  (numpy, n x n_outputs)
        :param x_test:         Test features
        :param y_test:         Test targets
        :param n_samples:      Total MCMC samples
        :param burn_in:        Burn-in fraction (0.85 = discard first 85%)
        :param w_step:         Weight proposal step size
        :param tau_step:       Tau proposal step size
        :param lr:             Langevin gradient learning rate
        :param langevin_steps: Number of gradient steps per Langevin proposal
        :param verbose:        Print progress
        :return:               Results dict
        """
        ########################################################################################
        n_weights        = len(self.bnn.encode())
        burnin_idx       = int(burn_in * n_samples)
        ########################################################################################
        pos_w            = np.zeros((n_samples, n_weights))
        pos_tau          = np.zeros(n_samples)
        fx_train_samples = np.zeros((n_samples, len(x_train)))
        fx_test_samples  = np.zeros((n_samples, len(x_test)))
        r2_train_arr     = np.zeros(n_samples)
        r2_test_arr      = np.zeros(n_samples)
        rmse_train_arr   = np.zeros(n_samples)
        rmse_test_arr    = np.zeros(n_samples)
        mape_train_arr   = np.zeros(n_samples)
        mape_test_arr    = np.zeros(n_samples)
        ########################################################################################
        w              = np.random.randn(n_weights) * 0.1
        log_tau        = 0.0
        tau_sq         = np.exp(log_tau)
        self.bnn.decode(w=w)
        log_lik, pred_train = self.log_likelihood(x=x_train, y=y_train, w=w, tau_sq=tau_sq)
        _, pred_test        = self.log_likelihood(x=x_test,  y=y_test,  w=w, tau_sq=tau_sq)
        log_post            = log_lik + self.log_prior(w=w, tau_sq=tau_sq)
        n_accept            = 0
        n_langevin          = 0
        ########################################################################################
        for i in range(n_samples):
            if self.use_langevin and np.random.rand() < self.l_prob:
                w_grad      = self.bnn.langevin_gradient(x_np=x_train, y_np=y_train, w=w.copy(), lr=lr, steps=langevin_steps)
                w_prop      = np.random.normal(w_grad, w_step)
                w_prop_grad = self.bnn.langevin_gradient(x_np=x_train, y_np=y_train, w=w_prop.copy(), lr=lr, steps=langevin_steps)
                wc_delta    = w      - w_prop_grad
                wp_delta    = w_prop - w_grad
                diff_prop   = (-0.5 * np.sum(wc_delta ** 2) / w_step + 0.5 * np.sum(wp_delta ** 2) / w_step)
                n_langevin += 1
            else:
                w_prop    = w + np.random.normal(0, w_step, n_weights)
                diff_prop = 0.0
            log_tau_prop = log_tau + np.random.normal(0, tau_step)
            tau_sq_prop  = np.exp(log_tau_prop)
            ########################################################################################
            self.bnn.decode(w=w_prop)
            log_lik_prop, pred_train_prop = self.log_likelihood(x=x_train, y=y_train, w=w_prop, tau_sq=tau_sq_prop)
            _,            pred_test_prop  = self.log_likelihood(x=x_test,  y=y_test,  w=w_prop, tau_sq=tau_sq_prop)
            log_post_prop = log_lik_prop + self.log_prior(w=w_prop, tau_sq=tau_sq_prop)
            log_alpha     = min(0.0, log_post_prop - log_post + diff_prop)
            ########################################################################################
            if np.log(np.random.rand() + 1e-300) < log_alpha:
                w          = w_prop
                log_tau    = log_tau_prop
                tau_sq     = tau_sq_prop
                log_post   = log_post_prop
                pred_train = pred_train_prop
                pred_test  = pred_test_prop
                n_accept  += 1
            ########################################################################################
            pos_w[i]             = w
            pos_tau[i]           = tau_sq
            fx_train_samples[i]  = pred_train.ravel()
            fx_test_samples[i]   = pred_test.ravel()
            r2, rmse, mape       = self.metrics(predictions=pred_train, targets=y_train)
            r2_train_arr[i]      = r2;  rmse_train_arr[i] = rmse;  mape_train_arr[i] = mape
            r2, rmse, mape       = self.metrics(predictions=pred_test,  targets=y_test)
            r2_test_arr[i]       = r2;  rmse_test_arr[i]  = rmse;  mape_test_arr[i]  = mape
            ########################################################################################
            if verbose and (i + 1) % max(1, n_samples // 10) == 0:
                print(f"  Sample {i+1}/{n_samples} | Train R2: {r2_train_arr[i]*100:.2f}% | Test R2: {r2_test_arr[i]*100:.2f}% | Accept: {n_accept/(i+1)*100:.1f}%")
        ########################################################################################
        accept_rate = n_accept / n_samples * 100
        if verbose:
            print(f"\nAcceptance rate : {accept_rate:.2f}%")
            print(f"Langevin count  : {n_langevin}")
        ########################################################################################
        return {
            'pos_w':       pos_w,            'pos_tau':     pos_tau,
            'fx_train':    fx_train_samples, 'fx_test':     fx_test_samples,
            'r2_train':    r2_train_arr,     'r2_test':     r2_test_arr,
            'rmse_train':  rmse_train_arr,   'rmse_test':   rmse_test_arr,
            'mape_train':  mape_train_arr,   'mape_test':   mape_test_arr,
            'burnin_idx':  burnin_idx,       'accept_rate': accept_rate,
            'n_langevin':  n_langevin,
        }

############################################################################################
def plot_bnn_results_tf(results={}, y_train=[], y_test=[], path_db='', sampling_method=''):
    """
    Plot BNN MCMC convergence metrics and uncertainty bands.
    Saves figures to path_db.

    :param results:          Output dict from MCMCSampler.sample()
    :param y_train:          Training targets numpy array
    :param y_test:           Test targets numpy array
    :param path_db:          Output directory path
    :param sampling_method:  Label string for plot titles
    """
    ########################################################################################
    burnin_idx = results['burnin_idx']
    x_train    = np.linspace(0, 1, len(y_train))
    x_test     = np.linspace(0, 1, len(y_test))
    ########################################################################################
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'BNN MCMC Convergence — {sampling_method}', fontsize=14)
    for ax, train_vals, test_vals, title, ylabel in [
        (axes[0, 0], results['r2_train'],   results['r2_test'],   'R²',               'R²'),
        (axes[0, 1], results['rmse_train'], results['rmse_test'], 'RMSE',             'RMSE'),
        (axes[1, 0], results['mape_train'], results['mape_test'], 'MAPE',             'MAPE (%)'),
        (axes[1, 1], results['r2_train'],   results['r2_test'],   'R² (post burn-in)','R²'),
    ]:
        ax.plot(train_vals, color='red',   alpha=0.7, label='Train')
        ax.plot(test_vals,  color='green', alpha=0.7, label='Test')
        if 'post burn-in' in title: ax.axvline(x=burnin_idx, color='gray', linestyle='--', label=f'Burn-in ({burnin_idx})')
        ax.set_title(title, size=11); ax.set_xlabel('Sample iteration'); ax.set_ylabel(ylabel); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(path_db, f'BNN_TF_Convergence_{sampling_method}.png'), bbox_inches='tight', dpi=150)
    ########################################################################################
    fx_train_post = results['fx_train'][burnin_idx:]
    fx_test_post  = results['fx_test'][burnin_idx:]
    mu_train      = fx_train_post.mean(axis=0);  p10_train = np.percentile(fx_train_post, 10, axis=0);  p90_train = np.percentile(fx_train_post, 90, axis=0)
    mu_test       = fx_test_post.mean(axis=0);   p10_test  = np.percentile(fx_test_post,  10, axis=0);  p90_test  = np.percentile(fx_test_post,  90, axis=0)
    ########################################################################################
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'BNN Uncertainty Bands — {sampling_method}', fontsize=13)
    for ax, x, y_true, mu, p10, p90, split in [
        (axes[0], x_train, y_train.ravel(), mu_train, p10_train, p90_train, 'Training'),
        (axes[1], x_test,  y_test.ravel(),  mu_test,  p10_test,  p90_test,  'Testing'),
    ]:
        ax.plot(x, y_true, color='black', label='Actual',   linewidth=1.5)
        ax.plot(x, mu,     color='blue',  label='BNN Mean', linewidth=1.2)
        ax.plot(x, p10,    color='gray',  label='P10',      linewidth=0.8, linestyle='--')
        ax.plot(x, p90,    color='gray',  label='P90',      linewidth=0.8, linestyle='--')
        ax.fill_between(x, p10, p90, alpha=0.3, color='yellow', label='P10-P90 band')
        ax.set_title(f'{split} Data', size=11); ax.set_xlabel('Sample index (normalized)'); ax.set_ylabel('Predicted value'); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(path_db, f'BNN_TF_Uncertainty_{sampling_method}.png'), bbox_inches='tight', dpi=150)
    ########################################################################################
    print(f"\nBNN (TF) Results — {sampling_method} (post burn-in samples {burnin_idx}+):")
    for metric, train_arr, test_arr, scale in [
        ('R²',   results['r2_train'],   results['r2_test'],   100),
        ('RMSE', results['rmse_train'], results['rmse_test'],  100),
        ('MAPE', results['mape_train'], results['mape_test'],    1),
    ]:
        tr_mean = np.mean(train_arr[burnin_idx:]) * scale;  tr_std = np.std(train_arr[burnin_idx:]) * scale
        te_mean = np.mean(test_arr[burnin_idx:])  * scale;  te_std = np.std(test_arr[burnin_idx:])  * scale
        print(f"  {metric:<6} Train: {tr_mean:.3f}% ± {tr_std:.3f}% | Test: {te_mean:.3f}% ± {te_std:.3f}%")
    print(f"  Acceptance rate : {results['accept_rate']:.2f}%")
    print(f"  Langevin count  : {results['n_langevin']}")
############################################################################################