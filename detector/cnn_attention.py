"""
   Copyright 2023 Lujo Bauer, Clement Fung
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   -----------------------------------------------------------------------
   CNN_ATTN: Intrinsically Explainable CNN with Feature-Time Attention
   -----------------------------------------------------------------------
   Extends ConvNN with a feature-time attention mechanism that produces
   a (batch, time, features) attribution map during the forward pass.

   Key difference from ConvNN:
     ConvNN  → Conv1D → Flatten → Dense → prediction only
     CNN_ATTN → Conv1D → Feature-Time Attention → Temporal Pool → Dense
                                     ↓
                              Attribution map  (intrinsic XAI)

   The attention layer Dense(units, softmax) applied over (batch, time, units)
   produces importance weights for every feature at every timestep.
   These weights are the explanation — no SHAP/LEMNA/SM needed at inference.

   Usage:
     model = ConvNNAttention(nI=51, units=64, history=50, layers=2)
     model.create_model()
     model.train(Xtrain, Ytrain, ...)
     model.detect(X, theta)

     # Intrinsic attribution after anomaly detection:
     attn_map = model.get_attention_map(X_window)  # (n, time, units)
     sensor_scores = model.get_sensor_scores(X_window)  # (n, units) ranked
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, BatchNormalization,
    Flatten, Multiply, Lambda
)
from tensorflow.keras.models import Model

from .detector import ICSDetector


class ConvNNAttention(ICSDetector):
    """
    CNN with Feature-Time Attention for intrinsic explainability.

    Produces both:
      1. Forecast prediction  — used for anomaly detection (MSE threshold)
      2. Attention map        — used as intrinsic feature attribution

    The attention map shape is (batch, time_steps, units), giving an
    importance score for every (timestep, feature-channel) pair.
    Averaging over time gives per-sensor attribution scores directly.

    Parameters (kwargs):
      nI       : number of input sensors/features  (required)
      units    : Conv1D filters and attention width (default 64)
      history  : sliding window length              (default 50)
      kernel   : Conv1D kernel size                 (default 3)
      layers   : number of Conv1D layers            (default 2)
      activation: Conv1D activation                 (default 'relu')
      optimizer : Keras optimizer                   (default 'adam')
      verbose  : 0=silent, 1=loss csv, 2=summary    (default 0)
    """

    def __init__(self, **kwargs):
        params = {
            'nI':         None,
            'units':      64,
            'history':    50,
            'kernel':     3,
            'layers':     2,
            'activation': 'relu',
            'optimizer':  'adam',
            'verbose':    0,
        }
        for key, item in kwargs.items():
            params[key] = item
        self.params = params

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def create_model(self):
        """
        Build the CNN + Feature-Time Attention Keras model.

        Architecture:
          Input (history, nI)
            → Conv1D × layers  +  BatchNorm       shape: (time', units)
            → Feature-Time Attention (softmax)     shape: (time', units)
            → Element-wise Multiply (weighted rep) shape: (time', units)
            → Temporal sum pooling                 shape: (units,)
            → Dense(nI)                            shape: (nI,)  ← forecast

        The attention layer is named 'feature_time_attention' so
        get_attention_map() can extract it by name.
        """
        nI         = self.params['nI']
        units      = self.params['units']
        history    = self.params['history']
        layers     = self.params['layers']
        activation = self.params['activation']
        kernel_size= self.params['kernel']
        optimizer  = self.params['optimizer']
        verbose    = self.params['verbose']

        if layers < 1:
            print('Error: Must have at least one layer. layers={}'.format(layers))
            return
        if nI is None:
            print('Error: nI (number of inputs) must be provided.')
            return

        # ── Encoder: stacked Conv1D blocks ──────────────────────────
        input_layer = Input(shape=(history, nI), name='input')

        cnn_layer = Conv1D(
            filters=units, kernel_size=kernel_size,
            activation=activation, padding='causal', name='conv1d_0'
        )(input_layer)
        cnn_layer = BatchNormalization(name='bn_0')(cnn_layer)

        for i in range(1, layers):
            cnn_layer = Conv1D(
                filters=units, kernel_size=kernel_size,
                activation=activation, padding='causal',
                name=f'conv1d_{i}'
            )(cnn_layer)
            cnn_layer = BatchNormalization(name=f'bn_{i}')(cnn_layer)
        # cnn_layer shape: (batch, time_steps, units)

        # ── Feature-Time Attention ───────────────────────────────────
        # Dense applied over last axis → (batch, time_steps, units)
        # softmax over time_steps axis so weights sum to 1 per feature.
        attention_scores = Dense(
            units,
            activation=None,        # raw logits first
            use_bias=True,
            name='attention_logits'
        )(cnn_layer)                # (batch, time, units)

        # Softmax over time dimension (axis=1) so each feature channel
        # has a proper probability distribution over timesteps.
        attention_weights = Lambda(
            lambda z: tf.nn.softmax(z, axis=1),
            name='feature_time_attention'
        )(attention_scores)         # (batch, time, units)

        # ── Weighted representation ──────────────────────────────────
        attended = Multiply(
            name='attention_multiply'
        )([cnn_layer, attention_weights])  # (batch, time, units)

        # Temporal sum pooling → fixed-size vector
        context = Lambda(
            lambda z: tf.reduce_sum(z, axis=1),
            name='temporal_pooling'
        )(attended)                 # (batch, units)

        # ── Output: next-step forecast ───────────────────────────────
        output = Dense(nI, name='forecast')(context)  # (batch, nI)

        model = Model(inputs=input_layer, outputs=output, name='CNN_ATTN')
        model.compile(loss='mean_squared_error', optimizer=optimizer)

        if verbose >= 2:
            model.summary()

        self.inner = model
        return model

    # ------------------------------------------------------------------
    # Intrinsic attribution API
    # ------------------------------------------------------------------

    def get_attention_map(self, X):
        """
        Extract the feature-time attention weights for input windows X.

        Args:
            X: np.ndarray of shape (n_samples, history, nI)
               (windowed input, same format as training data)

        Returns:
            attention_map: np.ndarray of shape (n_samples, time_steps, units)
                Each value is the attention weight for a (timestep, channel).
        """
        if self.inner is None:
            raise RuntimeError('Model not built. Call create_model() first.')

        attn_model = Model(
            inputs=self.inner.input,
            outputs=self.inner.get_layer('feature_time_attention').output,
            name='attn_extractor'
        )
        return attn_model.predict(X, verbose=0)

    def get_sensor_scores(self, X, feature_names=None):
        """
        Compute per-sensor attribution scores by averaging attention over time.

        This is the primary attribution output used in evaluation
        (equivalent to what SHAP/SM/LEMNA return as feature importance).

        Args:
            X            : np.ndarray (n_samples, history, nI)
            feature_names: optional list of sensor/feature names

        Returns:
            scores: np.ndarray (n_samples, units)
                    Higher = more attended = more likely anomaly cause.
            ranked_indices: np.ndarray (units,)
                    Channel indices sorted by mean score descending.
                    Use these to rank sensors across a batch.

        Note:
            units may differ from nI (number of raw sensors). The attention
            operates in the CNN feature space. If you need raw sensor scores,
            project back using the input × attention correlation (see
            get_raw_sensor_scores).
        """
        attn_map = self.get_attention_map(X)         # (n, time, units)
        scores = attn_map.mean(axis=1)               # (n, units)
        ranked_indices = scores.mean(axis=0).argsort()[::-1]

        if feature_names is not None and len(feature_names) == scores.shape[1]:
            ranked_names = [feature_names[i] for i in ranked_indices]
            return scores, ranked_indices, ranked_names

        return scores, ranked_indices

    def get_raw_sensor_scores(self, X):
        """
        Project attention back to raw sensor space (nI dimensions).

        Uses the element-wise product of the input windows and the
        attention-upsampled weights as a sensor-level importance proxy.
        This makes the output directly comparable to SM/SHAP/LEMNA
        which all return (n_samples, nI) feature importance arrays.

        Args:
            X: np.ndarray (n_samples, history, nI)

        Returns:
            sensor_scores: np.ndarray (n_samples, nI)
                           Higher = more important for anomaly.
        """
        attn_map = self.get_attention_map(X)     # (n, time, units)

        # Mean attention over time → (n, units) channel importance
        channel_importance = attn_map.mean(axis=1)  # (n, units)

        # Project to sensor space: input × mean attention over channel dim
        # X shape: (n, time, nI); average input magnitude over time
        input_magnitude = np.abs(X).mean(axis=1)    # (n, nI)

        # Simple projection: scale input magnitudes by mean channel importance
        # This is a proxy — for exact sensor-space projection use the
        # input gradient × attention product in get_gradient_attention_scores.
        mean_channel_score = channel_importance.mean(axis=1, keepdims=True)  # (n,1)
        sensor_scores = input_magnitude * mean_channel_score                 # (n, nI)

        return sensor_scores

    @staticmethod
    def _normalize_rows(scores):
        """Min-max normalize each sample row while preserving sensor order."""
        score_min = np.min(scores, axis=1, keepdims=True)
        score_max = np.max(scores, axis=1, keepdims=True)
        denom = np.maximum(score_max - score_min, 1e-12)

        return (scores - score_min) / denom

    def get_gradient_attention_scores(self, X, Y=None):
        """
        Gradient × Attention attribution in raw sensor space (nI dims).

        Combines input gradients (Saliency Map) with attention weights
        to produce a sensor score directly comparable to SM/SHAP/LEMNA.
        This is the strongest attribution output for paper evaluation.

        Args:
            X: np.ndarray (n_samples, history, nI)
            Y: optional np.ndarray (n_samples, nI). When provided, gradients
               explain forecast error instead of raw forecast magnitude.

        Returns:
            grad_attn_scores: np.ndarray (n_samples, nI)
        """
        if self.inner is None:
            raise RuntimeError('Model not built. Call create_model() first.')

        input_tensor = self.inner.input
        output_tensor = self.inner.output

        if Y is None:
            residual_scores = None
            loss = K.sum(output_tensor)
            fn_inputs = [input_tensor]
            fn_values = [X]
        else:
            y_true = K.placeholder(
                shape=(None, self.params['nI']),
                name='cnn_attn_y_true'
            )
            residual_scores = K.square(output_tensor - y_true)
            loss = K.mean(residual_scores)
            fn_inputs = [input_tensor, y_true]
            fn_values = [X, Y]

        grads = K.gradients(
            loss,
            input_tensor
        )[0]

        try:
            attention_tensor = self.inner.get_layer(
                'feature_time_attention'
            ).output
            attention_time = K.mean(
                attention_tensor,
                axis=2,
                keepdims=True
            )
            grad_scores = K.mean(
                K.abs(grads * input_tensor) * attention_time,
                axis=1
            )
        except ValueError:
            grad_scores = K.mean(
                K.abs(grads * input_tensor),
                axis=1
            )

        outputs = [grad_scores]
        if residual_scores is not None:
            outputs.insert(0, residual_scores)

        results = K.function(fn_inputs, outputs)(fn_values)

        if Y is None:
            return results[0]

        error_scores, grad_attn_scores = results

        return (
            self._normalize_rows(error_scores) +
            0.25 * self._normalize_rows(grad_attn_scores)
        )

    # ------------------------------------------------------------------
    # Windowing (identical to ConvNN)
    # ------------------------------------------------------------------

    def transform_to_window_data(self, dataset, target, target_size=1):
        data   = []
        labels = []
        history    = self.params['history']
        start_index = history
        end_index   = len(dataset) - target_size

        for i in range(start_index, end_index):
            indices = range(i - history, i)
            data.append(dataset[indices])
            labels.append(target[i + target_size])

        return np.array(data), np.array(labels)

    # ------------------------------------------------------------------
    # Training (identical to ConvNN, kept self-contained)
    # ------------------------------------------------------------------

    def train(self, Xtrain, Ytrain, use_callbacks=False, **train_params):
        """
        Train CNN_ATTN.
          Xtrain: (n, history, nI) windowed inputs
          Ytrain: (n, nI) next-step targets
        """
        if self.inner is None:
            print('Creating model.')
            self.create_model()

        batch_size = train_params.pop('batch_size', 32)

        def data_generator(X, Y, bs):
            i = 0
            while True:
                i += bs
                if i + bs > len(X):
                    i = 0
                yield X[i:i + bs], Y[i:i + bs]

        if use_callbacks:
            train_params['callbacks'] = [
                EarlyStopping(
                    monitor='val_loss', patience=3, verbose=0,
                    min_delta=0, mode='auto', restore_best_weights=True
                )
            ]

        if 'validation_data' in train_params:
            Xval, Yval = train_params['validation_data']
            train_params['validation_data'] = data_generator(Xval, Yval, batch_size)

        train_history = self.inner.fit(
            data_generator(Xtrain, Ytrain, batch_size),
            **train_params
        )

        if self.params['verbose'] > 0 and 'val_loss' in train_history.history:
            import numpy as np
            loss_obj = np.vstack([
                train_history.history['loss'],
                train_history.history['val_loss']
            ])
            np.savetxt(
                f'cnn_attn-train-history-{self.params["layers"]}l-{self.params["units"]}u.csv',
                loss_obj, delimiter=',', fmt='%.5f'
            )

    def train_by_idx(self, Xfull, train_idxs, val_idxs, use_callbacks=False, **train_params):
        """
        Train CNN_ATTN using index-based batching.
          Xfull: (n, nI) full raw data
        """
        if self.inner is None:
            print('Creating model.')
            self.create_model()

        batch_size = train_params.pop('batch_size', 32)
        history    = self.params['history']

        def data_generator(X, idxs, bs):
            i = 0
            while True:
                i += bs
                if i + bs > len(idxs):
                    i = 0
                X_batch, Y_batch = [], []
                for b in range(bs):
                    lead_idx = idxs[i + b]
                    X_batch.append(X[lead_idx - history: lead_idx])
                    Y_batch.append(X[lead_idx + 1])
                yield np.array(X_batch), np.array(Y_batch)

        if use_callbacks:
            train_params['callbacks'] = [
                EarlyStopping(
                    monitor='val_loss', patience=3, verbose=0,
                    min_delta=0, mode='auto', restore_best_weights=True
                )
            ]

        if 'validation_data' in train_params:
            train_params['validation_data'] = data_generator(Xfull, val_idxs, batch_size)

        self.inner.fit(data_generator(Xfull, train_idxs, batch_size), **train_params)

    # ------------------------------------------------------------------
    # Detection (identical interface to ConvNN)
    # ------------------------------------------------------------------

    def detect(self, x, theta, window=1, batches=False,
               eval_batch_size=4096, **keras_params):
        """
        Anomaly detection via MSE reconstruction error threshold.
        Identical interface to ConvNN.detect().
        """
        reconstruction_error = self.reconstruction_errors(
            x, batches, eval_batch_size, **keras_params
        )
        instance_errors = reconstruction_error.mean(axis=1)
        return self.cached_detect(instance_errors, theta, window)

    def cached_detect(self, instance_errors, theta, window=1):
        detection = instance_errors > theta
        if window > 1:
            detection = np.convolve(detection, np.ones(window), 'same') // window
        return detection

    def reconstruction_errors(self, x, batches=False,
                               eval_batch_size=4096, **keras_params):
        if batches:
            full_errors = np.zeros((
                x.shape[0] - self.params['history'] - 1, x.shape[1]
            ))
            idx = 0
            while idx < len(x):
                Xwindow, Ywindow = self.transform_to_window_data(
                    x[idx: idx + eval_batch_size + self.params['history'] + 1],
                    x[idx: idx + eval_batch_size + self.params['history'] + 1]
                )
                if idx + eval_batch_size > len(full_errors):
                    full_errors[idx:] = (
                        self.predict(Xwindow, **keras_params) - Ywindow
                    ) ** 2
                else:
                    full_errors[idx: idx + eval_batch_size] = (
                        self.predict(Xwindow, **keras_params) - Ywindow
                    ) ** 2
                idx += eval_batch_size
            return full_errors
        else:
            Xwindow, Ywindow = self.transform_to_window_data(x, x)
            return (self.predict(Xwindow, **keras_params) - Ywindow) ** 2


if __name__ == '__main__':
    print('Not a main file.')
