import numpy as np

import os
import pickle
import sys

sys.path.append('..')

# Ignore ugly futurewarnings from np vs tf.
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow.keras.backend as K

from data_loader import load_train_data
from main_train2 import load_saved_model

from utils import utils
from utils.tep_utils import load_tep_attack, get_skip_list


att_skip_list = get_skip_list()


def _normalize_scores(scores):
    """Normalize one score vector without changing its ranking shape."""
    scores = np.asarray(scores, dtype=np.float64)
    score_min = np.min(scores)
    score_max = np.max(scores)

    if score_max - score_min < 1e-12:
        return np.zeros_like(scores)

    return (scores - score_min) / (score_max - score_min)


def build_attention_error_explainer(event_detector):
    """Create a TF1-compatible sensor attribution function.

    The old CNN_ATTN runner differentiated mean(model output), which explains
    high forecasts rather than anomaly cause.  This explains forecast error per
    sensor and uses the attention layer to weight the input gradients over time.
    """
    input_tensor = event_detector.inner.input
    output_tensor = event_detector.inner.output

    y_true = K.placeholder(
        shape=(None, event_detector.params['nI']),
        name='cnn_attn_y_true'
    )

    residual_scores = K.square(output_tensor - y_true)
    mse_loss = K.mean(residual_scores)

    grads = K.gradients(
        mse_loss,
        input_tensor
    )[0]

    try:
        attention_tensor = event_detector.inner.get_layer(
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

    return K.function(
        [input_tensor, y_true],
        [residual_scores, grad_scores]
    )


def explain_true_position(
    event_detector,
    run_name,
    model_name,
    Xtest,
    footer,
    num_samples=100
):

    attack_start = 10000

    history = event_detector.params['history']

    explain_fn = build_attention_error_explainer(
        event_detector
    )

    # Save sensor attribution scores
    # Shape:
    # (num_samples, num_sensors)
    attn_outputs = np.zeros(
        (num_samples, Xtest.shape[1])
    )

    count = 0

    for i in range(num_samples):

        # =====================================================
        # BUILD ATTACK WINDOW
        # =====================================================

        att_start = attack_start + i - history - 1
        att_end = attack_start + i + 1

        Xattack = Xtest[att_start:att_end]

        Xattack_src, Yattack_src = (
            event_detector.transform_to_window_data(
                Xattack,
                Xattack
            )
        )

        print(
            f'For attack {footer}, '
            f'Processing {att_start} to {att_end}'
        )

        # =====================================================
        # ATTENTION-WEIGHTED FORECAST-ERROR ATTRIBUTION
        # =====================================================

        residual_scores, grad_scores = explain_fn([
            Xattack_src,
            Yattack_src
        ])

        # Per-sensor forecast error is the strongest signal for synthetic
        # manipulations; attention-weighted gradients break close ties.
        attn_outputs[i] = (
            _normalize_scores(residual_scores[0]) +
            0.25 * _normalize_scores(grad_scores[0])
        )

        count += 1

        if count >= num_samples:
            break

    # =====================================================
    # SAVE PICKLE
    # =====================================================

    save_path = (
        'explanations-dir/cnn-attn-pkl/'
        f'explanations-CNNATTN-'
        f'{model_name}-{run_name}-'
        f'{footer}-true{num_samples}.pkl'
    )

    pickle.dump(
        attn_outputs,
        open(save_path, 'wb')
    )

    print(f'Created {save_path}')

    return


def parse_arguments():

    parser = utils.get_argparser()

    parser.add_argument(
        'attack',
        help='Which attack to explore?',
        type=str
    )

    parser.add_argument(
        '--num_samples',
        default=150,
        type=int,
        help='Number of samples'
    )

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    model_type = args.model
    dataset_name = args.dataset
    attack_footer = args.attack

    os.chdir('..')

    if attack_footer in att_skip_list:
        print(
            f'{attack_footer} is in skip list. returning.....'
        )
        exit(0)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    run_name = args.run_name

    config = {}

    utils.update_config_model(
        args,
        config,
        model_type,
        dataset_name
    )

    model_name = config['name']

    # =====================================================
    # CREATE OUTPUT DIRECTORY
    # =====================================================

    os.makedirs(
        'explanations-dir/cnn-attn-pkl',
        exist_ok=True
    )

    # =====================================================
    # LOAD MODEL
    # =====================================================

    Xfull, sensor_cols = load_train_data(
        dataset_name
    )

    event_detector = load_saved_model(
        model_type,
        f'models/{run_name}/{model_name}.json',
        f'models/{run_name}/{model_name}.h5'
    )

    # =====================================================
    # LOAD ATTACK DATA
    # =====================================================

    Xtest, Ytest, sensor_cols = load_tep_attack(
        dataset_name,
        attack_footer
    )

    # =====================================================
    # GENERATE ATTENTION EXPLANATIONS
    # =====================================================

    explain_true_position(
        event_detector,
        run_name,
        model_name,
        Xtest,
        attack_footer,
        num_samples=args.num_samples
    )

    print('Finished!')
