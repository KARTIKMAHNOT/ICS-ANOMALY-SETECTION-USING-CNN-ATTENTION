## CNN_ATTN Architecture

The proposed `CNN_ATTN` model extends the baseline CNN anomaly detector by integrating a feature-time attention mechanism for intrinsic explainability. The model first extracts temporal representations using stacked 1D convolutional layers over historical TEP process windows. An attention layer is then applied to learn the relative importance of different timesteps and process variables.

Unlike post-hoc explainability methods such as SHAP, SM, and LEMNA, the proposed model generates attribution scores directly from the network using attention-guided gradients:

Attribution = |Gradients| × Attention

This enables the model to simultaneously perform:

1. Forecasting-based anomaly detection
2. Intrinsic feature attribution

without requiring separate explainability pipelines.

The final attention-based attribution scores are used to rank process variables and identify the attacked sensor or actuator responsible for the anomaly.




## Dataset Setup

This project uses the Tennessee Eastman Process (TEP) dataset for anomaly detection and attribution experiments. Due to dataset size and licensing considerations, the dataset is not included in this repository.

Please download the TEP dataset separately and place the required files inside the `data/TEP/` directory before running the training or evaluation scripts.

After downloading, ensure the dataset structure matches the paths expected by the repository scripts.

##

## Novelty Pipeline

### 1. Train Attention-Based Model

```bash
python main_train.py CNN_ATTN TEP --train_params_epochs 10
```

### 2. Generate Intrinsic Attention Explanations

```bash
cd explain-eval-manipulations

python main_attention_explain_tep.py CNN_ATTN TEP cons_p2s_s1 \
--run_name results \
--num_samples 150
```

### 3. Evaluate Attribution Rankings

```bash
cd ..

python main_feature_properties_attention.py \
--md CNN_ATTN-TEP-l2-hist50-kern3-units64-results
```
