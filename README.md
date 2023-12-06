# Multi-task Learning (MTL) and T0 Model Experiments

This README provides instructions on how to set up environments and run fine-tuning scripts for MTL datasets as well as for the T0 model.

## Setting Up for T0

The T0 model requires a specific setup. To install the T0 module, follow these steps:

1. Install the T0 module by running:
    ```bash
    pip install -e .
    ```
   
2. For applications that require the original seqio tasks used for massively multitask fine-tuning, install additional requirements with:
    ```bash
    pip install -e .[seqio_tasks]
    ```

3. To run an experiment with the T0 model, use the following command:
    ```bash
    python run_t_zero.py \
        --dataset_name super_glue \
        --dataset_config_name rte \
        --template_name "must be true" \
        --model_name_or_path bigscience/T0_3B \
        --output_dir ./debug
    ```

## Setting Up for MTL Datasets

For experiments involving NLI and PI datasets, ensure the following dependencies are satisfied:

- Python 3.6
- MXNet 1.6.0 (for CUDA 10.0, install with `pip install mxnet-cu100`)
- GluonNLP 0.9.0

### Running Experiments on MTL Datasets

To train on MNLI and test on MNLI's development set and HANS, use the following command:

```bash
make train-bert exp=mnli_seed/bert task=MNLI test-split=dev_matched bs=32 gpu=0 \
    nepochs=3 seed=2 lr=0.00002
```

To train on QQP and test on QQP's development set and PAWS, run:

```bash
make train-bert exp=mnli_seed/bert task=QQP test-split=dev bs=32 gpu=0 \
    nepochs=3 seed=2 lr=0.00002
```

## Notes

- Replace `mnli_seed/bert` with your experiment name.
- Adjust `bs=32` (batch size) if needed based on your GPU memory.
- `gpu=0` indicates using the first GPU; adjust if necessary.
- The `seed` parameter can be set to any integer for reproducibility.
- `lr` is the learning rate; modify according to your model's requirements.

Please ensure your environment is correctly set up with the necessary dependencies before running the experiments.