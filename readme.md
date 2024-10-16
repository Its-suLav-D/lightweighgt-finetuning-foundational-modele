# Fine-tuning Language Models with PEFT

This project demonstrates how to fine-tune a pre-trained language model using Parameter-Efficient Fine-Tuning (PEFT) techniques. It uses the IMDB dataset as an example application for text classification.

## Project Overview

The project showcases the following key steps:

1. Loading a pre-trained language model (GPT-2 in this case)
2. Adapting the model for a specific task (sequence classification)
3. Evaluating the initial model performance
4. Applying PEFT (specifically LoRA - Low-Rank Adaptation) to fine-tune the model
5. Evaluating the fine-tuned model and comparing its performance to the initial model

## Requirements

To run this project, you'll need:

- Python 3.10+
- PyTorch
- Transformers
- Datasets
- PEFT
- Evaluate
- NumPy

You can install the required packages using pip:

```
pip install torch transformers datasets peft evaluate numpy
```

## Usage

1. Clone the repository and navigate to the project directory.

2. Run the script in a Jupyter notebook or as a Python script. The main steps include:

   - Loading the pre-trained model and dataset
   - Tokenizing the data
   - Setting up and evaluating the initial model
   - Applying PEFT (LoRA) and fine-tuning the model
   - Evaluating the fine-tuned model and comparing its performance

## Key Components

- **Base Model**: The project uses GPT-2 as the foundational language model.
- **PEFT Method**: Low-Rank Adaptation (LoRA) is used for efficient fine-tuning.
- **Task**: The model is adapted for sequence classification.
- **Dataset**: The IMDB dataset is used as an example for binary text classification.
- **Evaluation**: The model's performance is evaluated using accuracy as the metric.

## Customization

You can customize this project by:

1. Changing the `model_name` to use a different pre-trained model.
2. Modifying the `load_dataset()` call to use a different dataset.
3. Adjusting the `LoraConfig` parameters to experiment with different PEFT settings.
4. Modifying the `TrainingArguments` to change training parameters like learning rate, batch size, or number of epochs.
5. Adapting the `compute_metrics` function for different evaluation metrics.

## Results

The script outputs the performance metrics (accuracy in this case) of both the initial model and the fine-tuned model, allowing for a direct comparison before and after applying PEFT.

## Files

- `main.py` (or Jupyter notebook): The main script that runs the entire pipeline
- `./peft_results/`: Directory containing the saved PEFT model weights