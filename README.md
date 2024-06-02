# COMP 542 Hexagons Project

This project involves training and evaluating various machine-learning models on the Hexagons dataset. The models include T5 for instruction simplification and DeBERTa for action label and abstraction level classification. This README provides instructions on how to set up, run, and understand the different scripts included in the project.

## Prerequisites

- **Python 3.7+**
- **pip** (Python package installer)

## Installation

### Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

### Install the required packages:

```bash
pip install -r requirements.txt
```

## Data Preprocessing 

Data preprocessing is done using the related notebook. Chatgpt API was also accessed through this notebook. For privacy reasons, the api-key is not present.

## T5 Models

### T5 Model Training on Hexagons Dataset

To train the T5 model, use the following command:

```bash
python T5_Training.py --data_file path/to/your/dataset.xlsx --include_abstraction_level --no_color --epochs 50
```

 ### Arguments

--data_file: Path to the dataset file (required).

--include_abstraction_level: Include abstraction levels in the input (optional).

--no_color: Use the dataset without color information (optional).

--epochs: Number of training epochs (default: 50).

## T5 Model Inference for Simplifying Instructions

### Running the Inference

To run the T5 model inference for simplifying instructions, use the following command:

```bash
python inference.py --device cuda --model_path path/to/your/model --batch_size 100 --show_time_remaining --input_file path/to/your/input.xlsx --output_file path/to/your/output.xlsx
```

### Arguments

--device: Device to run the model on (cuda or cpu). Defaults to cuda.

--model_path: Path to the pre-trained T5 model (required).

--batch_size: Batch size for processing. Defaults to 100.

--show_time_remaining: Show estimated time remaining for processing (optional).

--input_file: Path to the input Excel file (required).

--output_file: Path to save the output Excel file (required).


## Classification Models

### Classification Evaluation

To evaluate a pre-trained DeBERTa model, use the following command:

```bash
python classification_evaluation.py
```

## Training the Model

To train the DeBERTa model with abstraction level, use the following command:

```bash
python classificationbased-abstraction.py
```

To train the DeBERTa model for action label prediction, use the following command:

```bash
python classificationbased.py
```

To train the DeBERTa model for action label prediction without using color information, use the following command:

```bash
python classificationbased_nocolor.py
```

## Abstraction Level Classifiers (Seperate Goal from the previous part)

To train the DeBERTa model for abstraction level classification, use the following command:

```bash
python deberta_abs.py
```

To train the BiLSTM model for abstraction level classification, use the following command:

```bash
python lstm_abs.py
```

