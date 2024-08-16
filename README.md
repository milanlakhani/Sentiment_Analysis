## Introduction

This project is about performing Sentiment Analysis on a set of movie reviews. The repository contains a simple model, `baseline.py` and a more complex model, `train.py`.

The database can be downloaded from here https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download , although in this case there is no need, since I have included a copy of the csv in the repository.


## Setup

The Python version used to make this was Python 3.12.3
To install the requirements for this project, after cloning navigate to the root directory on the command line, and run `pip install -r requirements.txt`.

You can download a checkpoint file, `LSTM-2_ckpt_epch_3.pth`, from this location, https://drive.google.com/file/d/19ix-uCPuUZ7dHrzJbesuq86ytv4QDQFi/view
You should add the checkpoint to the root directory of the repository.


## Training

You can set parameter values for all models in `config.yaml`.
These include:
 - batch_size (The amount of data processed at once)
 - embedding_dim: 400 (Size of vectors representing words)
 - hidden_dim: 256 (Size of hidden state vector)
 - n_layers: 2 (Number of LSTM Layers)
 - output_size: 1 (This should stay as one for a binary classification)
 - epochs: 3 (Number of training cycles)
 - print_every: 100 (Frequency of feedback from training model)
 - gradient_clipping: 5 (Gradient at which the gradient should be 'clipped' to prevent exploding gradients)
 - learning_rate: 0.0005 (Rate at which the model adjusts)
 - dropout_prob_1: 0.5 (Level of dropout in the early layers)
 - dropout_prob_2: 0.3 (Level of dropout in the late layers)
 - seq_length: 500 (Maximum length of a review in words before it gets truncated)
 - split_frac: 0.8 (Ratio of data used in the training set as opposed to validation or testing)
 - num_heads: 8 (Number of heads used in the attention mechanism)


## Wandb

If you want to record results of a model, near the top of the model file (`baseline.py` or `train.py`), update the value of `WANDB_API_KEY` from `None` to your API key. During training you will see results in project `SA1`.

I have generated some Wandb reports of runs with different hyperparameters, which you can find here https://api.wandb.ai/links/milanlkhn-city-university-of-london/zgdtg3jb


## Inference

Open the root folder, which contains the inference file, and add checkpoint `LSTM-2_ckpt_epch_3.pth` to the folder if you have not already, as detailed in the Setup section.

Run all the cells of the inference notebook in order. If using Google Colab, you will need to first save the directory in your drive, then at the top of this notebook add
```
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Sentiment_Analysis
```
assuming the last line is the correct location of the directory. Any hyperameters can be adjusted in config.yaml
