from collections import Counter
from string import punctuation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import wandb

wandb.login()

with open("config.yaml", 'r') as file:
    settings = yaml.safe_load(file)

# Hyperparameters
batch_size = settings["batch_size"]
embedding_dim = settings["embedding_dim"]
hidden_dim = settings["hidden_dim"]
n_layers = settings["n_layers"]
output_size = settings["output_size"]
epochs = settings["epochs"]
print_every = settings["print_every"]
gradient_clipping = settings["gradient_clipping"]
learning_rate = settings["learning_rate"]
dropout_prob_1 = settings["dropout_prob_1"]
dropout_prob_2 = settings["dropout_prob_2"]
seq_length = settings["seq_length"]
split_frac = settings["split_frac"]

run = wandb.init(
    project = "SA1",
    config = {
        "batch_size": batch_size,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "output_size": output_size,
        "epochs": epochs,
        "gradient_clipping": gradient_clipping,
        "learning_rate": learning_rate,
        "dropout_prob_1": dropout_prob_1,
        "dropout_prob_2": dropout_prob_2,
        "seq_length": seq_length,
        "split_frac": split_frac,
    },
)

def save_checkpoint(epoch, model, model_name, optimizer):
    ckpt = {'epoch': epoch, 'model_weights': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
    torch.save(ckpt, f"checkpoints/{model_name}_ckpt_epch_{str(epoch)}.pth")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_checkpoint(model, file_name, device=None, optimizer=None):
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(file_name, map_location=device)
    model_weights = ckpt['model_weights']
    model.load_state_dict(model_weights)
    print("Model's pretrained weights loaded!")
    if optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    print("Optimizer's state loaded!")

df = pd.read_csv("./IMDB Dataset.csv")

# Convert to lower case
df['review'] = df['review'].apply(lambda x:x.lower())

# Remove punctuation
df['clean_text'] = df['review'].apply(lambda x:''.join([c for c in x if c not in punctuation]))
df['len_review'] = df['clean_text'].apply(lambda x:len(x))

# Create list of reviews
review_list = df['clean_text'].tolist()

def get_vocab_int_dict(review_list):
    # Create a list of words
    review_list = ' '.join(review_list)
    words = review_list.split()
    # Count words using Counter Method
    count_words = Counter(words)
    # sorted_words = count_words.most_common(len(words))
    # vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
    vocab_to_int = {w:i+1 for i, (w,c) in enumerate(count_words.items())}
    return vocab_to_int

vocab_to_int = get_vocab_int_dict(review_list)

reviews_split = df['clean_text'].tolist()

# Tokenize — Encode the words
reviews_int = []
for review in reviews_split:
    r = [vocab_to_int[w] for w in review.split()]
    reviews_int.append(r)

# Tokenize — Encode the labels
labels_split = df['sentiment'].tolist()
encoded_labels = [1 if label =='positive' else 0 for label in labels_split]
encoded_labels = np.array(encoded_labels)

# Analyze Reviews Length
reviews_len = [len(x) for x in reviews_int]
pd.Series(reviews_len).hist()
plt.xlabel('Review Length (Number of Words)')
plt.ylabel('Frequency')
plt.title('Distribution of Review Lengths')
plt.show()
pd.Series(reviews_len).describe()
wandb.log({"Review lengths": wandb.Image(plt)})

# Remove reviews of 0 length
reviews_int = [ reviews_int[i] for i, l in enumerate(reviews_len) if l>0 ]
encoded_labels = [ encoded_labels[i] for i, l in enumerate(reviews_len) if l> 0 ]

# Padding / Truncating the remaining data
def pad_features(reviews_int, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    features = np.zeros((len(reviews_int), seq_length), dtype = int)

    for i, review in enumerate(reviews_int):
        review_len = len(review)

        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length-review_len))
            new = zeroes+review
        elif review_len > seq_length:
            new = review[0:seq_length]

        features[i,:] = np.array(new)

    return features

features = pad_features(reviews_int,seq_length)
len_feat = len(features)

# Training, Validation, Test Dataset Split
train_x = features[0:int(split_frac*len_feat)]
train_y = encoded_labels[0:int(split_frac*len_feat)]
remaining_x = features[int(split_frac*len_feat):]
remaining_y = encoded_labels[int(split_frac*len_feat):]
valid_x = remaining_x[0:int(len(remaining_x)*0.5)]
valid_y = remaining_y[0:int(len(remaining_y)*0.5)]
test_x = remaining_x[int(len(remaining_x)*0.5):]
test_y = remaining_y[int(len(remaining_y)*0.5):]

train_y = np.array(train_y)
test_y = np.array(test_y)
valid_y = np.array(valid_y)

# Dataloaders and Batching
# Create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# Obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = next(dataiter)
print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size()) # batch_size
print('Sample label: \n', sample_y)

# Define the LSTM Network Architecture

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob_1=0.5, drop_prob_2=0.3):
        super().__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob_1, batch_first=True)
        self.dropout = nn.Dropout(drop_prob_2)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()


    def forward(self, x, hidden):
        batch_size = x.size(0)

        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # Stack up LSTM outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # Dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # Sigmoid function
        sig_out = self.sig(out)

        # Reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels

        return sig_out, hidden


    def init_hidden(self, batch_size, train_on_gpu=False):

        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden

# Training the Network

# Instantiate the model w/ hyperparams
vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding
output_size = 1
model = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
print(model)

# First checking if GPU is available
train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

# Loss and optimization functions
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training params
counter = 0

# Move model to GPU, if available
if(train_on_gpu):
    model.cuda()

model.train()
for epoch in range(epochs):
    # Initialize hidden state
    h = model.init_hidden(batch_size, train_on_gpu)

    # Batch loop
    for inputs, labels in train_loader:
        counter += 1
        #print(counter)

        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # Zero accumulated gradients
        model.zero_grad()

        # Get the output from the model
        output, h = model(inputs, h)

        # Calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

        # Loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = model.init_hidden(batch_size, train_on_gpu)
            val_losses = []
            model.eval()
            for inputs, labels in valid_loader:

                # Variables for hidden state for backprop
                val_h = tuple([each.data for each in val_h])

                if(train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, val_h = model(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            model.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
    wandb.log({"epoch": epoch+1, "step": counter, "loss": loss.item(), "val loss": np.mean(val_losses)})
    print(f"Saving epoch {epoch+1} checkpoint")
    save_checkpoint(epoch+1, model, "LSTM-1", optimizer)
    print(f"Saved epoch {epoch+1} checkpoint")

# Testing

test_losses = []
num_correct = 0

h = model.init_hidden(batch_size, train_on_gpu)

true_positives = 0
false_positives = 0
false_negatives = 0

model.eval()
for inputs, labels in test_loader:

    # Variables for hidden state for backprop
    h = tuple([each.data for each in h])

    if(train_on_gpu):
        inputs, labels = inputs.cuda(), labels.cuda()

    # Get predicted outputs
    output, h = model(inputs, h)

    # Calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())

    # Convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer

    # Compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

    for p in range(len(pred)):
        if pred[p] == 1 and labels[p] == 1:
            true_positives += 1
        if pred[p] == 1 and labels[p] == 0:
            false_positives += 1
        if pred[p] == 0 and labels[p] == 1:
            false_negatives += 1

# -- stats! -- ##
# Avg test loss
test_loss = np.mean(test_losses)
print("Test loss: {:.3f}".format(test_loss))

# Accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))

# Calculate precision, recall, and F1-score
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * (precision * recall) / (precision + recall)
print("Precision: {:.3f}".format(precision))
print("Recall: {:.3f}".format(recall))
print("F1 Score: {:.3f}".format(f1))

wandb.log({"Test loss": test_loss, "Test accuracy": test_acc, "Precision": precision, "Recall": recall, "F1 Score": f1})

