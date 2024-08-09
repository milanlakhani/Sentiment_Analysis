import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from string import punctuation
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import wandb

wandb.login()

# Hyperparameters
batch_size = 50
embedding_dim = 400
hidden_dim = 256
n_layers = 2
output_size = 1
epochs = 3
print_every = 100
gradient_clipping = 5
learning_rate = 0.001
dropout_prob = 0.3

run = wandb.init(
    project = "SA1",
    config = {
        "batch_size": 50,
        "embedding_dim": 400,
        "hidden_dim": 256,
        "n_layers": 2,
        "output_size": 1,
        "epochs": 3,
        "gradient_clipping": 5,
        "learning_rate": 0.001,
        "dropout_prob": 0.3
    },
)

df = pd.read_csv("./IMDB Dataset.csv")

# Convert to lower case
df['review'] = df['review'].apply(lambda x:x.lower())

# Remove punctuation
df['clean_text'] = df['review'].apply(lambda x:''.join([c for c in x if c not in punctuation]))
df['len_review'] = df['clean_text'].apply(lambda x:len(x))

# Create list of reviews
review_list = df['clean_text'].tolist()
# Create a list of words
review_list = ' '.join(review_list)
words = review_list.split()

# Count words using Counter Method
count_words = Counter(words)
total_words = len(words)

sorted_words = count_words.most_common(total_words)
vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
reviews_split = df['clean_text'].tolist()

# Tokenize — Encode the words
reviews_int = []
for review in reviews_split:
    r = [vocab_to_int[w] for w in review.split()]
    reviews_int.append(r)
print (reviews_int[0:3])

# Tokenize — Encode the labels
labels_split = df['sentiment'].tolist()
encoded_labels = [1 if label =='positive' else 0 for label in labels_split]
encoded_labels = np.array(encoded_labels)

# Analyze Reviews Length
reviews_len = [len(x) for x in reviews_int]
pd.Series(reviews_len).hist()
plt.show()
pd.Series(reviews_len).describe()
wandb.log("Review lengths": wandb.Image(plt))

# Remove Outliers — Getting rid of extremely long or short reviews
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

features = pad_features(reviews_int,200)
len_feat = len(features)
split_frac = 0.8

# Training, Validation, Test Dataset Split
split_frac = 0.8
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

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output, final_hidden_state):
        attn_weights = self.attn(final_hidden_state)
        attn_weights = torch.bmm(lstm_output, attn_weights.unsqueeze(2)).squeeze(2)
        soft_attn_weights = self.softmax(attn_weights)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super().__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)

        embeds = self.embedding(x)
        enc_lstm_out, hidden = self.encoder_lstm(embeds, hidden)

        # Pass the encoder output to the decoder
        dec_lstm_out, hidden = self.decoder_lstm(enc_lstm_out, hidden)

        # Apply attention mechanism
        attn_output = self.attention(dec_lstm_out, hidden[0][-1])

        # Dropout and fully-connected layer
        out = self.dropout(attn_output)
        out = self.fc(out)
        # Sigmoid function
        sig_out = self.sig(out)

        # Reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels

        return sig_out, hidden


    def init_hidden(self, batch_size):
        ''' Create two new tensors with sizes n_layers x batch_size x hidden_dim,
            initialized to zero, for hidden state and cell state of LSTM
        '''
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
net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
print(net)

# First checking if GPU is available
train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

# Loss and optimization functions
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Training params
counter = 0

# Move model to GPU, if available
if(train_on_gpu):
    net.cuda()

net.train()
for e in range(epochs):
    # Initialize hidden state
    h = net.init_hidden(batch_size)

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
        net.zero_grad()

        # Get the output from the model
        output, h = net(inputs, h)

        # Calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), gradient_clipping)
        optimizer.step()

        # Loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:

                # Variables for hidden state for backprop
                val_h = tuple([each.data for each in val_h])

                if(train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
            wandb.log({"epoch": epoch, "step": counter, "loss": loss.item(), "val loss": np.mean(val_losses)})

# Testing

test_losses = []
num_correct = 0

h = net.init_hidden(batch_size)

true_positives = 0
false_positives = 0
false_negatives = 0

net.eval()
for inputs, labels in test_loader:

    # Variables for hidden state for backprop
    h = tuple([each.data for each in h])

    if(train_on_gpu):
        inputs, labels = inputs.cuda(), labels.cuda()

    # Get predicted outputs
    output, h = net(inputs, h)

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

# On User-generated Data
# First, we will define a tokenize function that will take care of pre-processing steps and then we will create a predict function that will give us the final output after parsing the user provided review.

def preprocess(review, vocab_to_int):
    review = review.lower()
    word_list = review.split()
    num_list = []
    reviews_int = []
    for word in word_list:
        if word in vocab_to_int.keys():
            num_list.append(vocab_to_int[word])
    reviews_int.append(num_list)
    return reviews_int

def predict(net, test_review, sequence_length=200):
    ''' Prints out whether a given review is predicted to be positive or negative in sentiment.'''

    int_rev = preprocess(test_review, vocab_to_int)
    features = pad_features(int_rev, seq_length=seq_length)

    features = torch.from_numpy(features)

    net.eval()
    val_h = net.init_hidden(1)
    val_h = tuple([each.data for each in val_h])

    if(train_on_gpu):
        features = features.cuda()

    output, val_h = net(features, val_h)

    pred = torch.round(output)
    output = ["Positive" if pred.item() == 1 else "Negative"]

    print(output)

# Test reviews
test_review_pos = 'This movie had the best acting and the dialogue was so good. I loved it.'
test_review_neg = 'The worst movie I have seen; acting was terrible and I want my money back. This movie had bad acting and the dialogue was slow.'

# Call function
seq_length=200
print(test_review_pos)
predict(net, test_review_pos, seq_length)
print(test_review_neg)
predict(net, test_review_neg, seq_length)
