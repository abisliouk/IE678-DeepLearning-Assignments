# Outline 

This is a repository with the assignments of IE678 Deep Learning course at University of Mannheim. 

Assignments have been done by pair of students @[Artem Bisliouk](https://github.com/abisliouk) and @[Nico Ramin](https://github.com/NicoRamin)

## 1st Assignment. FNNs

In this assignment feedforward networks in PyTorch were explored.

### Research area
FNN, MLP, Logistic Regression, Scipy’s BFGS optimizer, ReLU, Backpropagation.

### Tasks
- **Implement an MLP**.
  - Implementation of a Multilayer Perceptron (MLP) manually in PyTorch
  - Examining the provided LogisticRegression class, understanding how its parameters are stored, how the forward pass calculations are performed, and how the model is initialized and used
  - The MLP class is to be completed as per the functionality documented in the Python file, without using pre-canned PyTorch neural network functionality, but torch.nn.Parameter and torch.nn.Module can be used
  - The MLP implementation is then to be expanded to compute the network's output as a vector when a vector is provided, and as a matrix when a matrix is provided, without using torch.vmap, but by directly implementing the required operations efficiently.
- **Experiment with MLPs**.
  - Exploring regression using Multilayer Perceptrons (MLPs) with varying numbers of units in the hidden layer.
  - Examining the training data and conjecturing how a fit for an FNN with zero, one, two, and three hidden neurons would look.
  - An FNN with two hidden neurons is to be trained using Scipy's BFGS optimizer, and the mean squared error (MSE) on the training and test data is to be determined and plotted. This process is to be repeated multiple times.
  - FNNs with 1, 2, 3, 10, 50, and 100 hidden neurons are to be trained, with the MSE on the training and test datasets determined in each case.
  - The dataset and the predictions of each FNN on the test set are to be plotted in a single plot.
  - The output of the hidden neurons is to be visualized when training an FNN with 2 hidden neurons, and then repeated with 3 and 10 hidden neurons.
  - Optionally, the experiments can be repeated with different optimizers, different numbers of hidden layers, and/or different types of units in the hidden layers.
- **Backpropagation**.
  - Computing various gradients of the network output with respect to its inputs and weights manually for a network with a single hidden layer of size Z = 50.
  - The network operations include matrix-vector multiplications, element-wise addition, and the application of the logistic function element-wise.
  - The loss between prediction y and target y-hat is computed.
  - The backward graph for this network is also provided.
  - Code is provided to extract the model parameters from a trained model, run the model using PyTorch on a single input x, and extract some of the gradients using PyTorch's autograd.
  - The task requires computing the results of the forward pass directly using basic PyTorch operations, and computing the results of the backward pass directly using basic PyTorch operations without using autograd. The derivations and resulting formulas for each of these quantities are to be provided in the report.
  
### Structure
- Assignment [content](https://github.com/abisliouk/IE678-DeepLearning-Assignments/tree/main/Assignment%201/task)
- Solution [notebook](https://github.com/abisliouk/IE678-DeepLearning-Assignments/blob/main/Assignment%201/a01.ipynb)
- Solution [reports](https://github.com/abisliouk/IE678-DeepLearning-Assignments/tree/main/Assignment%201/solution%20report)
- Visualization [plots and notes](https://github.com/abisliouk/IE678-DeepLearning-Assignments/tree/main/Assignment%201/visualization)
- [Helper Functions](https://github.com/abisliouk/IE678-DeepLearning-Assignments/blob/main/Assignment%201/a01helper.py) for results validation


## 2nd Assignment. RNNs

In this assignment the basic sentiment analysis (positive/negative) for movie reviews (text) using PyTorch was performed and investigated.

### Research area
Tokenization, Sentiment Analysis, Data Loaders, Embedding layer, LSTM encoder, Linear layer, Logistic function, GloVe word embeddings, t-SNE, PCA, Elman cells, GRU cells, LSTM cells, dropout influence, RNN direction.

### Tasks

- **Datasets**.
  - The goal of this task is to implement a custom PyTorch Dataset class, ReviewsDataset, that preprocesses movie review data (tokenizing reviews and converting labels to binary), and provides functionality to retrieve the length of the dataset and individual (review, label) pairs, with the option to return reviews as a list of token IDs instead of textual tokens.
- **Data Loaders**.
  - The task involves understanding the PyTorch DataLoader, running the provided example, and modifying it to increase the batch size, which will cause it to fail.
  - The task then requires implementing a collate function to standardize the length of all reviews in each batch by adding zero padding for reviews shorter than MAX_SEQ_LEN and cropping reviews longer than MAX_SEQ_LEN, and using this function to construct data loaders for training, validation, and testing.
- **Recurrent Neural Network**.
  - Training a Recurrent Neural Network (RNN) using PyTorch's pre-built layers, specifically an Embedding layer, an LSTM encoder, a Linear layer, and a Logistic function. 
  - Completing the init_hidden method, constructing the RNN by instantiating these components.
  - Implementing the forward pass to return predicted probabilities and thought vectors
  - Completing the reviews_eval function to compute accuracy and average loss.
  - Implementing the reviews_train function to complete the optimization loop.
- **Pretrained Embeddings & Visualization**.
  - The task asks understanding and explaining the content of the provided GloVe word embeddings file, loading these embeddings into an embedding layer, and exploring them using t-SNE or other methods.
  - Then, we are to train the provided SimpleRNN model for 10 epochs and use t-SNE to visualize the embeddings of the thought vectors of both the training and validation datasets. This process is to be repeated twice, first by initializing the word embedding layer with the GloVe embeddings (pretraining with finetuning), and then by freezing the word embeddings to the GloVe embeddings (pretraining without finetuning), discussing findings each time.
- **Exploration**.
  - Exploring different hyperparameter settings and RNN architectures, using the SimpleLSTM model as a blueprint. Starting from the model with pretraining and finetuning, the effects of changing the dropout probability, the type of RNN cell, and the directionality of the RNN are to be studied.
  - At least seven different setups must be included and discussed, with results reported and visualized.
  - Conclusions should be drawn about the most suitable design for the problem.
  - The model and training process are then to be optimized, with changes potentially including dimensionality, dropout, and early abort on validation.
  
### Structure
- Assignment [content](https://github.com/abisliouk/IE678-DeepLearning-Assignments/tree/main/Assignment%202/task)
- Solution [notebook](https://github.com/abisliouk/IE678-DeepLearning-Assignments/blob/main/Assignment%202/a02.ipynb)
- Solution [reports](https://github.com/abisliouk/IE678-DeepLearning-Assignments/tree/main/Assignment%202/solution%20report)
- Visualization [plots and code](https://github.com/abisliouk/IE678-DeepLearning-Assignments/tree/main/Assignment%202/visualization)
- [Helper module](https://github.com/abisliouk/IE678-DeepLearning-Assignments/blob/main/Assignment%202/a02helper.py)
