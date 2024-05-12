import torch
import torch.nn as nn
import numpy as np

#for losses_plot
import matplotlib
import matplotlib.pyplot as plt

import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

# from ecg_ptbxl_benchmarking.code.models.base_model import ClassificationModel

# class YourModel(ClassificationModel):
#     def __init__(self, name, n_classes,  sampling_frequency, outputfolder, input_shape):
#         self.name = name
#         self.n_classes = n_classes
#         self.sampling_frequency = sampling_frequency
#         self.outputfolder = outputfolder
#         self.input_shape = input_shape 

#     def fit(self, X_train, y_train, X_val, y_val):
#         pass

#     def predict(self, X):
#         pass
# Define the Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # ReLU activation function
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection (if needed)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        shortcut = self.shortcut(x)
        
        out += shortcut  # Element-wise addition
        out = self.relu(out)
        return out

# Define the ResNet-like model
class YourModel(nn.Module):
    def __init__(self, categories):
        super(YourModel, self).__init__()
        # Define the layers for the model
        self.conv1 = nn.Conv1d(12, 32, kernel_size=5, stride=1, padding=2)  # Adjust the input channels to match the input data
        self.block1 = ResidualBlock(32, 32)
        self.block2 = ResidualBlock(32, 32)
        self.fc1 = nn.Linear(32000, 32)  # Adjust the input size based on your data
        self.fc2 = nn.Linear(32, categories)  # Updated to match the desired number of output categories
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)  # Apply sigmoid activation
        return x
    
    def accuracy_multi(self, inp, targ, thresh=0.5, sigmoid=True):
        if sigmoid:
            inp = inp.sigmoid()
        return ((inp > thresh) == targ.bool()).float().mean()
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load_model(cls, path, categories = 5):
        model = cls(categories=categories)  # Create an instance of your model class
        model.load_state_dict(torch.load(path))
        model.eval()  # Set the model to evaluation mode
        return model

        
    def fit(self, X_train, y_train, X_val, y_val, sigmoid= True, model_save_path=None):
        # Convert data to the appropriate data type (e.g., torch.float32)
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)

        X_train = X_train.permute(0, 2, 1)  # Permute the dimensions to change to [batch_size, 12, 1000]
        X_val = X_val.permute(0, 2, 1)  # Permute the dimensions to change to [batch_size, 12, 1000]

        # Implement the training process for your model here
        # You should use torch.nn.Module's optimization, loss, and train loop
        # Make sure your model is compatible with float input and target data

        # Example of training loop (update with your model and data):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        # criterion = nn.BCELoss() 
        criterion = nn.BCEWithLogitsLoss() 

        epochs = 100
        batch_size = 128

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


        train_losses = []  # To store training losses
        val_losses = []    # To store validation losses
        train_accuracies = []
        val_accuracies = []

        for epoch in range(epochs):
            self.train()
            num_batches = len(X_train) // batch_size

            num_correct = 0
            num_samples = 0
            train_loss = 0.0
            val_loss = 0.0

            for batch in range(num_batches):
                start = batch * batch_size
                end = (batch + 1) * batch_size
                batch_X = X_train[start:end]
                batch_y = y_train[start:end]
                # # Check batch_y
                # print("Value of batch_y:", batch_y)

                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                # Calculate accuracy
                num_correct += (self.accuracy_multi(outputs, batch_y, sigmoid=sigmoid) * batch_X.size(0)).item()
                num_samples += batch_X.size(0)
            train_accuracy = (num_correct / num_samples) * 100
            train_accuracies.append(train_accuracy)

            # Validation loss calculation
            self.eval()
            num_val_batches = len(X_val) // batch_size
            num_correct = 0
            num_samples = 0

            for batch in range(num_val_batches):
                start = batch * batch_size
                end = (batch + 1) * batch_size
                batch_X_val = X_val[start:end]
                batch_y_val = y_val[start:end]

                with torch.no_grad():
                    outputs_val = self(batch_X_val) 
                    loss_val = criterion(outputs_val, batch_y_val)
                    val_loss += loss_val.item()
                # #Calculate accuracy
                num_correct += (self.accuracy_multi(outputs_val, batch_y_val, sigmoid=sigmoid) * batch_X_val.size(0)).item()
                num_samples += batch_X_val.size(0)
            # Calculate accuracy for validation at the end of the epoch
            val_accuracy = (num_correct / num_samples) * 100
            val_accuracies.append(val_accuracy)

            train_loss /= num_batches
            val_loss /= num_val_batches
            # scheduler.step(val_loss)  # Update learning rate

            # Append the losses to the lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Epoch [{epoch+1}/{epochs}], Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')

            if model_save_path:
                self.save_model(model_save_path)

        # Plot the training and validation loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', marker='o')
        plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', marker='o')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Save the plot as an image
        loss_plot_path = "/global/D1/homes/jayao/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2/output/custom_test/models/custom/loss_plot.png"
        plt.savefig(loss_plot_path)

    def predict(self, X, model_path=None, categories=5):
        X = torch.FloatTensor(X)
        X = X.permute(0, 2, 1) 

        self.eval()
        predictions = []
        if model_path:
            model = YourModel.load_model(model_path, categories=categories)

            batch_size = 128
            num_batches = len(X) // batch_size
            remainder = len(X) % batch_size  # Number of remaining samples

            for batch in range(num_batches):
                start = batch * batch_size
                end = (batch + 1) * batch_size
                batch_X = X[start:end]

                with torch.no_grad():
                    outputs = model(batch_X)
                    predictions.append(outputs)

            # Process the remaining samples
            if remainder > 0:
                start = num_batches * batch_size
                batch_X = X[start:]
                with torch.no_grad():
                    outputs = model(batch_X)
                    predictions.append(outputs)

        # Concatenate the list of predictions into a single array
        predictions = torch.cat(predictions, dim=0)

        return predictions
    
    def gradcam(self, input_data, target_class=None, layer_name='conv1', cuda=True):
        self.eval()

        # Move model to GPU if cuda is True
        if cuda:
            self.cuda()

        # Convert input_data to torch tensor
        input_tensor = torch.FloatTensor(input_data).permute(0, 2, 1)
        input_tensor.requires_grad = True

        # Forward pass
        output = self(input_tensor)

        # Set target class to the one with the maximum probability if not specified
        if target_class is None:
            target_class = torch.argmax(output, dim=1)

        # Create GradCAM attribute for the specified layer
        cam = LayerGradCam(self, self._modules[layer_name])

        # Attribute calculation
        attributions = cam.attribute(input_tensor, target=target_class, n_steps=50)

        # Detach attributions from the graph and move to CPU
        attributions = attributions.detach().cpu().numpy()

        return attributions
