import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

import Model
import dataGeneration


# model = Model.ConvolutionalNeuralNetwork()
model = Model.NeuralNetwork()

# train_dataset = dataGeneration.create_concat_dataset('GeneratedData', 'train')
# test_dataset = dataGeneration.create_concat_dataset('GeneratedData', 'test')
# val_dataset = dataGeneration.create_concat_dataset('GeneratedData', 'val')
train_dataset = dataGeneration.create_concat_dataset('GeneratedHOGData', 'train')
test_dataset = dataGeneration.create_concat_dataset('GeneratedHOGData', 'test')
val_dataset = dataGeneration.create_concat_dataset('GeneratedHOGData', 'val')

print('Train Dataset Size:', len(train_dataset))
print('Test Dataset Size:', len(test_dataset))
print('Validation Dataset Size:', len(val_dataset))

BATCH_SIZE = 64

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def train_model(model, train_dataloader, val_dataloader, num_epochs, learning_rate):
    NUM_BATCHES_PER_PRINT = 100

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    train_losses_per_epoch = []
    val_losses_per_epoch = []

    train_accuracies_per_epoch = []
    val_accuracies_per_epoch = []

    val_predictions = []
    val_labels = []

    for epoch in range(num_epochs):
        model.train()

        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_dataloader):
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_function(outputs, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

            if (batch_idx + 1) % NUM_BATCHES_PER_PRINT == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_dataloader)}')

        train_loss /= len(train_dataloader)
        train_accuracy = 1.0 * correct / total

        train_losses_per_epoch.append(train_loss)
        train_accuracies_per_epoch.append(train_accuracy)

        print(f'Training, Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss}, Accuracy: {train_accuracy}')

        model.eval()

        val_loss = 0.0
        correct = 0
        total = 0

        val_predictions = []
        val_labels = []

        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in enumerate(val_dataloader):
                X_batch = X_batch.to(device).float()
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = loss_function(outputs, y_batch)

                val_loss += loss.item()

                _, predicted = outputs.max(1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

                val_predictions.extend(predicted.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

                if (batch_idx + 1) % NUM_BATCHES_PER_PRINT == 0:
                    print(f'Validation, Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(val_dataloader)}')

        val_loss /= len(val_dataloader)
        val_accuracy = 1.0 * correct / total

        val_losses_per_epoch.append(val_loss)
        val_accuracies_per_epoch.append(val_accuracy)

        print(f'Validation, Epoch {epoch + 1}/{num_epochs}, Loss: {val_loss}, Accuracy: {val_accuracy}')

    return train_losses_per_epoch, val_losses_per_epoch, train_accuracies_per_epoch, val_accuracies_per_epoch, val_predictions, val_labels


def plot_train_results(train_losses_per_epoch, val_losses_per_epoch, train_accuracies_per_epoch, val_accuracies_per_epoch, val_predictions, val_labels):
      plt.figure(figsize=(12, 6))

      plt.subplot(1, 2, 1)
      plt.plot(train_losses_per_epoch, label='Training Loss', color='blue')
      plt.plot(val_losses_per_epoch, label='Validation Loss', color='red')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.title('Training and Validation Losses')
      plt.legend()

      plt.subplot(1, 2, 2)
      plt.plot(train_accuracies_per_epoch, label='Training Accuracy', color='blue')
      plt.plot(val_accuracies_per_epoch, label='Validation Accuracy', color='red')
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy')
      plt.title('Training and Validation Accuracies')
      plt.legend()

      plt.tight_layout()
      # plt.savefig('train_results.png')
      plt.savefig('train_results_2.png')
      plt.show()

      val_predictions = np.array(val_predictions).reshape(-1)
      val_labels = np.array(val_labels).reshape(-1)

      conf_matrix = confusion_matrix(val_labels, val_predictions)

      conf_matrix_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '(', ')']  #, ',', '.']
      sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=conf_matrix_labels, yticklabels=conf_matrix_labels)

      plt.xlabel('Predicted Labels')
      plt.ylabel('True Labels')
      plt.title('Confusion Matrix')

      # plt.savefig('confusion_matrix.png')
      plt.savefig('confusion_matrix_2.png')
      plt.show()

      precision, recall, f1_score, _ = precision_recall_fscore_support(val_labels, val_predictions, average='macro', zero_division=0)

      print(f'Precision: {precision}')
      print(f'Recall: {recall}')
      print(f'F1 Score: {f1_score}')

      return precision, recall, f1_score


train_losses_per_epoch, val_losses_per_epoch, train_accuracies_per_epoch, val_accuracies_per_epoch, val_predictions, val_labels = train_model(model, train_dataloader, val_dataloader, num_epochs=3, learning_rate=0.001)
precision, recall, f1_score = plot_train_results(train_losses_per_epoch, val_losses_per_epoch, train_accuracies_per_epoch, val_accuracies_per_epoch, val_predictions, val_labels)


# torch.save(model.state_dict(), 'model.pth')
torch.save(model.state_dict(), 'model_2.pth')





