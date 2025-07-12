import torch
import os
import shutil
from torch.utils.data import DataLoader
import cv2 as cv


import Model
import dataGeneration


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model.ConvolutionalNeuralNetwork()
model.load_state_dict(torch.load('model.pth', weights_only=True))
model.to(device)

OUTPUT_PATH = 'MisclassifiedData'
if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)
os.makedirs(OUTPUT_PATH, exist_ok=True)

BATCH_SIZE = 1

test_dataset = dataGeneration.create_concat_dataset('GeneratedData', 'test')
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print('Test Dataset Size:', len(test_dataset))

model.eval()

with torch.no_grad():
    for batch_idx, (X_batch, y_batch) in enumerate(test_dataloader):
        X_batch = X_batch.to(device).float()
        y_batch = y_batch.to(device)

        outputs = model(X_batch)
        _, predicted = outputs.max(1)

        if y_batch != predicted:
            predicted_symbol = Model.from_index_to_symbol[predicted[0].item()]
            label_symbol = Model.from_index_to_symbol[y_batch[0].item()]

            if predicted_symbol == '*':
                predicted_symbol = 'times'
            elif predicted_symbol == '/':
                predicted_symbol = 'divide'

            if label_symbol == '*':
                label_symbol = 'times'
            elif label_symbol == '/':
                label_symbol = 'divide'

            image_path = os.path.join(OUTPUT_PATH, f'{label_symbol}/{predicted_symbol}-{batch_idx}.png')
            os.makedirs(os.path.dirname(image_path), exist_ok=True)

            image = X_batch[0].cpu().numpy().squeeze(0)
            cv.imwrite(image_path, image)

        print(f'Processed batch {batch_idx + 1}')









