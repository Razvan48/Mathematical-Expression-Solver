import os
import torch
import numpy as np
import cv2 as cv

GENERATED_DATA_PATH = 'GeneratedData'
OUTPUT_PATH = 'GeneratedDataVisualization'

def visualize_generated_data(symbol, type):
    image_path = f'{GENERATED_DATA_PATH}/{type}/{symbol}_{type}.pt'
    output_path = f'{OUTPUT_PATH}/{symbol}_{type}'

    if os.path.exists(image_path):
        os.makedirs(output_path, exist_ok=True)

        data = torch.load(image_path)

        for idx, image in enumerate(data['images']):
            image = image.squeeze(0).numpy().astype(np.uint8)
            cv.imwrite(f'{output_path}/{symbol}_{type}_{idx}.png', image)
            print(f'Saved image {idx + 1}/{len(data["images"])}: {symbol}_{type}_{idx}.png')

    else:
        print(f'Error: {image_path} does not exist.')



visualize_generated_data('close_bracket', 'train')


