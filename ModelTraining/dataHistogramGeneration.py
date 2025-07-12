import os
import shutil
import torch
from skimage.feature import hog

import dataGeneration



train_dataset = dataGeneration.create_concat_dataset('GeneratedData', 'train')
test_dataset = dataGeneration.create_concat_dataset('GeneratedData', 'test')
val_dataset = dataGeneration.create_concat_dataset('GeneratedData', 'val')


OUTPUT_PATH = 'GeneratedHOGData'
if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)
os.makedirs(OUTPUT_PATH, exist_ok=True)


def generate_histogram_file(dataset, type, file_name):
    images = []
    labels = []

    for (image, label) in dataset:

        image = image.squeeze(0).numpy()
        hog_features = hog(
            image,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            visualize=False,
            channel_axis=None
        )
        hog_features = torch.tensor(hog_features, dtype=torch.float32)

        images.append(hog_features)
        labels.append(label)

    images = torch.stack(images)
    labels = torch.tensor(labels)

    file_path = OUTPUT_PATH + f'/{type}/{file_name}'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    torch.save({
        'images': images,
        'labels': labels
    }, file_path)



generate_histogram_file(train_dataset, 'train', 'train_histogram.pt')
generate_histogram_file(test_dataset, 'test', 'test_histogram.pt')
generate_histogram_file(val_dataset, 'val', 'val_histogram.pt')


