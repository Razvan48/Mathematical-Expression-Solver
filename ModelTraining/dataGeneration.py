import torch
from torchvision import datasets
from torchvision import transforms
import torchvision.transforms.functional as TVF
from PIL import Image
import numpy as np
import cv2 as cv
import os
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
import torch.nn.functional as F

import Model


class CropBoundingBoxAndResize:
    def __init__(self, out_size=(28, 28), threshold=0.3):
        self.out_size = out_size
        self.threshold = threshold

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = TVF.to_tensor(image)

        image = image.squeeze(0)

        image = (image > self.threshold).float() * 255.0
        image = image.numpy().astype(np.uint8)

        x_min = image.shape[1]
        x_max = -1
        y_min = image.shape[0]
        y_max = -1

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j] == 255:
                    x_min = min(x_min, j)
                    x_max = max(x_max, j)
                    y_min = min(y_min, i)
                    y_max = max(y_max, i)

        if x_min > x_max or y_min > y_max:
            print(f'Warning: No bounding box found in the image.')

            x_min = 0
            x_max = image.shape[1] - 1
            y_min = 0
            y_max = image.shape[0] - 1

        image = image[y_min:y_max + 1, x_min:x_max + 1]

        total_padding = abs(image.shape[1] - image.shape[0])
        padding_0 = total_padding // 2
        padding_1 = total_padding - padding_0

        # print('Image Initial Shape Crop Bounding Box:', image.shape)

        if total_padding > 0:
            if image.shape[0] < image.shape[1]:
                image = F.pad(TVF.to_tensor(image), (0, 0, padding_0, padding_1), mode='constant', value=0)
            else:
                image = F.pad(TVF.to_tensor(image), (padding_0, padding_1, 0, 0), mode='constant', value=0)
        else:
            image = TVF.to_tensor(image)

        # print('Image Final Shape Crop Bounding Box:', image.shape)

        image = image.squeeze(0).numpy().astype(np.uint8) * 255
        # cv.imshow('Image Crop Bounding Box', image)
        # cv.waitKey(0)

        image = cv.resize(image, self.out_size, interpolation=cv.INTER_NEAREST)

        return image


class AddNoise():
    def __init__(self, noise_factor=0.85, max_noise_value=50):
        self.noise_factor = noise_factor
        self.max_noise_value = max_noise_value

    def __call__(self, image):
        image = image.squeeze(0).numpy().astype(np.uint8)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j] == 0 and np.random.rand() < self.noise_factor:
                    image[i, j] = np.random.randint(0, self.max_noise_value + 1)

        return image


class AddGrayScaleDimension:
    def __call__(self, image):
        image = torch.from_numpy(image).unsqueeze(0)
        return image


data_transforms = transforms.Compose([
    transforms.ToTensor(),
    CropBoundingBoxAndResize(),
    AddGrayScaleDimension(),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    # AddNoise(),
    # AddGrayScaleDimension()
])


def separate_mnist_data(mnist_data):
    data = (
        [],  # 0
        [],  # 1
        [],  # 2
        [],  # 3
        [],  # 4
        [],  # 5
        [],  # 6
        [],  # 7
        [],  # 8
        []  # 9
    )

    for image, label in mnist_data:
        data[label].append((image, label))

    return data


def load_data(path, symbol_index):
    data = []
    valid_extensions = ('.png')

    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.lower().endswith(valid_extensions):
                image = Image.open(entry.path).convert('L')  # grayscale
                image = data_transforms(image)
                data.append((image, symbol_index))

    return data


class ImageDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label


def save_data(path, filename, data):
    os.makedirs(path, exist_ok=True)

    filepath = os.path.join(path, filename)

    images = torch.stack([sample[0] for sample in data])
    labels = torch.tensor([sample[1] for sample in data])

    torch.save({
        'images': images,
        'labels': labels
    }, filepath)

    print(f'Info: Saved data to {filepath}, size: {len(data)}')


def save_dataset(path, filename, dataset):
    os.makedirs(path, exist_ok=True)

    filepath = os.path.join(path, filename)

    if isinstance(dataset, ImageDataset):
        data = dataset.data
    elif isinstance(dataset, Subset):
        data = [dataset.dataset[i] for i in dataset.indices]

    images = torch.stack([sample[0] for sample in data])
    labels = torch.tensor([sample[1] for sample in data])

    torch.save({
        'images': images,
        'labels': labels
    }, filepath)

    print(f'Info: Saved dataset to {filepath}, size: {len(dataset)}')


def load_data_into_dataset(filepath):
    data = torch.load(filepath, weights_only=True)

    images = data['images']
    labels = data['labels']

    dataset = ImageDataset(list(zip(images, labels)))
    print(f'Info: Loaded dataset from {filepath}, size: {len(dataset)}')
    return dataset


def create_concat_dataset(path, type):
    datasets = []

    filepath = os.path.join(path, type)

    with os.scandir(filepath) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.lower().endswith('.pt'):
                datasets.append(load_data_into_dataset(entry.path))

    if not datasets:
        raise ValueError(f'Error: No datasets found in {filepath}.')

    return ConcatDataset(datasets)


def generate_data():
    mnist_data_train = datasets.MNIST(root='../Datasets', train=True, download=True, transform=data_transforms)

    zero_data_train, one_data_train, two_data_train, three_data_train, four_data_train, five_data_train, six_data_train, seven_data_train, eight_data_train, nine_data_train = separate_mnist_data(mnist_data_train)

    print('Loaded MNIST Train Data:')
    print('Zero Data:', len(zero_data_train))
    print('One Data:', len(one_data_train))
    print('Two Data:', len(two_data_train))
    print('Three Data:', len(three_data_train))
    print('Four Data:', len(four_data_train))
    print('Five Data:', len(five_data_train))
    print('Six Data:', len(six_data_train))
    print('Seven Data:', len(seven_data_train))
    print('Eight Data:', len(eight_data_train))
    print('Nine Data:', len(nine_data_train))

    mnist_data_test = datasets.MNIST(root='../Datasets', train=False, download=True, transform=data_transforms)

    zero_data_test, one_data_test, two_data_test, three_data_test, four_data_test, five_data_test, six_data_test, seven_data_test, eight_data_test, nine_data_test = separate_mnist_data(mnist_data_test)

    print('Loaded MNIST Test Data:')
    print('Zero Data:', len(zero_data_test))
    print('One Data:', len(one_data_test))
    print('Two Data:', len(two_data_test))
    print('Three Data:', len(three_data_test))
    print('Four Data:', len(four_data_test))
    print('Five Data:', len(five_data_test))
    print('Six Data:', len(six_data_test))
    print('Seven Data:', len(seven_data_test))
    print('Eight Data:', len(eight_data_test))
    print('Nine Data:', len(nine_data_test))

    plus_data = load_data('../Datasets/symbol_dataset/plus', Model.from_symbol_to_index['+'])
    print('Loaded Plus Data:', len(plus_data))

    minus_data = load_data('../Datasets/symbol_dataset/minus', Model.from_symbol_to_index['-'])
    print('Loaded Minus Data:', len(minus_data))

    times_data = load_data('../Datasets/symbol_dataset/times', Model.from_symbol_to_index['*'])
    print('Loaded Times Data:', len(times_data))

    divide_data = load_data('../Datasets/symbol_dataset/divide', Model.from_symbol_to_index['/'])
    print('Loaded Divide Data:', len(divide_data))

    open_bracket_data = load_data('../Datasets/symbol_dataset/open_bracket', Model.from_symbol_to_index['('])
    print('Loaded Open Bracket Data:', len(open_bracket_data))

    close_bracket_data = load_data('../Datasets/symbol_dataset/close_bracket', Model.from_symbol_to_index[')'])
    print('Loaded Close Bracket Data:', len(close_bracket_data))


    TRAIN_TEST_SPLIT = 0.8
    TEST_VAL_SPLIT = 0.5


    zero_dataset_train = ImageDataset(zero_data_train)
    one_dataset_train = ImageDataset(one_data_train)
    two_dataset_train = ImageDataset(two_data_train)
    three_dataset_train = ImageDataset(three_data_train)
    four_dataset_train = ImageDataset(four_data_train)
    five_dataset_train = ImageDataset(five_data_train)
    six_dataset_train = ImageDataset(six_data_train)
    seven_dataset_train = ImageDataset(seven_data_train)
    eight_dataset_train = ImageDataset(eight_data_train)
    nine_dataset_train = ImageDataset(nine_data_train)

    zero_dataset_test = ImageDataset(zero_data_test)
    one_dataset_test = ImageDataset(one_data_test)
    two_dataset_test = ImageDataset(two_data_test)
    three_dataset_test = ImageDataset(three_data_test)
    four_dataset_test = ImageDataset(four_data_test)
    five_dataset_test = ImageDataset(five_data_test)
    six_dataset_test = ImageDataset(six_data_test)
    seven_dataset_test = ImageDataset(seven_data_test)
    eight_dataset_test = ImageDataset(eight_data_test)
    nine_dataset_test = ImageDataset(nine_data_test)

    zero_dataset_test, zero_dataset_val = random_split(zero_dataset_test, [int(len(zero_dataset_test) * TEST_VAL_SPLIT), len(zero_dataset_test) - int(len(zero_dataset_test) * TEST_VAL_SPLIT)])
    one_dataset_test, one_dataset_val = random_split(one_dataset_test, [int(len(one_dataset_test) * TEST_VAL_SPLIT), len(one_dataset_test) - int(len(one_dataset_test) * TEST_VAL_SPLIT)])
    two_dataset_test, two_dataset_val = random_split(two_dataset_test, [int(len(two_dataset_test) * TEST_VAL_SPLIT), len(two_dataset_test) - int(len(two_dataset_test) * TEST_VAL_SPLIT)])
    three_dataset_test, three_dataset_val = random_split(three_dataset_test, [int(len(three_dataset_test) * TEST_VAL_SPLIT), len(three_dataset_test) - int(len(three_dataset_test) * TEST_VAL_SPLIT)])
    four_dataset_test, four_dataset_val = random_split(four_dataset_test, [int(len(four_dataset_test) * TEST_VAL_SPLIT), len(four_dataset_test) - int(len(four_dataset_test) * TEST_VAL_SPLIT)])
    five_dataset_test, five_dataset_val = random_split(five_dataset_test, [int(len(five_dataset_test) * TEST_VAL_SPLIT), len(five_dataset_test) - int(len(five_dataset_test) * TEST_VAL_SPLIT)])
    six_dataset_test, six_dataset_val = random_split(six_dataset_test, [int(len(six_dataset_test) * TEST_VAL_SPLIT), len(six_dataset_test) - int(len(six_dataset_test) * TEST_VAL_SPLIT)])
    seven_dataset_test, seven_dataset_val = random_split(seven_dataset_test, [int(len(seven_dataset_test) * TEST_VAL_SPLIT), len(seven_dataset_test) - int(len(seven_dataset_test) * TEST_VAL_SPLIT)])
    eight_dataset_test, eight_dataset_val = random_split(eight_dataset_test, [int(len(eight_dataset_test) * TEST_VAL_SPLIT), len(eight_dataset_test) - int(len(eight_dataset_test) * TEST_VAL_SPLIT)])
    nine_dataset_test, nine_dataset_val = random_split(nine_dataset_test, [int(len(nine_dataset_test) * TEST_VAL_SPLIT), len(nine_dataset_test) - int(len(nine_dataset_test) * TEST_VAL_SPLIT)])

    plus_dataset = ImageDataset(plus_data)
    plus_dataset_train, plus_dataset_test = random_split(plus_dataset, [int(len(plus_dataset) * TRAIN_TEST_SPLIT), len(plus_dataset) - int(len(plus_dataset) * TRAIN_TEST_SPLIT)])
    plus_dataset_test, plus_dataset_val = random_split(plus_dataset_test, [int(len(plus_dataset_test) * TEST_VAL_SPLIT), len(plus_dataset_test) - int(len(plus_dataset_test) * TEST_VAL_SPLIT)])

    minus_dataset = ImageDataset(minus_data)
    minus_dataset_train, minus_dataset_test = random_split(minus_dataset, [int(len(minus_dataset) * TRAIN_TEST_SPLIT), len(minus_dataset) - int(len(minus_dataset) * TRAIN_TEST_SPLIT)])
    minus_dataset_test, minus_dataset_val = random_split(minus_dataset_test, [int(len(minus_dataset_test) * TEST_VAL_SPLIT), len(minus_dataset_test) - int(len(minus_dataset_test) * TEST_VAL_SPLIT)])

    times_dataset = ImageDataset(times_data)
    times_dataset_train, times_dataset_test = random_split(times_dataset, [int(len(times_dataset) * TRAIN_TEST_SPLIT), len(times_dataset) - int(len(times_dataset) * TRAIN_TEST_SPLIT)])
    times_dataset_test, times_dataset_val = random_split(times_dataset_test, [int(len(times_dataset_test) * TEST_VAL_SPLIT), len(times_dataset_test) - int(len(times_dataset_test) * TEST_VAL_SPLIT)])

    divide_dataset = ImageDataset(divide_data)
    divide_dataset_train, divide_dataset_test = random_split(divide_dataset, [int(len(divide_dataset) * TRAIN_TEST_SPLIT), len(divide_dataset) - int(len(divide_dataset) * TRAIN_TEST_SPLIT)])
    divide_dataset_test, divide_dataset_val = random_split(divide_dataset_test, [int(len(divide_dataset_test) * TEST_VAL_SPLIT), len(divide_dataset_test) - int(len(divide_dataset_test) * TEST_VAL_SPLIT)])

    open_bracket_dataset = ImageDataset(open_bracket_data)
    open_bracket_dataset_train, open_bracket_dataset_test = random_split(open_bracket_dataset, [int(len(open_bracket_dataset) * TRAIN_TEST_SPLIT), len(open_bracket_dataset) - int(len(open_bracket_dataset) * TRAIN_TEST_SPLIT)])
    open_bracket_dataset_test, open_bracket_dataset_val = random_split(open_bracket_dataset_test, [int(len(open_bracket_dataset_test) * TEST_VAL_SPLIT), len(open_bracket_dataset_test) - int(len(open_bracket_dataset_test) * TEST_VAL_SPLIT)])

    close_bracket_dataset = ImageDataset(close_bracket_data)
    close_bracket_dataset_train, close_bracket_dataset_test = random_split(close_bracket_dataset, [int(len(close_bracket_dataset) * TRAIN_TEST_SPLIT), len(close_bracket_dataset) - int(len(close_bracket_dataset) * TRAIN_TEST_SPLIT)])
    close_bracket_dataset_test, close_bracket_dataset_val = random_split(close_bracket_dataset_test, [int(len(close_bracket_dataset_test) * TEST_VAL_SPLIT), len(close_bracket_dataset_test) - int(len(close_bracket_dataset_test) * TEST_VAL_SPLIT)])

    save_dataset('GeneratedData/train', 'zero_train.pt', zero_dataset_train)
    save_dataset('GeneratedData/train', 'one_train.pt', one_dataset_train)
    save_dataset('GeneratedData/train', 'two_train.pt', two_dataset_train)
    save_dataset('GeneratedData/train', 'three_train.pt', three_dataset_train)
    save_dataset('GeneratedData/train', 'four_train.pt', four_dataset_train)
    save_dataset('GeneratedData/train', 'five_train.pt', five_dataset_train)
    save_dataset('GeneratedData/train', 'six_train.pt', six_dataset_train)
    save_dataset('GeneratedData/train', 'seven_train.pt', seven_dataset_train)
    save_dataset('GeneratedData/train', 'eight_train.pt', eight_dataset_train)
    save_dataset('GeneratedData/train', 'nine_train.pt', nine_dataset_train)

    save_dataset('GeneratedData/train', 'plus_train.pt', plus_dataset_train)
    save_dataset('GeneratedData/train', 'minus_train.pt', minus_dataset_train)
    save_dataset('GeneratedData/train', 'times_train.pt', times_dataset_train)
    save_dataset('GeneratedData/train', 'divide_train.pt', divide_dataset_train)
    save_dataset('GeneratedData/train', 'open_bracket_train.pt', open_bracket_dataset_train)
    save_dataset('GeneratedData/train', 'close_bracket_train.pt', close_bracket_dataset_train)

    save_dataset('GeneratedData/test', 'zero_test.pt', zero_dataset_test)
    save_dataset('GeneratedData/test', 'one_test.pt', one_dataset_test)
    save_dataset('GeneratedData/test', 'two_test.pt', two_dataset_test)
    save_dataset('GeneratedData/test', 'three_test.pt', three_dataset_test)
    save_dataset('GeneratedData/test', 'four_test.pt', four_dataset_test)
    save_dataset('GeneratedData/test', 'five_test.pt', five_dataset_test)
    save_dataset('GeneratedData/test', 'six_test.pt', six_dataset_test)
    save_dataset('GeneratedData/test', 'seven_test.pt', seven_dataset_test)
    save_dataset('GeneratedData/test', 'eight_test.pt', eight_dataset_test)
    save_dataset('GeneratedData/test', 'nine_test.pt', nine_dataset_test)

    save_dataset('GeneratedData/test', 'plus_test.pt', plus_dataset_test)
    save_dataset('GeneratedData/test', 'minus_test.pt', minus_dataset_test)
    save_dataset('GeneratedData/test', 'times_test.pt', times_dataset_test)
    save_dataset('GeneratedData/test', 'divide_test.pt', divide_dataset_test)
    save_dataset('GeneratedData/test', 'open_bracket_test.pt', open_bracket_dataset_test)
    save_dataset('GeneratedData/test', 'close_bracket_test.pt', close_bracket_dataset_test)

    save_dataset('GeneratedData/val', 'zero_val.pt', zero_dataset_val)
    save_dataset('GeneratedData/val', 'one_val.pt', one_dataset_val)
    save_dataset('GeneratedData/val', 'two_val.pt', two_dataset_val)
    save_dataset('GeneratedData/val', 'three_val.pt', three_dataset_val)
    save_dataset('GeneratedData/val', 'four_val.pt', four_dataset_val)
    save_dataset('GeneratedData/val', 'five_val.pt', five_dataset_val)
    save_dataset('GeneratedData/val', 'six_val.pt', six_dataset_val)
    save_dataset('GeneratedData/val', 'seven_val.pt', seven_dataset_val)
    save_dataset('GeneratedData/val', 'eight_val.pt', eight_dataset_val)
    save_dataset('GeneratedData/val', 'nine_val.pt', nine_dataset_val)

    save_dataset('GeneratedData/val', 'plus_val.pt', plus_dataset_val)
    save_dataset('GeneratedData/val', 'minus_val.pt', minus_dataset_val)
    save_dataset('GeneratedData/val', 'times_val.pt', times_dataset_val)
    save_dataset('GeneratedData/val', 'divide_val.pt', divide_dataset_val)
    save_dataset('GeneratedData/val', 'open_bracket_val.pt', open_bracket_dataset_val)
    save_dataset('GeneratedData/val', 'close_bracket_val.pt', close_bracket_dataset_val)

    train_dataset = create_concat_dataset('GeneratedData', 'train')
    test_dataset = create_concat_dataset('GeneratedData', 'test')
    val_dataset = create_concat_dataset('GeneratedData', 'val')


generate_data()



