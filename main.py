import cv2 as cv
import numpy as np
import queue
import torch

import ModelTraining.Model as Model
from expressionEvaluator import evaluate


def process_image(image):
    IMAGE_THRESHOLD = 128
    image = (image > IMAGE_THRESHOLD).astype(np.uint8) * 255
    image = (255 - image).astype(np.uint8)

    return image


def find_bounding_boxes(image):
    bounding_boxes = []
    visited = [[False for _ in range(image.shape[1])] for _ in range(image.shape[0])]

    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] == 255 and not visited[i][j]:
                x_min, y_min = j, i
                x_max, y_max = j, i

                q = queue.Queue()

                q.put((i, j))
                visited[i][j] = True

                while not q.empty():
                    current_i, current_j = q.get()

                    x_min = min(x_min, current_j)
                    y_min = min(y_min, current_i)
                    x_max = max(x_max, current_j)
                    y_max = max(y_max, current_i)

                    for k in range(len(dx)):
                        new_i = current_i + dy[k]
                        new_j = current_j + dx[k]

                        if (0 <= new_i < image.shape[0] and
                                0 <= new_j < image.shape[1] and
                                image[new_i, new_j] == 255 and not visited[new_i][new_j]):
                            visited[new_i][new_j] = True
                            q.put((new_i, new_j))

                bounding_boxes.append((x_min, y_min, x_max, y_max))

    bounding_boxes.sort(key=lambda box: box[0])

    print('Bounding Boxes:', bounding_boxes)

    return bounding_boxes



def add_bounding_boxes_to_image(image, bounding_boxes):
    BOUNDING_BOX_GRAY = 128
    image_copy = image.copy()
    for (x_min, y_min, x_max, y_max) in bounding_boxes:
        cv.rectangle(image_copy, (x_min, y_min), (x_max, y_max), BOUNDING_BOX_GRAY, 3)

    return image_copy


def analyze_image(image, bounding_boxes):
    IMAGE_SIZE = (28, 28)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model.ConvolutionalNeuralNetwork()
    model.load_state_dict(torch.load('ModelTraining/model.pth', weights_only=True))
    model.to(device)

    mathematical_expression = ''

    for (x_min, y_min, x_max, y_max) in bounding_boxes:  # Bounding boxes are sorted from left to right.
        cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]
        print('Image Shape:', image.shape, 'Cropped Image Shape:', cropped_image.shape)
        cropped_image = cv.resize(cropped_image, IMAGE_SIZE)

        model.eval()
        with torch.no_grad():
            input_image = torch.tensor(cropped_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            output = model(input_image)
            _, predicted = output.max(1)

            mathematical_expression += Model.from_index_to_symbol[predicted.item()]

    print('Mathematical Expression:', mathematical_expression)
    result = evaluate(mathematical_expression)
    return result


IMAGE_PATH = 'ExpressionImages/0.png'

if __name__ == '__main__':
    image = cv.imread(IMAGE_PATH, cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f'Error: Image {IMAGE_PATH} not found.')

    cv.imshow('Initial Image', image)
    cv.waitKey(0)
    image = process_image(image)
    cv.imshow('Processed Image', image)
    cv.waitKey(0)

    bounding_boxes = find_bounding_boxes(image)
    image_copy = add_bounding_boxes_to_image(image, bounding_boxes)
    cv.imshow('Image with Bounding Boxes', image_copy)
    cv.waitKey(0)

    print('Mathematical Result:', analyze_image(image, bounding_boxes))

    cv.destroyAllWindows()



