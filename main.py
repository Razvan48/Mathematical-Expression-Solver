import cv2 as cv
import numpy as np
import queue
import torch

import pygame as pg

import ModelTraining.Model as Model
from expressionEvaluator import evaluate

import torchvision.transforms.functional as TVF
import torch.nn.functional as F



def process_image(image):
    IMAGE_THRESHOLD = 32
    image = (image > IMAGE_THRESHOLD).astype(np.uint8) * 255

    # image = (255 - image).astype(np.uint8)  # Nu mai trebuie flip daca folosim drawing_board.
    image = image.astype(np.uint8)

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

    # print('Bounding Boxes:', bounding_boxes)

    return bounding_boxes



def add_bounding_boxes_to_image(image, bounding_boxes):
    BOUNDING_BOX_GRAY = 128
    image_copy = image.copy()
    for (x_min, y_min, x_max, y_max) in bounding_boxes:
        cv.rectangle(image_copy, (x_min, y_min), (x_max, y_max), BOUNDING_BOX_GRAY, 3)

    return image_copy


def analyze_image(image, bounding_boxes):
    mathematical_expression = ''
    global result
    result = None

    for (x_min, y_min, x_max, y_max) in bounding_boxes:  # Bounding boxes are sorted from left to right.
        cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]

        total_padding = abs(cropped_image.shape[1] - cropped_image.shape[0])
        padding_0 = total_padding // 2
        padding_1 = total_padding - padding_0

        if total_padding > 0:
            if cropped_image.shape[0] < cropped_image.shape[1]:
                cropped_image = F.pad(TVF.to_tensor(cropped_image), (0, 0, padding_0, padding_1), mode='constant', value=0)
            else:
                cropped_image = F.pad(TVF.to_tensor(cropped_image), (padding_0, padding_1, 0, 0), mode='constant', value=0)
        else:
            cropped_image = TVF.to_tensor(cropped_image)

        cropped_image = cropped_image.squeeze(0).numpy().astype(np.uint8) * 255

        # print('Image Shape:', image.shape, 'Cropped Image Shape:', cropped_image.shape)
        cropped_image = cv.resize(cropped_image, IMAGE_SIZE)

        model.eval()
        with torch.no_grad():
            input_image = torch.tensor(cropped_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            output = model(input_image)
            _, predicted = output.max(1)

            mathematical_expression += Model.from_index_to_symbol[predicted.item()]

    # print('Mathematical Expression:', mathematical_expression)
    try:
        if len(mathematical_expression) > 0:
            result = evaluate(mathematical_expression)
    except Exception as e:
        # print(f'Error: Failed evaluating expression {mathematical_expression}: {e}.')
        pass

    return mathematical_expression, result


IMAGE_PATH = 'ExpressionImages/1.png'

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 600

FPS = 60

HEIGHT_PERCENT_DRAWING_AREA = 0.8

NUM_DRAWING_PIXELS_ON_WIDTH = 200
NUM_DRAWING_PIXELS_ON_HEIGHT = 50

PIXEL_WIDTH = SCREEN_WIDTH / NUM_DRAWING_PIXELS_ON_WIDTH
PIXEL_HEIGHT = SCREEN_HEIGHT * HEIGHT_PERCENT_DRAWING_AREA / NUM_DRAWING_PIXELS_ON_HEIGHT

BRUSH_SIZE_DRAWING = 1
BRUSH_SIZE_ERASING = 3

IMAGE_SIZE = (28, 28)

TEXT_PADDING = 0.05

drawing_board = [[0 for _ in range(NUM_DRAWING_PIXELS_ON_WIDTH)] for _ in range(NUM_DRAWING_PIXELS_ON_HEIGHT)]
drawing_speed = 7

formatted_mathematical_expression = ''
result = None


def draw():
    pg.draw.rect(screen, (50, 50, 50), (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT * (1.0 - HEIGHT_PERCENT_DRAWING_AREA)))

    for i in range(NUM_DRAWING_PIXELS_ON_HEIGHT):
        for j in range(NUM_DRAWING_PIXELS_ON_WIDTH):
            pg.draw.rect(screen, (drawing_board[i][j],
                                  drawing_board[i][j],
                                  drawing_board[i][j]), (j * PIXEL_WIDTH, SCREEN_HEIGHT * (1.0 - HEIGHT_PERCENT_DRAWING_AREA) + i * PIXEL_HEIGHT, PIXEL_WIDTH, PIXEL_HEIGHT))

    formatted_mathematical_expression_text = font.render(formatted_mathematical_expression, True, (255, 255, 255))
    equal_sign_text = font.render(' = ', True, (255, 255, 255))
    if result is not None:
        result_text = font.render(str(result), True, (255, 255, 255))
    else:
        result_text = font.render('ERROR', True, (255, 0, 0))

    screen.blit(formatted_mathematical_expression_text, (SCREEN_WIDTH * TEXT_PADDING, SCREEN_HEIGHT * TEXT_PADDING))
    screen.blit(equal_sign_text, (SCREEN_WIDTH * TEXT_PADDING + formatted_mathematical_expression_text.get_width(), SCREEN_HEIGHT * TEXT_PADDING))
    screen.blit(result_text, (SCREEN_WIDTH * TEXT_PADDING + formatted_mathematical_expression_text.get_width() + equal_sign_text.get_width(), SCREEN_HEIGHT * TEXT_PADDING))


def dist(i_1, j_1, i_2, j_2):
    return max(abs(i_1 - i_2), abs(j_1 - j_2))


def update():
    x, y = pg.mouse.get_pos()
    if y > SCREEN_HEIGHT * (1.0 - HEIGHT_PERCENT_DRAWING_AREA):
        i = int((y - SCREEN_HEIGHT * (1.0 - HEIGHT_PERCENT_DRAWING_AREA)) / PIXEL_HEIGHT)
        j = int(x / PIXEL_WIDTH)
        if pg.mouse.get_pressed()[0]:
            for crt_i in range(NUM_DRAWING_PIXELS_ON_HEIGHT):
                for crt_j in range(NUM_DRAWING_PIXELS_ON_WIDTH):
                    if dist(i, j, crt_i, crt_j) > BRUSH_SIZE_DRAWING:
                        continue
                    drawing_board[crt_i][crt_j] = drawing_board[crt_i][crt_j] + drawing_speed * delta_time
                    drawing_board[crt_i][crt_j] = min(255, drawing_board[crt_i][crt_j])
        elif pg.mouse.get_pressed()[2]:
            for crt_i in range(NUM_DRAWING_PIXELS_ON_HEIGHT):
                for crt_j in range(NUM_DRAWING_PIXELS_ON_WIDTH):
                    if dist(i, j, crt_i, crt_j) > BRUSH_SIZE_ERASING:
                        continue
                    drawing_board[crt_i][crt_j] = drawing_board[crt_i][crt_j] - drawing_speed * delta_time
                    drawing_board[crt_i][crt_j] = max(0, drawing_board[crt_i][crt_j])


def predict():
    global drawing_board
    global formatted_mathematical_expression
    global result

    image = np.zeros((NUM_DRAWING_PIXELS_ON_HEIGHT, NUM_DRAWING_PIXELS_ON_WIDTH), dtype=np.uint8)
    for i in range(NUM_DRAWING_PIXELS_ON_HEIGHT):
        for j in range(NUM_DRAWING_PIXELS_ON_WIDTH):
            image[i][j] = drawing_board[i][j]

    image = process_image(image)

    bounding_boxes = find_bounding_boxes(image)

    mathematical_expression, result = analyze_image(image, bounding_boxes)

    formatted_mathematical_expression = ''
    for idx in range(len(mathematical_expression)):
        formatted_mathematical_expression += mathematical_expression[idx]

        if mathematical_expression[idx].isdigit() and idx + 1 < len(mathematical_expression) and mathematical_expression[idx + 1].isdigit():
            continue
        if mathematical_expression[idx] == '.' or mathematical_expression[idx] == ',':
            continue

        if idx + 1 < len(mathematical_expression):
            formatted_mathematical_expression += ' '

    if result is not None and result.is_integer():
        result = int(result)

    # print('MATHEMATICAL EXPRESSION:', formatted_mathematical_expression)
    # print('RESULT:', result)

    # print(formatted_mathematical_expression + ' = ' + str(result))


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model.ConvolutionalNeuralNetwork()
    model.load_state_dict(torch.load('ModelTraining/model.pth', weights_only=True))
    # model = Model.NeuralNetwork()
    # model.load_state_dict(torch.load('ModelTraining/model_2.pth', weights_only=True))

    model.to(device)

    pg.init()
    screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pg.display.set_caption('Mathematical Expression Solver')

    font = pg.font.SysFont('Arial', 36)

    is_running = True
    clock = pg.time.Clock()
    while is_running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                is_running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    is_running = False
                elif event.key == pg.K_c or event.key == pg.K_C:
                    drawing_board = [[0 for _ in range(NUM_DRAWING_PIXELS_ON_WIDTH)] for _ in range(NUM_DRAWING_PIXELS_ON_HEIGHT)]

        update()
        predict()
        screen.fill((0, 0, 0))
        draw()

        pg.display.flip()

        delta_time = clock.tick(FPS)

    pg.quit()

    '''
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
    '''



