import cv2
import numpy as np


#REDUCCIÓN DE RUIDO
def reduce_noise(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred


# CÁLCULO DEL GRADIENTE
def calculate_gradient(image):
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    return gradient_magnitude, gradient_direction


# SUPRESIÓN NO MÁXIMA
def non_maximum_suppression(gradient_magnitude, gradient_direction):
    gradient_direction = np.degrees(gradient_direction)
    gradient_direction[gradient_direction < 0] += 180
    suppressed = np.zeros_like(gradient_magnitude)
    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            angle = gradient_direction[i, j]
            q = 255
            r = 255
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = gradient_magnitude[i, j + 1]
                r = gradient_magnitude[i, j - 1]
            elif 22.5 <= angle < 67.5:
                q = gradient_magnitude[i + 1, j - 1]
                r = gradient_magnitude[i - 1, j + 1]
            elif 67.5 <= angle < 112.5:
                q = gradient_magnitude[i + 1, j]
                r = gradient_magnitude[i - 1, j]
            elif 112.5 <= angle < 157.5:
                q = gradient_magnitude[i - 1, j - 1]
                r = gradient_magnitude[i + 1, j + 1]
            if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                suppressed[i, j] = gradient_magnitude[i, j]
    return suppressed


# UMBRAL DOBLE
def double_threshold(suppressed, low_threshold, high_threshold):
    strong_edges = (suppressed > high_threshold)
    weak_edges = (suppressed >= low_threshold) & (suppressed <= high_threshold)
    return strong_edges, weak_edges


# HISTÉRESIS
def edge_tracking_by_hysteresis(strong_edges, weak_edges):
    h, w = strong_edges.shape
    edge_map = np.zeros((h, w), dtype=np.uint8)
    edge_map[strong_edges] = 255

    dx = [-1, -1, -1, 0, 0, 1, 1, 1]
    dy = [-1, 0, 1, -1, 1, -1, 0, 1]

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if weak_edges[y, x]:
                for direction in range(8):
                    new_x = x + dx[direction]
                    new_y = y + dy[direction]
                    if strong_edges[new_y, new_x]:
                        edge_map[y, x] = 255
                        break
    return edge_map


# LECTURA DE IMÁGEN
input_image = cv2.imread('uno1.jpg', cv2.IMREAD_GRAYSCALE)

# APLICA FUNCIONES

input_image = reduce_noise(input_image)
gradient_magnitude, gradient_direction = calculate_gradient(input_image)
suppressed = non_maximum_suppression(gradient_magnitude, gradient_direction)
strong_edges, weak_edges = double_threshold(suppressed, 50, 150)
edge_map = edge_tracking_by_hysteresis(strong_edges, weak_edges)

# RESULTADOS
cv2.imshow('Resultado de Canny', edge_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
