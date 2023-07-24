import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
def detect_red_color(image, num_colors=5, red_threshold=120):
    image_array = np.array(image)
    red_pixels = image_array[:, :, 0] > red_threshold
    red_pixels = np.stack((red_pixels,) * 3, axis=-1)
    red_image = np.where(red_pixels, image_array, 0)
    pixels = red_image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, n_init=10)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    red_detected = False
    for color in dominant_colors:
        if color[0] > red_threshold:
            red_detected = True
            break
    return red_detected, red_image
if __name__ == "__main__":
    num_colors = 5
    red_threshold = 120
    image_path = "color.jpg"
    image = Image.open(image_path)
    red_detected, red_image = detect_red_color(image, num_colors, red_threshold)
    plt.figure(figsize=(3, 3))
    plt.imshow(image)
    plt.axis('off')
    plt.title("given Image")
    if red_detected:
        plt.text(0, image.size[1]+20, "Red color detected.", fontsize=12, color='red', ha='left')
    plt.show()

