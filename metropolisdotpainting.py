import random

import cv2
import numpy as np


class MetropolistDotPainting:
    def __init__(self, img_path, save_path):
        self.img = cv2.imread(img_path)
        self.save_path = save_path
    
    def draw(self, n, size):
        img_paint = np.ones(self.img.shape)
        img_paint *= 255

        img_width = self.img.shape[1]
        img_height = self.img.shape[0]

        x = img_width * random.random()
        y = img_height * random.random()

        for i in range(n):
            x_n = img_width * random.random()
            y_n = img_height * random.random()

            prob = self.get_prob(x, y)
            prob_n = self.get_prob(x_n, y_n)

            if prob_n < prob:
                ratio = prob_n / prob
                rand = random.random()
                if ratio > rand:
                    cv2.circle(img_paint, (int(x_n), int(y_n)), size, (0, 0, 0), -1)
                    x = x_n
                    y = y_n
            else:
                cv2.circle(img_paint, (int(x_n), int(y_n)), size, (0, 0, 0), -1)
                x = x_n
                y = y_n
        
        cv2.imwrite(self.save_path, img_paint)
    
    def get_prob(self, x, y):
        x = int(x)
        y = int(y)
        b = self.img[y, x, 0]
        g = self.img[y, x, 1]
        r = self.img[y, x, 2]
        prob = 0.299 * r + 0.587 * g + 0.114 * b
        prob /= 255
        prob = 1 - prob
        prob = prob ** 5
        return prob


if __name__ == '__main__':
    img_path = './data/lena.png'
    save_path = './data/lena_dot.jpg'
    mdp = MetropolistDotPainting(img_path, save_path)
    mdp.draw(100000, 1)
