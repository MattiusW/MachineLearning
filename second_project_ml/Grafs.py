import matplotlib.pyplot as plt


class Grafs():
    def plots_digit(self, image_data):
        image = image_data.reshape(28,28)
        plt.imshow(image, cmap="binary")
        plt.axis("off")
        plt.show()