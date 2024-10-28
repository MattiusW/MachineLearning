from sklearn.datasets import fetch_openml

class Mnist():
    def __init__(self):
        self.load_mnist_data = fetch_openml('mnist_784', as_frame=False)
        