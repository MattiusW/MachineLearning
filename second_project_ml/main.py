from Mnist import Mnist

def main():
    print("Classification in Machine Learning")

    # Załadowanie danych z MNIST
    mnist = Mnist().load_mnist_data

    X, y = mnist.data, mnist.target
    print(X)
    print(X.shape)
    print(y)
    print(y.shape)


if __name__ == '__main__':
    main()
