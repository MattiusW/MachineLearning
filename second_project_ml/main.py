from Grafs import Grafs
from Mnist import Mnist

def main():
    print("Classification in Machine Learning")

    # Załadowanie danych z MNIST
    mnist = Mnist().load_mnist_data

    # Inicjalizacja danych
    X, y = mnist.data, mnist.target

    # Wyswietlenie danych za pomoca grafu
    graf = Grafs()
    graf.plots_digit(X[0])

if __name__ == '__main__':
    main()
