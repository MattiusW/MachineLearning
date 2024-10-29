from Classificator import Classificator
from Grafs import Grafs
from Mnist import Mnist

def main():
    print("Classification in Machine Learning")

    # Załadowanie danych z MNIST
    mnist = Mnist().load_mnist_data

    # Inicjalizacja danych
    X, y = mnist.data, mnist.target

    # Rozdzielenie zbiorow danych
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    # Klasyfikator binarny
    y_train_5 = (y_train == '5') # Wartosc true dla piatek
    y_test_5 = (y_test == '5')
    classificator = Classificator()
    classificator.sgd_model(X_train,y_train_5, X[0])
    print("Is five?: ", classificator.sgd_predict)

    # Wyswietlenie danych za pomoca grafu
    graf = Grafs()
    graf.plots_digit(X[0])

if __name__ == '__main__':
    main()
