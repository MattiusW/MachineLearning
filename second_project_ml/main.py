import matplotlib.pyplot as plt

from CheckAccuracy import CheckAccuracy
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

    # Klasyfikator binarny sgd
    y_train_5 = (y_train == '5') # Wartosc true dla piatek
    y_test_5 = (y_test == '5')
    classificator = Classificator()
    sgd_test_model = classificator.sgd_clf_model(X_train, y_train_5, X[0])
    print("Is five correct?: ", sgd_test_model)
    sgd_test_false = classificator.sgd_clf_model(X_train, y_train_5, X[1])
    print("Is five?: ", sgd_test_false)

    # Klasyfikator binarny dummy
    dummy_test_model = classificator.dummy_clf_model(X_train, y_train_5, X_train)
    print("Piątki w zbiorze X_train: ", any(dummy_test_model))

    # Sprawdzanie modelu pod wzgledem wydajnosci
    accuracy_test = CheckAccuracy()
    sgd_clf = classificator.get_sgd_clf_model()
    sgd_result = accuracy_test.model_cross_value_precision(sgd_clf, X_train, y_train_5)
    dummy_clf = classificator.get_dummy_clf_model()
    dummy_result = accuracy_test.model_cross_value_precision(dummy_clf, X_train, y_train_5)
    print(f"Test sprawdzianu krzyżowego dla klasyfikatora sgd:\n{sgd_result}")
    print(f"Test sprawdzianu krzyżowego dla klasyfikatora dummy:\n{dummy_result}")

    # Wyswietlenie danych za pomoca grafu
    graf = Grafs()
    graf.plots_digit(X[0])
    graf.plots_digit(X[1])

if __name__ == '__main__':
    main()
