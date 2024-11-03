import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve

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

    # Sprawdzanie modelu macierz pomylek
    cm = accuracy_test.model_confusion_matrix(sgd_clf, X_train, y_train_5)
    print(f"Macierz pomylek:\n{cm}")
    # Sprawdzanie precyzji modelu
    predict = accuracy_test.predict
    prec_score = precision_score(y_train_5, predict)
    reca_score = recall_score(y_train_5, predict)
    F1 = f1_score(y_train_5, predict)
    print(f"W ilu procentach model sie nie myli: {prec_score}")
    print(f"Ile procent przypadków rozpoznaje: {reca_score}")
    print(f"Średnia harmonicza F1 = {F1}")

    # Wyliczanie precyzji i pełności modelu
    precisions, recalls, thresholds = accuracy_test.model_decision_score(sgd_clf, X_train, y_train_5)

    # Uzyskanie 90% precyzji
    y_scores = accuracy_test.decision_score
    idx_for_90_precision = (precisions >= 90).argmax()
    treshold_for_90_precision = thresholds[idx_for_90_precision]
    y_train_pred_90 = (y_scores > treshold_for_90_precision)
    precision_90 = precision_score(y_train_5, y_train_pred_90)
    recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)
    print(f"Precyzja modelu: {precision_90}, Dokładność modelu {recall_at_90_precision}")

    # Wykres krzywej ROC
    fpr, tpr, thresholds_roc = roc_curve(y_train_5, y_scores)
    idx_for_treshold_at_90 = (thresholds <= treshold_for_90_precision).argmax()
    tpr_90, fpr_90 = tpr[idx_for_treshold_at_90], fpr[idx_for_treshold_at_90]
    print("ROC SCORE: " , roc_auc_score(y_train_5, y_scores))

    # RandomForestClassifier
    y_probas_forest = classificator.forest_clf_model(X_train, y_train_5)
    print(f"Forest cfl: {y_probas_forest[:2]}")

    # Dane do pokazania precyzji i czulosci modelu na grafie
    y_scores_forest = y_probas_forest[:, 1]
    precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(y_train_5, y_scores_forest)

    # Wynik F1 i ROC AUC
    y_train_pred_forest = y_probas_forest[:, 1] >= 0.5
    forest_f1_score = f1_score(y_train_5, y_train_pred_forest)
    forest_roc_score = roc_auc_score(y_train_5, y_scores_forest)
    print(f"F1 = {forest_f1_score}, ROC = {forest_roc_score}")

    # Wyswietlenie danych za pomoca grafu
    graf = Grafs()
    graf.forest_graf(recalls, precisions, recalls_forest, precisions_forest)
    graf.roc_graf(fpr, tpr, tpr_90, fpr_90)
    graf.decision_graf(thresholds,precisions,recalls)
    graf.plots_digit(X[0])
    graf.plots_digit(X[1])

if __name__ == '__main__':
    main()
