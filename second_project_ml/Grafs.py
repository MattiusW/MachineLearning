import matplotlib.pyplot as plt
from scipy.ndimage import label
from sympy.printing.pretty.pretty_symbology import line_width


class Grafs():
    def plots_digit(self, image_data):
        image = image_data.reshape(28,28)
        plt.imshow(image, cmap="binary")
        plt.axis("off")
        plt.show()

    def decision_graf(self, thresholds, precisions, recalls):
        plt.plot(thresholds, precisions[:-1], "b--", label="Precyzja", linewidth=2)
        plt.plot(thresholds, recalls[:-1], "g--", label="Czułość", linewidth=2)
        plt.vlines(thresholds, 0, 1.0, "k", "dotted", label="Próg")
        plt.plot(recalls, precisions, linewidth=2, label="Krzywa precyzji/czułości")
        plt.show()

    def roc_graf(self, fpr, tpr, fpr_90, tpr_90):
        plt.plot(fpr, tpr, linewidth=2, label="Krzywa ROC")
        plt.plot([0, 1], [0, 1], "k:", label="Krzywa ROC klasyfikatora losowego")
        plt.plot([fpr_90], [tpr_90], "ko", label="Próg dla precyzji 90%")
        plt.show()