import keras
from keras.utils import plot_model

def load_model():
    model = keras.models.load_model("Conf_Base/conf_baseline.model")
    plot_model(model, to_file='model.pdf')

if __name__ == '__main__':
    load_model()
