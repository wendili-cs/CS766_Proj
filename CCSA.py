import numpy as np
import adapt
from adapt.feature_based import CCSA
from tensorflow import keras
from keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Reshape,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    Convolution2D,
    MaxPooling2D,
)
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import warnings
from PIL import Image

img_h, img_w = 32, 16
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
save_CCSA = "checkpoints_CCSA/"
mapping = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "A",
    11: "B",
    12: "C",
    13: "D",
    14: "E",
    15: "F",
    16: "G",
    17: "H",
    18: "I",
    19: "J",
    20: "K",
    21: "L",
    22: "M",
    23: "N",
    24: "P",
    25: "Q",
    26: "R",
    27: "S",
    28: "T",
    29: "U",
    30: "V",
    31: "W",
    32: "X",
    33: "Y",
    34: "Z",
}


def get_encoder(input_shape=(32, 16, 3)):
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), activation="relu", padding="valid", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(48, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(720, activation="relu"))
    model.add(Dense(168 * 3, activation="relu"))
    return model


def get_task(input_shape=(168 * 3,)):
    model = Sequential()
    model.add(Dense(168 * 3, activation="relu", input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(34, activation="softmax"))
    return model


def process_image(image):
    img = image.convert("RGB")
    img = img.resize([img_w, img_h], Image.ANTIALIAS)
    return img


ccsa_model = CCSA(
    encoder=get_encoder(),
    task=get_task(),
    optimizer=Adam(0.001),
    loss="CategoricalCrossentropy",
    metrics=["acc"],
    random_state=0,
)

xs = np.zeros((8446, 32, 16, 3))
ys = np.zeros((8446, 34))
xt = np.zeros((762, 32, 16, 3))
yt = np.zeros((762, 34))


def predict_model(size, name, img_input):
    curr_name = save_CCSA + name
    curr_model = CCSA(
        encoder=get_encoder(),
        task=get_task(),
        optimizer=Adam(0.001),
        loss="CategoricalCrossentropy",
        metrics=["acc"],
        random_state=0,
    )
    curr_model.fit(xs, ys, xt, yt, epochs=0, verbose=1, batch_size=32)
    curr_model.load_weights(curr_name)
    curr_prediction = np.argmax(curr_model.predict(img_input))
    return curr_prediction


ccsa_model.fit(xs, ys, xt, yt, epochs=0, verbose=1, batch_size=32)

ccsa_scenario_dict = {
    "Dark night": "db",
    "Rainy, snow, or fog": "weather",
    "Far or near to the camera": "fn",
    "Other challenging scenarios": "challenge",
}
