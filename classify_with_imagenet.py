import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt

# Load the MobileNetV2 model
model = MobileNetV2(weights='imagenet')

def classify_image(img_path):
    # Load the image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Perform prediction
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds, top=3)[0])

    # Display the image
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Example usage
classify_image('path_to_your_image.jpg')
