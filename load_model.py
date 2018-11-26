import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,classification_report

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense



def load_model():
    # load model
    from keras.models import model_from_json
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model

def create_confusion_matrix(model,validation_generator,batch_size = 32):
    num_of_test_samples = validation_generator.n
    # Confution Matrix and Classification Report
    labels = np.empty(len(validation_generator.class_indices),dtype="S10")
    for key,value in validation_generator.class_indices.items():
        print key
        print value
        labels[int(value)] = str(key)
    print labels

    Y_pred = model.predict_generator(validation_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(validation_generator.classes, y_pred))
    print('Classification Report')
    # target_names = ['Cats', 'Dogs', 'Horse']
    print(classification_report(validation_generator.classes, y_pred, target_names=labels))


model = load_model()


# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
valid_datagen = ImageDataGenerator(rescale=1./255)
valid_set = valid_datagen.flow_from_directory('dataset/I_Forgot',
target_size=(64, 64),
batch_size=32,
class_mode=None
)

create_confusion_matrix(model,valid_set)


# Part 3 - Making new predictions
# from keras.preprocessing import image
# test_image = image.load_img('dataset/I_Forgot/sigma/sigma5.jpg', target_size = (64, 64))
# # test_image = image.load_img('dataset/I_Forgot/!/!5.jpg', target_size = (64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = model.predict(test_image)
# # result = classifier.predict(test_image)
# # print training_set.class_indices
# print result
# if result[0][0] > 0.5:
#     prediction = 'sigma'
# else:
#     prediction = '!'
# print prediction

# predict = model.predict_generator(valid_set)