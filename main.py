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

def create_base_model(classes=3,finalAct = 'softmax'):
    # Initialising the CNN
    classifier = Sequential()
    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Step 3 - Flattening
    classifier.add(Flatten())
    # Step 4 - Full connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = classes, activation = finalAct))
    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier

def load_data():
    # Part 2 - Fitting the CNN to the images
    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    # valid_datagen = ImageDataGenerator(rescale=1./255)
    training_set = train_datagen.flow_from_directory('dataset/Training',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'categorical')
    test_set = test_datagen.flow_from_directory('dataset/test_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'categorical')
    # valid_set = valid_datagen.flow_from_directory('dataset/I_Forgot',
    # target_size=(64, 64),
    # batch_size=32,
    # class_mode=None
    # )
    return [training_set,test_set]


def plot(history):
    print history
    # plot metrics
    plt.plot(history.history['acc'])
    plt.savefig("accuracyPlot.png")
    plt.show()



def create_confusion_matrix(model,validation_generator,batch_size = 32):
    num_of_test_samples = validation_generator.n
    # Confution Matrix and Classification Report
    Y_pred = model.predict_generator(validation_generator, num_of_test_samples // batch_size + 1)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(validation_generator.classes, y_pred))
    print('Classification Report')
    target_names = ['Cats', 'Dogs', 'Horse']
    print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

def save_model(classifier):
    # save model
    # serialize model to JSON
    model_json = classifier.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    classifier.save_weights("model.h5")
    print("Saved model to disk")

model = create_base_model()
training_set,testing_set = load_data()

print testing_set.class_indices
print testing_set.classes

train_size = training_set.n
test_size = testing_set.n
epoch = 15
batch_size = 32
history = model.fit_generator(generator=training_set,
steps_per_epoch = train_size//batch_size,
epochs = epoch,
validation_data = testing_set,
validation_steps = test_size//batch_size)

plot(history)

# e = classifier.evaluate_generator(generator=test_set)

# print e


save_model(model)

# import load_model
# model = load_model.load_model()





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



# # predict = classifier.predict_generator(valid_set)
# print valid_set.filenames
#
# predict = model.predict_generator(valid_set)
# # for i in range(len(predict)):
# #     if
# #     predict[i]
# print predict
# print predict.shape
# for p in predict:
#     if

# def evaluate():
