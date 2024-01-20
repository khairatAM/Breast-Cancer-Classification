import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers.legacy import Adam
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from cancernet.cancernet import CancerNet
from cancernet import config
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

NUM_EPOCHS=7; INIT_LR=1e-3; BATCH_SIZE=32; IMG_SIZE=(48,48)

train_paths = list(paths.list_images(config.TRAIN_PATH))
len_train = len(train_paths)
len_val = len(list(paths.list_images(config.VAL_PATH)))
len_test = len(list(paths.list_images(config.TEST_PATH)))

train_labels = [int(p.split(os.path.sep)[-2]) for p in train_paths]
train_labels = to_categorical(train_labels)

class_totals = train_labels.sum(axis=0)
class_weight = class_totals.max()/class_totals
class_weight = {0: class_weight[0], 1: class_weight[1]}

train_aug = ImageDataGenerator(
        rescale=1/255.0,
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest"
    )

val_aug = ImageDataGenerator(rescale=1 / 255.0)

train_gen = train_aug.flow_from_directory(
        config.TRAIN_PATH,
        target_size = IMG_SIZE,
        batch_size = BATCH_SIZE,
        class_mode = 'categorical'
    )

val_gen = val_aug.flow_from_directory(
        config.VAL_PATH,
        target_size = IMG_SIZE,
        batch_size = BATCH_SIZE,
        class_mode = 'categorical'
    )

test_gen = val_aug.flow_from_directory(
        config.TEST_PATH,
        target_size = IMG_SIZE,
        batch_size = BATCH_SIZE,
        class_mode = 'categorical'
    )


model = CancerNet.build(width=IMG_SIZE[0],height=IMG_SIZE[1],depth=3,classes=2)
optimizer = Adam(learning_rate=INIT_LR)
model.compile(loss = "binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

print("Training the model")

M = model.fit(
        train_gen,
        steps_per_epoch = len_train//BATCH_SIZE,
        validation_data = val_gen,
        validation_steps = len_val//BATCH_SIZE,
        class_weight = class_weight,
        epochs = NUM_EPOCHS
    )

# print("\nSaving...\n")
# path = os.path.sep.join([config.MODEL_PATH, "my_model"])

# if not os.path.exists(path):
#     os.makedirs(path)

# model.save(path)

print("Evaluating the model")
test_gen.reset()
pred_idx = model.predict(
    test_gen,
    steps = (len_test // BATCH_SIZE) + 1
    )
pred_idx = np.argmax(pred_idx, axis = 1) #get the class with higher pred value for each instance

print(classification_report(test_gen.classes, #classes are inferred from testing directory, in this case they are 0 and 1
                            pred_idx,
                            target_names = test_gen.class_indices.keys()
                            ))

cm=confusion_matrix(test_gen.classes,pred_idx)
total=sum(sum(cm))
accuracy=(cm[0,0]+cm[1,1])/total
specificity=cm[1,1]/(cm[1,0]+cm[1,1])
sensitivity=cm[0,0]/(cm[0,0]+cm[0,1])
print(cm)
print(f'Accuracy: {accuracy}')
print(f'Specificity: {specificity}')
print(f'Sensitivity: {sensitivity}')

N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N), M.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), M.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), M.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0,N), M.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy on the IDC Dataset")
plt.xlabel("Epoch No.")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('plot.png')



            
