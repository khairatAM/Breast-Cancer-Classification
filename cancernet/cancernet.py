from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import SeparableConv2D
from keras.layers import AveragePooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras import backend as K

class CancerNet:
    @staticmethod
    def build(width,height,depth,classes):
        model = Sequential()
        shape = (height, width, depth)

        channelDim = -1
        if K.image_data_format() == 'channels_first':
            shape = (depth, width, height)
            channelDim = 1

        model.add(SeparableConv2D(32, (5,5), padding='same',input_shape=shape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))
        model.add(AveragePooling2D(pool_size=(2,2)))
        model.add(SeparableConv2D(64, (5,5), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))
        model.add(SeparableConv2D(128, (5,5), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))
        model.add(AveragePooling2D(pool_size=(2,2)))
        model.add(SeparableConv2D(128, (5,5), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))
        model.add(SeparableConv2D(256, (5,5), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))
        model.add(SeparableConv2D(256, (5,5), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))
        model.add(AveragePooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
