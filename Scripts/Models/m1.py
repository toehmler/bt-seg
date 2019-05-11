from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.regularizers import l1_l2
from keras.models import load_model

def compile():
    single = Sequential()

    single.add(Conv2D(64, (7,7), 
                      strides=(1,1), padding='valid', activation='relu',
                      kernel_regularizer=l1_l2(l1=0.01, l2=0.01), 
                      input_shape=(33,33,4)))

    single.add(BatchNormalization(axis=-1))
    single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
    single.add(Dropout(0.5))

    single.add(Conv2D(128, (5,5), strides=(1,1), padding='valid', 
                      kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                      activation='relu'))
    single.add(BatchNormalization(axis=-1))
    single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
    single.add(Dropout(0.5))

    single.add(Conv2D(128, (5,5), strides=(1,1), padding='valid', 
                      kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                      activation='relu'))
    single.add(BatchNormalization(axis=-1))
    single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
    single.add(Dropout(0.5))

    single.add(Conv2D(128, (3,3), strides=(1,1), padding='valid', 
                      kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                      activation='relu'))
    single.add(Dropout(0.25))

    single.add(Flatten())
    single.add(Dense(5, activation='softmax'))

    sgd = SGD(lr=0.001, decay=0.01, momentum=0.9)
    single.compile(loss='categorical_crossentropy', optimizer='sgd')

    return single

def jeb():
    single = Sequential()

    single.add(Conv2D(64, (7,7), 
                      strides=(1,1), padding='valid', activation='relu',
                      kernel_regularizer=l1_l2(l1=0.01, l2=0.01), 
                      input_shape=(33,33,4)))

    single.add(BatchNormalization())
    single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
    single.add(Dropout(0.5))

    single.add(Conv2D(128, (5,5), strides=(1,1), padding='valid', 
                      kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                      activation='relu'))
    single.add(BatchNormalization())
    single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
    single.add(Dropout(0.5))

    single.add(Conv2D(128, (5,5), strides=(1,1), padding='valid', 
                      kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                      activation='relu'))
    single.add(BatchNormalization())
    single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
    single.add(Dropout(0.5))

    single.add(Conv2D(128, (3,3), strides=(1,1), padding='valid', 
                      kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                      activation='relu'))
    single.add(BatchNormalization())
    single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
    single.add(Dropout(0.25))

    single.add(Conv2D(128, (3,3), strides=(1,1), padding='valid', 
                      kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                      activation='relu'))
    single.add(BatchNormalization())
    single.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
    single.add(Dropout(0.25))

    single.add(Conv2D(128, (3,3), strides=(1,1), padding='valid', 
                      kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                      activation='relu'))
    single.add(Dropout(0.25))

    single.add(Flatten())
    single.add(Dense(5, activation='softmax'))

    sgd = SGD(lr=0.0005, decay=0.01, momentum=0.9)
    single.compile(loss='categorical_crossentropy', optimizer='sgd')

    return single

if __name__ == '__main__':
    model = load_model('/Users/treyoehmler/dev/tumors/seg/Outputs/Models/Trained/m1_7.h5')
    compile('slop')



