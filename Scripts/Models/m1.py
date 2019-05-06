from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Maximum, Concatenate, Activation
from keras.optimizers import SGD
from keras.models import Model
from keras.regularizers import l1_l2

def compile():
    single = Sequential()

    single.add(Conv2D(64, (7,7), 
                      strides=(1,1), padding='valid', activation='relu',
                      kernel_regularizer=l1_l2(l1=0.01, l2=0.01), 
                      input_shape=(33,33,4)))

    single.add(BatchNormalization())
    single.add(Dropout(0.5))

    single.add(Conv2D(128, (5,5), strides=(1,1), padding='valid', 
                      kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                      activation='relu'))
    single.add(BatchNormalization())
    single.add(Dropout(0.5))

    single.add(Conv2D(128, (5,5), strides=(1,1), padding='valid', 
                      kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                      activation='relu'))
    single.add(BatchNormalization())
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

def two_path(input_shape):
    X_input = Input(input_shape)
  
    X = Conv2D(64,(7,7),strides=(1,1),padding='valid')(X_input)
    X = BatchNormalization()(X)
    X1 = Conv2D(64,(7,7),strides=(1,1),padding='valid')(X_input)
    X1 = BatchNormalization()(X1)
    X = Maximum()([X,X1])
    X = Conv2D(64,(4,4),strides=(1,1),padding='valid',activation='relu')(X)

    X2 = Conv2D(160,(13,13),strides=(1,1),padding='valid')(X_input)
    X2 = BatchNormalization()(X2)
    X21 = Conv2D(160,(13,13),strides=(1,1),padding='valid')(X_input)
    X21 = BatchNormalization()(X21)
    X2 = Maximum()([X2,X21])

    X3 = Conv2D(64,(3,3),strides=(1,1),padding='valid')(X)
    X3 = BatchNormalization()(X3)
    X31 =  Conv2D(64,(3,3),strides=(1,1),padding='valid')(X)
    X31 = BatchNormalization()(X31)
    X = Maximum()([X3,X31])
    X = Conv2D(64,(2,2),strides=(1,1),padding='valid',activation='relu')(X)

    X = Concatenate()([X2,X])
    X = Conv2D(5,(21,21),strides=(1,1),padding='valid')(X)
    X = Flatten()(X)
    X = Activation('softmax')(X)

    model = Model(inputs = X_input, outputs = X)
    sgd = SGD(lr=0.001, decay=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    return model

if __name__ == '__main__':
    model = two_path((33, 33, 4))
    #model = compile()
    print(model.summary())


