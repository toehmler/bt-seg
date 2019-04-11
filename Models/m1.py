import required

class m1:
    
    def __init__(self):
        self.compiled = self.compile()
    
    def compile(self):
        single = Sequential()

        single.add(Conv2D(64, (7,7), strides=(1,1), padding='valid', activation='relu', input_shape=(33,33,4)))
        single.add(BatchNormalization())
        single.add(Dropout(0.5))

        single.add(Conv2D(128, (5,5), strides=(1,1), padding='valid', activation='relu'))
        single.add(BatchNormalization())
        single.add(Dropout(0.5))

        single.add(Conv2D(128, (5,5), strides=(1,1), padding='valid', activation='relu'))
        single.add(BatchNormalization())
        single.add(Dropout(0.5))

        single.add(Conv2D(128, (3,3), strides=(1,1), padding='valid', activation='relu'))
        single.add(Dropout(0.25))

        single.add(Flatten())
        single.add(Dense(5, activation='softmax'))

        sgd = SGD(lr=0.001, decay=0.01, momentum=0.9)
        single.compile(loss='categorical_crossentropy', optimizer='sgd')

        return single

 
