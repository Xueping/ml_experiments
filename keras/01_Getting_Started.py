from keras.models import Sequential
from keras.layers import Dense, Activation

# Here is the Sequential model:
model = Sequential()

# Stacking layers is as easy as .add():
model.add(Dense(units=64, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))

# Once your model looks good, configure its learning process with .compile():
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])