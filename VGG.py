from tensorflow.keras.models import Sequential 

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 

# Initialize the model 

model = Sequential() 

# Block 1 

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3))) 

model.add(Conv2D(64, (3, 3), activation='relu', padding='same')) 

# Block 2 

model.add(MaxPooling2D(pool_size=(2, 2), strides=2)) 

model.add(Conv2D(128, (3, 3), activation='relu', padding='same')) 

model.add(Conv2D(128, (3, 3), activation='relu', padding='same')) 

# Block 3 

model.add(MaxPooling2D(pool_size=(2, 2), strides=2)) 

model.add(Conv2D(256, (3, 3), activation='relu', padding='same')) 

model.add(Conv2D(256, (3, 3), activation='relu', padding='same')) 

model.add(Conv2D(256, (3, 3), activation='relu', padding='same')) 

# Block 4 

model.add(MaxPooling2D(pool_size=(2, 2), strides=2)) 

model.add(Conv2D(512, (3, 3), activation='relu', padding='same')) 

model.add(Conv2D(512, (3, 3), activation='relu', padding='same')) 

model.add(Conv2D(512, (3, 3), activation='relu', padding='same')) 

# Block 5 

model.add(MaxPooling2D(pool_size=(2, 2), strides=2)) 

model.add(Conv2D(512, (3, 3), activation='relu', padding='same')) 

model.add(Conv2D(512, (3, 3), activation='relu', padding='same')) 

model.add(Conv2D(512, (3, 3), activation='relu', padding='same')) 

# Block 6 

model.add(MaxPooling2D(pool_size=(2, 2), strides=2)) 

# Fully connected layers 

model.add(Flatten()) 

model.add(Dense(4096, activation='relu')) 

model.add(Dropout(0.5)) 

model.add(Dense(4096, activation='relu')) 

model.add(Dropout(0.5)) 

model.add(Dense(1000, activation='softmax'))   

# Summary 

model.summary() 