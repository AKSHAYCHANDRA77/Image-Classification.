import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define directories for training and validation images
train_dir = 'path/to/training/directory'
validation_dir = 'path/to/validation/directory'

# Create ImageDataGenerator instances for data augmentation and preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches using train_datagen
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Rescale images to 224x224 pixels
    batch_size=32,
    class_mode='categorical'  # Assuming categorical classification
)

# Flow validation images in batches using validation_datagen
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Define CNN model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # Assuming 10 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using fit_generator
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,  # Number of batches per epoch
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50   # Number of validation batches
)
