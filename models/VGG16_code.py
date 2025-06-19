from tensorflow.keras.applications import VGG16

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers (optional - experiment with unfreezing)
for layer in base_model.layers:
  layer.trainable = False

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping

x = layers.Flatten()(base_model.output)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation='sigmoid')(x)

# Create model
model = models.Model(inputs=base_model.input, outputs=output)

early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

model.compile(optimizer=optimizers.Adam(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_generator, epochs=100, validation_data=test_generator, callbacks=[early_stopping])

model.save("vgg16_model1.keras")