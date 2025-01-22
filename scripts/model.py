# scripts/model.py
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

def build_model(input_shape, num_classes):
    print("Building the model...")
    
    # Load pre-trained VGG16 model without the top layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the layers of the base model
    print("Freezing the base VGG16 model layers...")
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top
    print("Adding custom layers on top of the base model...")
    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    
    print(f"Model architecture summary:\n{model.summary()}")
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
