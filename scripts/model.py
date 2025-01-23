from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_model(input_shape, num_classes):
    print("Building the model...")
    
    # Load pre-trained NASNetMobile model without the top layers
    base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the layers of the base model
    print("Freezing the base NASNetMobile model layers...")
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top
    print("Adding custom layers on top of the base model...")
    x = Flatten()(base_model.output)
    x = Dense(64, activation='relu')(x)  # Reduced complexity
    x = BatchNormalization()(x)          # Add Batch Normalization
    x = Dropout(0.5)(x)                  # Dropout for regularization
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    
    print(f"Model architecture summary:\n{model.summary()}")
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
