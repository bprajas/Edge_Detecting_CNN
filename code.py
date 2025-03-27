import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or cannot be opened: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0 
    edges = edges / 255.0
    
    input_tensor = np.dstack((image, edges))
    return input_tensor, edges  

def edge_detection_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),  
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    ])
    return model

def save_results(original_image, edge_map, model_output, save_path):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(edge_map, cmap='gray')
    plt.title("Edge Map (Canny)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(model_output, cmap='gray')
    plt.title("Model Output")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    image_path = 'C:\\Users\\praja_mfg91s7\\OneDrive\\Pictures\\freepik.png'
    weights_path = 'model.weights.h5' 
    save_path = 'final_output.png' 
    
    input_tensor, edge_map = preprocess_image(image_path)
    original_image = input_tensor[:, :, :3] 
    
    training_data = np.expand_dims(input_tensor, axis=0) 
    labels = np.expand_dims(edge_map, axis=(0, -1)) 

    input_shape = training_data.shape[1:]
    model = edge_detection_model(input_shape)

    directory = os.path.dirname(weights_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    try:
        model.load_weights(weights_path)
        print(f"Loaded pre-trained weights from {weights_path}")
    except Exception as e:
        print(f"No pre-trained weights found. Starting fresh. ({e})")

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(training_data, labels, epochs=100, batch_size=1)
    
    model.save_weights(weights_path)
    print(f"Model weights saved to {weights_path}")
    
    model_output = model.predict(training_data)[0, :, :, 0]
    
    save_results(original_image, edge_map, model_output, save_path)
