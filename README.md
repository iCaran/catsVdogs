# Cats vs Dogs Image Classification

This project involves building a deep learning model to classify images of cats and dogs. Using Python and popular libraries, the model is trained to accurately distinguish between images of cats and dogs.

## Tech Stack

- Python
- Keras
- TensorFlow
- NumPy
- OpenCV

## Features

- Data preprocessing with OpenCV
- Model building using Keras and TensorFlow
- Image classification and prediction

## Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/iCaran/catsVdogs.git
   cd catsVdogs
   ```
2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook:
   ```sh
   jupyter notebook Copy_of_cats_and_dogs.ipynb
   ```

## Usage

- Load and preprocess the dataset using OpenCV.
- Train the model using Keras and TensorFlow.
- Evaluate the model's performance on test data.
- Use the model to predict whether new images are of cats or dogs.

## Example

```python
import cv2
import numpy as np
from keras.models import load_model

# Load the model
model = load_model('model.h5')

# Preprocess the image
image = cv2.imread('path_to_image.jpg')
image = cv2.resize(image, (128, 128))
image = np.reshape(image, [1, 128, 128, 3])

# Predict
result = model.predict(image)
print("Cat" if result[0][0] > 0.5 else "Dog")
```

## Resources

- [Project Guide](https://youtu.be/0K4J_PTgysc)

---
