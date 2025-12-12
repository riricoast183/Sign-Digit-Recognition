# Learning to Speak Numbers: My Journey with Sign-Digit Recognition

Machine learning is transforming how we interact with technology—from facial recognition to self-driving cars. But could I teach a computer to understand hand gestures representing numbers? That question led me to my latest project: **sign-digit recognition**.  

In this article, I’ll share my journey, the dataset I used, how I built the model, the results I achieved, and the lessons I learned along the way.  

---

## The Spark

Humans can instantly recognize numbers from hand signs, but can a machine do the same?  

I wanted a hands-on project to understand computer vision better and apply machine learning in a practical way. Teaching a computer to read hand gestures seemed like the perfect challenge.  

---

## Dataset

I collected images of hands showing digits from **0 to 9** from the internet and organized them into folders by digit, effectively creating my own labeled dataset. I then split the images into **training** and **testing** sets to ensure the model could generalize well to new, unseen images.  

**Dataset structure:**
dataset/
│
├── train/
│ ├── 0/
│ ├── 1/
│ └── …
│
└── test/
├── 0/
├── 1/
└── …


---

## Preprocessing

Before feeding the data to the model, I had to preprocess it:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Normalize pixel values and split data
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

## Building the Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


Training the Model

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

Results

After training, the model achieved an accuracy of XX% on the test set. It successfully recognized most digits, with occasional misclassifications between similar hand shapes (e.g., 3 vs. 5).

Example Predictions:

Input Image	Predicted Digit

	3

	7

	1

Lessons Learned and Next Steps

This project taught me the importance of clean, well-labeled data, how CNNs recognize patterns, and that patience and iteration are key in machine learning projects. It also opened doors for future work, such as full sign language recognition or gesture-based interfaces.

For those interested in extending the project, I recommend experimenting with techniques like data augmentation to improve model accuracy, exploring advanced architectures such as ResNet or EfficientNet, and implementing real-time recognition using a webcam for interactive applications.


Conclusion

Sign-digit recognition may sound simple, but it’s a powerful way to explore computer vision and machine learning. Projects like this show that with curiosity, coding, and persistence, you can create something meaningful—and even a little magical.

Author: Richard Paculob
Project implemented using a Convolutional Neural Network (CNN) in Python with TensorFlow and Keras.




