# Basira – AI-Powered Smart Glasses

Basira is an assistive smart-glasses system designed to help blind users

understand their surroundings through audio feedback in Arabic.

The system allows users to select different functions using dedicated

physical buttons, providing an accessible and user-controlled experience.

## Key Features

- Object Recognition

- Personal Object Recognition

- Arabic & English Text Recognition

- Face Recognition

- Face Description

- Time Announcement

- Arabic Text-to-Speech

- Offline Processing

## How It Works

The user selects a function using the physical buttons on the glasses.

Depending on the selected function, Basira captures an image, processes

it using the appropriate AI model, and provides the result through

Arabic audio feedback.

### 1. Object Recognition

The camera captures an image and Basira first checks whether the detected

object matches a previously registered personal object.

If a match is found, the system announces the personal object's name.

Otherwise, YOLOv5 is used for general object detection.

### 2. Text Recognition

The camera captures an image and EasyOCR recognizes Arabic or English text.

The recognized text is then delivered through audio.

### 3. Personal Object Registration

The user provides the object's name through the microphone.

Basira captures multiple images of the object from different views,

extracts feature embeddings using MobileNet, and stores the representation

in the SQLite database.

### 4. Time Announcement

The system retrieves the device's current time and announces it

through Arabic speech.

### 5. Face Recognition

The camera captures an image and the system extracts a facial encoding.

The encoding is compared with registered faces stored in the database

to identify known people.

### 6. Face Description

Basira analyzes visible facial features and provides an audio description

such as face direction, glasses, hair characteristics, and eye color.

## AI Technologies

- YOLOv5 – Object Detection

- EasyOCR – Arabic & English Text Recognition

- MobileNetV2 – Personal Object Embeddings

- Face Recognition – Face Detection and Recognition

- Vosk – Offline Speech Recognition

- eSpeak-NG – Arabic Text-to-Speech

## Hardware

- Raspberry Pi 4

- Raspberry Pi Camera

- USB Microphone

- Earphones

- Power Bank

- Physical Buttons

## Processing

Basira is designed to process the system functions locally on the device,

reducing dependence on cloud services and supporting privacy and offline use.

## Database

SQLite is used to store:

- Personal object embeddings

- Registered face encodings

- Object translation mappings

## Testing

Basira was evaluated across 110 test cases:

- Object Recognition: 96.7%

- Personal Object Recognition: 95%

- Text Reading (OCR): 88%

- Face Recognition: 95%

- Time Announcement: 100%

- Overall Success Rate: 94.5%

## Project

Basira aims to provide blind users with a more accessible and personalized

way to obtain visual information through audio.
