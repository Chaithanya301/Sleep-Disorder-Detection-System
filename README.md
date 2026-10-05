# Sleep-Disorder-Detection-System
A Flask-based web application for sleep disorder analysis using machine learning and image-based eye-state detection.
## Project Overview
This project combines machine learning with a Flask web application to analyze sleep-related information and provide sleep disorder predictions.

The application includes:

- Sleep disorder prediction using a trained machine learning model
- Form-based data collection and prediction
- Image/video-based eye-state analysis
- Flask-based web interfaces
- SQLite database integration
## Technologies Used
- Python
- Flask
- TensorFlow
- Keras
- Pillow
- Gunicorn
- SQLite
- HTML/CSS
## Project Structure
```text 
Sleep-Disorder-Classification/
│
├── app.py
├── c.py
├── sqlite.py
├── test.py
├── model.json
├── requirements.txt
│
├── models/
│   └── sleep_disorder_model.pkl
│
└── templates/
    ├── camera.html
    ├── dashboard.html
    ├── data_collection.html
    ├── edashboard.html
    ├── form.html
    ├── index.html
    ├── indexa.html
    ├── main.html
    ├── newform.html
    ├── prediction.html
    ├── result.html
    └── video.html
```
## Key Features
### Sleep Disorder Prediction

Uses a trained machine learning model to process sleep-related input data and generate a predicted sleep disorder category.

### Eye-State Analysis

Includes image/video-based eye-state analysis as part of the sleep-related assessment workflow.

### Web Application

A Flask-based web interface connects the application logic with the HTML templates.

### Database Integration

SQLite is used for application-related data storage and management.

### Machine Learning

The project includes a trained sleep disorder model stored in:

`models/sleep_disorder_model.pkl`

The Flask application loads the trained model and associated preprocessing components to perform predictions.
## Requirements 
The project dependencies are listed in `requirements.txt`:
```text
Flask==1.1.2
gunicorn==20.0.4
Keras==2.4.3
Pillow==7.2.0
tensorflow
```
## Academic Project
This project was developed as part of my final-year B.Tech Computer Science and Engineering (Artificial Intelligence & Machine Learning) academic work.





