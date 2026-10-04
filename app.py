from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import sqlite3
import base64
from PIL import Image
from io import BytesIO
from c import EnhancedDrowsinessDetector
from werkzeug.utils import secure_filename
from datetime import datetime
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'webm'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

detector = EnhancedDrowsinessDetector()

# Load the saved model
with open('models/sleep_disorder_model.pkl', 'rb') as model_file:
    saved_data = pickle.load(model_file)
    model = saved_data['model']
    scaler = saved_data['scaler']
    le_gender = saved_data['le_gender']
    le_occupation = saved_data['le_occupation']
    le_bmi = saved_data['le_bmi']
    le_sleep_disorder = saved_data['le_sleep_disorder']

@app.route('/')
def home():
    # Get unique categories for dropdowns
    genders = le_gender.classes_
    occupations = le_occupation.classes_
    bmi_categories = le_bmi.classes_

    return render_template('index.html', 
                           genders=genders, 
                           occupations=occupations, 
                           bmi_categories=bmi_categories)


@app.route('/dashboard')
def dashboard():
    # Connect to the SQLite database
    conn = sqlite3.connect('sleep_predictions.db')
    cursor = conn.cursor()

    # Fetch the most recent prediction
    recent_prediction = cursor.execute('''
        SELECT * FROM predictions 
        ORDER BY timestamp DESC 
        LIMIT 1
    ''').fetchone()

    # Fetch historical readings (last 3 entries)
    historical_readings = cursor.execute('''
        SELECT timestamp, sleep_duration, quality_of_sleep, 
               predicted_sleep_disorder, 
               ROUND(RANDOM() * 100, 2) as confidence 
        FROM predictions 
        ORDER BY timestamp DESC 
        LIMIT 3
    ''').fetchall()

    # Close the database connection
    conn.close()

    # Prepare the data for the template
    if recent_prediction and historical_readings:
        dashboard_data = {
            'recent_prediction': {
                'disorder': recent_prediction[12] or 'No Disorder',
                'confidence': round(len(historical_readings) * 25, 2),  # Simple confidence calculation
                'risk_level': 'Moderate'
            },
            'health_metrics': {
                'age': recent_prediction[2],
                'gender': recent_prediction[3],
                'sleep_duration': recent_prediction[5],
                'sleep_quality': recent_prediction[6],
                'stress_level': recent_prediction[7],
                'physical_activity': recent_prediction[6],
                'heart_rate': recent_prediction[10],
                'daily_steps': recent_prediction[11],
                'blood_pressure': f"{recent_prediction[9]}/80"  # Assuming diastolic is 80
            },
            'historical_readings': [
                {
                    'date': reading[0],
                    'sleep_duration': reading[1],
                    'sleep_quality': 'High' if reading[2] > 7 else 'Moderate' if reading[2] > 5 else 'Low',
                    'prediction': reading[3],
                    'confidence': reading[4]
                } for reading in historical_readings
            ]
        }
        return render_template('dashboard.html', data=dashboard_data)
    
    # Fallback if no data
    return render_template('dashboard.html', data=None)


@app.route('/form')
def forms():
    return render_template('form.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/newform')
def new_form():

    return render_template('newform.html')


@app.route('/process')
def process():
    d = detector.process_video("uploads/20241219_143842_recording.webm")
    print(d)
    return render_template('newform.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    gender = request.form['gender']
    occupation = request.form['occupation']
    bmi_category = request.form['bmi_category']

    # Encode categorical variables
    gender_encoded = le_gender.transform([gender])[0]
    occupation_encoded = le_occupation.transform([occupation])[0]
    bmi_encoded = le_bmi.transform([bmi_category])[0]

    # Extract systolic BP
    systolic_bp = int(request.form['blood_pressure'].split('/')[0])

    # Collect features for prediction
    features = [
        float(request.form['age']),
        gender_encoded,
        occupation_encoded,
        float(request.form['sleep_duration']),
        float(request.form['quality_of_sleep']),
        float(request.form['physical_activity_level']),
        float(request.form['stress_level']),
        bmi_encoded,
        systolic_bp,
        float(request.form['heart_rate']),
        float(request.form['daily_steps'])
    ]
    
    # Scale the features
    features_scaled = scaler.transform([features])
    
    # Make prediction
    prediction = model.predict(features_scaled)
    prediction_label = le_sleep_disorder.inverse_transform(prediction)[0]
    
    # Connect to SQLite database
    try:
        # Establish database connection
        conn = sqlite3.connect('sleep_predictions.db')
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                age REAL,
                gender TEXT,
                occupation TEXT,
                sleep_duration REAL,
                quality_of_sleep REAL,
                physical_activity_level REAL,
                stress_level REAL,
                bmi_category TEXT,
                systolic_bp INTEGER,
                heart_rate REAL,
                daily_steps REAL,
                predicted_sleep_disorder TEXT
            )
        ''')
        
        # Insert prediction data
        cursor.execute('''
            INSERT INTO predictions (
                age, gender, occupation, sleep_duration, quality_of_sleep, 
                physical_activity_level, stress_level, bmi_category, 
                systolic_bp, heart_rate, daily_steps, predicted_sleep_disorder
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            float(request.form['age']),
            gender,
            occupation,
            float(request.form['sleep_duration']),
            float(request.form['quality_of_sleep']),
            float(request.form['physical_activity_level']),
            float(request.form['stress_level']),
            bmi_category,
            systolic_bp,
            float(request.form['heart_rate']),
            float(request.form['daily_steps']),
            prediction_label
        ))
        
        # Commit changes and close connection
        conn.commit()
    except sqlite3.Error as e:
        # Log the error (you might want to use proper logging in a production app)
        print(f"Database error: {e}")
    finally:
        # Ensure connection is closed
        if conn:
            conn.close()
    
    return render_template('result.html', prediction=prediction_label)


@app.route('/predictvideo', methods=['POST'])
def predict_video():
    try:
        # 1. Handle video file upload
        if 'video' not in request.files:
            return jsonify({'error': 'No video file in request'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400

        # Save video file with secure filename
        filename = secure_filename(video_file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        video_file.save(filepath)

        # 2. Process video and get drowsiness metrics
        video_analysis = detector.process_video(filepath)

        # 3. Process form data
        form_data = {
            'age': float(request.form['age']),
            'gender': request.form['gender'],
            'occupation': request.form['occupation'],
            'sleep_duration': float(request.form['sleep_duration']),
            'quality_of_sleep': float(request.form['quality_of_sleep']),
            'physical_activity_level': float(request.form['physical_activity_level']),
            'stress_level': float(request.form['stress_level']),
            'bmi_category': request.form['bmi_category'],
            'systolic_bp': int(request.form['blood_pressure'].split('/')[0]),
            'heart_rate': float(request.form['heart_rate']),
            'daily_steps': float(request.form['daily_steps'])
        }

        # 4. Prepare features for sleep disorder prediction
        features = [
            form_data['age'],
            le_gender.transform([form_data['gender']])[0],
            le_occupation.transform([form_data['occupation']])[0],
            form_data['sleep_duration'],
            form_data['quality_of_sleep'],
            form_data['physical_activity_level'],
            form_data['stress_level'],
            le_bmi.transform([form_data['bmi_category']])[0],
            form_data['systolic_bp'],
            form_data['heart_rate'],
            form_data['daily_steps']
        ]

        # 5. Make sleep disorder prediction
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)
        predicted_sleep_disorder = le_sleep_disorder.inverse_transform(prediction)[0]

        # 6. Store data in SQLite
        conn = sqlite3.connect('sleep_analysis.db')
        cursor = conn.cursor()

        # Create tables if they don't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                video_filename TEXT,
                age REAL,
                gender TEXT,
                occupation TEXT,
                sleep_duration REAL,
                quality_of_sleep REAL,
                physical_activity_level REAL,
                stress_level REAL,
                bmi_category TEXT,
                systolic_bp INTEGER,
                heart_rate REAL,
                daily_steps REAL,
                predicted_sleep_disorder TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                assessment_id INTEGER,
                drowsiness_state TEXT,
                confidence REAL,
                drowsy_percentage REAL,
                avg_ear REAL,
                avg_mar REAL,
                avg_blink_rate REAL,
                avg_drowsy_time REAL,
                total_frames INTEGER,
                video_duration REAL,
                ear_violations INTEGER,
                mar_violations INTEGER,
                FOREIGN KEY (assessment_id) REFERENCES user_assessments (id)
            )
        ''')

        # Insert user assessment data
        cursor.execute('''
            INSERT INTO user_assessments (
                video_filename, age, gender, occupation, sleep_duration,
                quality_of_sleep, physical_activity_level, stress_level,
                bmi_category, systolic_bp, heart_rate, daily_steps,
                predicted_sleep_disorder
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            filename,
            form_data['age'],
            form_data['gender'],
            form_data['occupation'],
            form_data['sleep_duration'],
            form_data['quality_of_sleep'],
            form_data['physical_activity_level'],
            form_data['stress_level'],
            form_data['bmi_category'],
            form_data['systolic_bp'],
            form_data['heart_rate'],
            form_data['daily_steps'],
            predicted_sleep_disorder
        ))
        
        assessment_id = cursor.lastrowid

        # Insert video analysis data
        cursor.execute('''
            INSERT INTO video_analysis (
                assessment_id, drowsiness_state, confidence, drowsy_percentage,
                avg_ear, avg_mar, avg_blink_rate, avg_drowsy_time,
                total_frames, video_duration, ear_violations, mar_violations
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            assessment_id,
            video_analysis['state'],
            video_analysis['confidence'],
            video_analysis['drowsy_percentage'],
            video_analysis['average_metrics']['ear'],
            video_analysis['average_metrics']['mar'],
            video_analysis['average_metrics']['blink_rate'],
            video_analysis['average_metrics']['drowsy_time'],
            video_analysis['analysis_summary']['total_frames_analyzed'],
            video_analysis['analysis_summary']['video_duration'],
            video_analysis['analysis_summary']['ear_violations'],
            video_analysis['analysis_summary']['mar_violations']
        ))

        conn.commit()

        # 7. Prepare response
        response = {
            'message': 'Analysis completed successfully',
            'video_analysis': video_analysis,
            'sleep_disorder_prediction': predicted_sleep_disorder,
            'assessment_id': assessment_id
        }

        print(response)

        return jsonify(response), 200

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

    finally:
        if 'conn' in locals():
            conn.close()


@app.route('/newdashboard')
def new_dashboard():
    return render_template('edashboard.html')

@app.route('/api/dashboard-data')
def dashboard_data():
    conn = sqlite3.connect('sleep_analysis.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get latest assessment and its video analysis
    cursor.execute('''
        SELECT ua.*, va.*
        FROM user_assessments ua
        LEFT JOIN video_analysis va ON ua.id = va.assessment_id
        ORDER BY ua.timestamp DESC
        LIMIT 1
    ''')
    latest = dict(cursor.fetchone())

    # Get historical data for charts and table
    cursor.execute('''
        SELECT 
            ua.id,
            ua.timestamp,
            ua.sleep_duration,
            ua.quality_of_sleep,
            ua.predicted_sleep_disorder,
            va.drowsiness_state,
            va.drowsy_percentage
        FROM user_assessments ua
        LEFT JOIN video_analysis va ON ua.id = va.assessment_id
        ORDER BY ua.timestamp DESC
        LIMIT 30
    ''')
    history = [dict(row) for row in cursor.fetchall()]

    # Get statistical summaries
    cursor.execute('''
        SELECT 
            AVG(quality_of_sleep) as avg_sleep_quality,
            AVG(sleep_duration) as avg_sleep_duration,
            AVG(drowsy_percentage) as avg_drowsiness
        FROM user_assessments ua
        LEFT JOIN video_analysis va ON ua.id = va.assessment_id
        WHERE ua.timestamp >= datetime('now', '-30 days')
    ''')
    stats = dict(cursor.fetchone())

    conn.close()

    print({
        'latest_assessment': latest,
        'video_analysis': {
            'drowsiness_state': latest.get('drowsiness_state'),
            'confidence': latest.get('confidence'),
            'drowsy_percentage': latest.get('drowsy_percentage'),
            'avg_ear': latest.get('avg_ear'),
            'avg_mar': latest.get('avg_mar'),
            'avg_blink_rate': latest.get('avg_blink_rate'),
            'avg_drowsy_time': latest.get('avg_drowsy_time'),
            'total_frames': latest.get('total_frames'),
            'video_duration': latest.get('video_duration')
        },
        'history': history,
        'statistics': {
            'avg_sleep_quality': round(stats['avg_sleep_quality'], 2) if stats['avg_sleep_quality'] else 0,
            'avg_sleep_duration': round(stats['avg_sleep_duration'], 2) if stats['avg_sleep_duration'] else 0,
            'avg_drowsiness': round(stats['avg_drowsiness'], 2) if stats['avg_drowsiness'] else 0
        }
    })
    return jsonify({
        'latest_assessment': latest,
        'video_analysis': {
            'drowsiness_state': latest.get('drowsiness_state'),
            'confidence': latest.get('confidence'),
            'drowsy_percentage': latest.get('drowsy_percentage'),
            'avg_ear': latest.get('avg_ear'),
            'avg_mar': latest.get('avg_mar'),
            'avg_blink_rate': latest.get('avg_blink_rate'),
            'avg_drowsy_time': latest.get('avg_drowsy_time'),
            'total_frames': latest.get('total_frames'),
            'video_duration': latest.get('video_duration')
        },
        'history': history,
        'statistics': {
            'avg_sleep_quality': round(stats['avg_sleep_quality'], 2) if stats['avg_sleep_quality'] else 0,
            'avg_sleep_duration': round(stats['avg_sleep_duration'], 2) if stats['avg_sleep_duration'] else 0,
            'avg_drowsiness': round(stats['avg_drowsiness'], 2) if stats['avg_drowsiness'] else 0
        }
    })

