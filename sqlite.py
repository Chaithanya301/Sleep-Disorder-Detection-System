import sqlite3
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path='sleep_disorder_data.db'):
        self.db_path = db_path
        self.create_tables()
    
    def create_tables(self):
        """Create necessary tables for data collection"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main user data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            age INTEGER,
            gender TEXT,
            sleep_duration FLOAT,
            sleep_quality INTEGER,
            has_sleep_disorder INTEGER,
            stress_level INTEGER
        )
        ''')
        
        # Facial emotion features table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS facial_emotion_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_data_id INTEGER,
            landmark_symmetry FLOAT,
            emotion_prob_happy FLOAT,
            emotion_prob_sad FLOAT,
            emotion_prob_neutral FLOAT,
            emotion_intensity FLOAT,
            FOREIGN KEY (user_data_id) REFERENCES user_data(id)
        )
        ''')
        
        # Voice emotion features table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS voice_emotion_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_data_id INTEGER,
            pitch_mean FLOAT,
            pitch_variance FLOAT,
            speech_rate FLOAT,
            speech_energy FLOAT,
            tone_valence FLOAT,
            tone_arousal FLOAT,
            audio_file_path TEXT,
            FOREIGN KEY (user_data_id) REFERENCES user_data(id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_user_data(self, data):
        """Insert user's primary data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO user_data 
        (timestamp, age, gender, sleep_duration, sleep_quality, 
        has_sleep_disorder, stress_level)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(), 
            data['age'], 
            data['gender'], 
            data['sleep_duration'], 
            data['sleep_quality'], 
            data['has_sleep_disorder'], 
            data['stress_level']
        ))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return user_id
    
    def insert_facial_emotion_data(self, user_id, facial_data):
        """Insert facial emotion features"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO facial_emotion_data 
        (user_data_id, landmark_symmetry, emotion_prob_happy, 
        emotion_prob_sad, emotion_prob_neutral, emotion_intensity)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            facial_data.get('landmark_symmetry', 0),
            facial_data.get('emotion_prob_happy', 0),
            facial_data.get('emotion_prob_sad', 0),
            facial_data.get('emotion_prob_neutral', 0),
            facial_data.get('emotion_intensity', 0)
        ))
        
        conn.commit()
        conn.close()
    
    def insert_voice_emotion_data(self, user_id, voice_data, audio_path):
        """Insert voice emotion features"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO voice_emotion_data 
        (user_data_id, pitch_mean, pitch_variance, 
        speech_rate, speech_energy, tone_valence, 
        tone_arousal, audio_file_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            voice_data.get('pitch_mean', 0),
            voice_data.get('pitch_variance', 0),
            voice_data.get('speech_rate', 0),
            voice_data.get('speech_energy', 0),
            voice_data.get('tone_valence', 0),
            voice_data.get('tone_arousal', 0),
            audio_path
        ))
        
        conn.commit()
        conn.close()
    
    def get_training_data(self):
        """Retrieve data for model training"""
        conn = sqlite3.connect(self.db_path)
        
        # Complex join to get comprehensive training dataset
        query = '''
        SELECT 
            ud.age, ud.gender, ud.sleep_duration, 
            ud.sleep_quality, ud.stress_level,
            fd.landmark_symmetry, fd.emotion_prob_happy,
            fd.emotion_prob_sad, fd.emotion_prob_neutral,
            fd.emotion_intensity,
            vd.pitch_mean, vd.pitch_variance,
            vd.speech_rate, vd.speech_energy,
            vd.tone_valence, vd.tone_arousal,
            ud.has_sleep_disorder
        FROM user_data ud
        JOIN facial_emotion_data fd ON ud.id = fd.user_data_id
        JOIN voice_emotion_data vd ON ud.id = vd.user_data_id
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df