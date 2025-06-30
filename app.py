import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
import requests
import time
import sqlite3
from datetime import datetime
import random
import math
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash

# --- MySQL Config (update with your credentials) ---
MYSQL_HOST = 'localhost'
MYSQL_USER = 'root'
MYSQL_PASSWORD = ''
MYSQL_DB = 'supply_chain'

# --- Config ---
NEWS_API_KEY = '0609f46bf58746b1914450b1b4768c37'
WEATHER_API_KEY = '17bfbb7848815abacf2aa74ed198c46c'
DB_FILE = 'simulation_logs.db'
ORS_API_KEY = '5b3ce3597851110001cf62486680af0eb1db44bebdfc3f67e66a484c'
CITY_COORDS = {
    'mumbai': [19.076, 72.8777],
    'delhi': [28.6139, 77.209],
    'bangalore': [12.9716, 77.5946],
    'chennai': [13.0827, 80.2707],
    'kolkata': [22.5726, 88.3639],
    'hyderabad': [17.385, 78.4867],
    'pune': [18.5204, 73.8567],
    'ahmedabad': [23.0225, 72.5714]
}

app = Flask(__name__)
app.secret_key = '1234567890'  # Set this to a strong, random value in production
CORS(app)

# --- SQLite Logging Setup ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS simulation_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        input_data TEXT,
        predictions TEXT
    )''')
    conn.commit()
    conn.close()
init_db()

def log_simulation(input_data, predictions):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('INSERT INTO simulation_logs (timestamp, input_data, predictions) VALUES (?, ?, ?)',
              (datetime.now().isoformat(), str(input_data), str(predictions)))
    conn.commit()
    conn.close()

# --- Load ML Models (with fallback logic) ---
def load_models():
    models = {}
    model_files = {
        'inventory_stockout': 'inventory_stockout_model.pkl',
        'lead_time_anomaly': 'lead_time_anomaly_model.pkl',
        'product_condition': 'product_condition_model.pkl',
        'route_delay': 'route_delay_rf_model.pkl'
    }
    for name, file in model_files.items():
        try:
            with open(file, 'rb') as f:
                models[name] = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Could not load {name}: {e}")
            models[name] = None
    return models

models = load_models()

# --- Fallback Heuristics for Each Model ---
def predict_inventory_stockout_fallback(data):
    stock = float(data.get('current_stock', 1000))
    reorder = float(data.get('reorder_point', 500))
    sales = float(data.get('sales_rate', 100))
    risk = min(1.0, max(0.0, (reorder - stock + sales) / (reorder + 1)))
    return int(risk > 0.5), risk

def predict_lead_time_anomaly_fallback(data):
    planned = float(data.get('planned_lead_time', 48))
    actual = float(data.get('actual_lead_time', planned))
    if actual <= planned:
        risk = 0.0
    else:
        deviation = (actual - planned) / (planned + 1)
        risk = min(1.0, deviation * 2)
    return int(risk > 0.3), risk

def predict_product_condition_fallback(data):
    temp = float(data.get('temperature', 25))
    humidity = float(data.get('humidity', 50))
    delay = float(data.get('handling_delay', 0))
    risk = 0.2 + 0.02 * (temp - 25) + 0.01 * (humidity - 50) + 0.05 * delay
    risk = min(1.0, max(0.0, risk))
    return int(risk > 0.5), risk

def predict_route_delay_fallback(data):
    traffic = int(data.get('traffic_level', 2))
    weather = int(data.get('weather_severity', 1))
    dist = float(data.get('distance', 100))
    risk = 0.2 + 0.1 * (traffic - 2) + 0.05 * (weather - 1) + 0.001 * dist
    risk = min(1.0, max(0.0, risk))
    return int(risk > 0.4), risk

# --- Weather API Integration ---
def fetch_weather(city):
    try:
        url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric'
        r = requests.get(url, timeout=5)
        data = r.json()
        return {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'condition': data['weather'][0]['main'],
            'severity': 2 if data['weather'][0]['main'] in ['Rain', 'Storm', 'Thunderstorm'] else 1
        }
    except Exception as e:
        print(f"[WARN] Weather API failed: {e}")
        return {'temperature': 30, 'humidity': 60, 'condition': 'Clear', 'severity': 1}

# --- News API Integration ---
def fetch_news(product, city):
    try:
        # Ensure product name is not empty
        if not product or product.lower() == 'product':
            return {'headline': 'Please specify a product name for news', 'url': '', 'city': city, 'product': product}
        
        # Build a more specific query with location context and language filter
        # Add supply chain related terms to improve relevance
        query = f'("{product}" AND "{city}") AND (supply OR logistics OR transport OR business OR market OR industry)'
        url = (f'https://newsapi.org/v2/everything'
               f'?q={query}'
               f'&apiKey={NEWS_API_KEY}'
               f'&language=en'  # English only
               f'&sortBy=relevancy'  # Sort by relevance instead of date
               f'&pageSize=3')  # Get top 3 most relevant articles
        
        r = requests.get(url, timeout=5)
        data = r.json()
        
        if data['status'] == 'ok' and data['totalResults'] > 0:
            # Filter articles to ensure they are truly relevant
            relevant_articles = []
            for article in data['articles']:
                title = article['title'].lower()
                # Check if both product and city (or related terms) are mentioned in the title
                if (product.lower() in title or any(term in title for term in ['supply', 'logistics', 'transport', 'market'])):
                    relevant_articles.append({
                        'headline': article['title'],
                        'url': article['url'],
                        'city': city,
                        'product': product
                    })
                if len(relevant_articles) > 0:
                    return relevant_articles[0]  # Return the most relevant article
            
            # If no filtered articles, return the first one from the API
            return {
                'headline': data['articles'][0]['title'],
                'url': data['articles'][0]['url'],
                'city': city,
                'product': product
            }
        else:
            return {
                'headline': f'No recent news found for {product} in {city}',
                'url': '',
                'city': city,
                'product': product
            }
    except Exception as e:
        print(f"[WARN] News API failed for {product} in {city}: {e}")
        return {
            'headline': f'Unable to fetch news for {product} in {city}',
            'url': '',
            'city': city,
            'product': product
        }

# --- Vehicle speed table (all +20 km/h)
VEHICLE_SPEEDS = {
    'Bike': 60,              # 40+20
    'Van': 70,               # 50+20
    'Mini Truck': 65,        # 45+20
    'Truck': 60,             # 40+20
    'Container': 60,         # 40+20
    'Refrigerated Truck': 58,# 38+20
    'Frozen Truck': 55       # 35+20
}

# --- Vehicle cost per km (in ₹) ---
VEHICLE_COSTS = {
    'Bike': 5,
    'Van': 12,
    'Mini Truck': 15,
    'Truck': 20,
    'Container': 25,
    'Refrigerated Truck': 30,
    'Frozen Truck': 35
}

def calculate_transport_cost(distance, vehicle_type, lead_time_hrs):
    """Calculate transport cost based on distance, vehicle type and lead time"""
    base_cost_per_km = VEHICLE_COSTS.get(vehicle_type, 20)  # default to truck cost
    # Base cost from distance
    base_cost = distance * base_cost_per_km
    # Additional cost for longer lead times (10% extra per day)
    days = lead_time_hrs / 24
    time_factor = 1 + (0.1 * days)
    return base_cost * time_factor

# --- Simulated Traffic API ---
def fetch_traffic(city, distance_km, traffic_level, weather_severity, vehicle_type):
    # Use speed based on vehicle type, default to 60 if unknown
    base_speed = VEHICLE_SPEEDS.get(vehicle_type, 60)
    base_time_hr = distance_km / base_speed if base_speed > 0 else 0
    traffic_delay_factor = 0.15 if traffic_level == 3 else (0.05 if traffic_level == 2 else 0.0)
    weather_delay_factor = 0.10 if weather_severity == 2 else 0.0
    total_time_hr = base_time_hr * (1 + traffic_delay_factor + weather_delay_factor)
    estimated_travel_time = int(total_time_hr * 60)  # in minutes
    return {
        'level': traffic_level,
        'description': random.choice(['Smooth', 'Moderate', 'Congested']),
        'estimated_travel_time': estimated_travel_time,
        'average_speed': base_speed,
        'incidents': random.choice([0, 1, 2]),
        'trend': random.choice(['Improving', 'Stable', 'Worsening'])
    }

# --- Rule-based Optimization ---
def get_optimizations(preds):
    recs = []
    if preds['route_delay']['risk'] > 0.5:
        recs.append('Suggest alternate route due to high delay risk.')
    if preds['product_condition']['risk'] > 0.5:
        recs.append('Use cold chain transport for product safety.')
    if preds['inventory_stockout']['risk'] > 0.5:
        recs.append('Trigger replenishment order for warehouse.')
    if preds['lead_time_anomaly']['risk'] > 0.5:
        recs.append('Notify stakeholders: lead time anomaly likely.')
    return recs

# --- Advanced Vehicle Selection Logic ---
def select_vehicle(data):
    load_weight = float(data.get('load_weight', 500))
    temp_sensitive = int(data.get('temperature_sensitive', 0))
    # Temperature-based override
    if temp_sensitive == 2:
        return 'Frozen Truck'
    if temp_sensitive == 1:
        return 'Refrigerated Truck'
    # Load-based selection
    if load_weight <= 50:
        return 'Bike'
    elif load_weight <= 500:
        return 'Van'
    elif load_weight <= 2000:
        return 'Mini Truck'
    elif load_weight <= 7000:
        return 'Truck'
    else:
        return 'Container'

# --- Flask-Login Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- MySQL Connection Helper ---
def get_mysql_connection():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB
    )

# --- User Model for Flask-Login ---
class User(UserMixin):
    def __init__(self, id_, company_name, gmail, username, password_hash):
        self.id = id_
        self.company_name = company_name
        self.gmail = gmail
        self.username = username
        self.password_hash = password_hash

    @staticmethod
    def get(user_id):
        conn = get_mysql_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))
        user = cursor.fetchone()
        conn.close()
        if user:
            return User(user['id'], user['company_name'], user['gmail'], user['username'], user['password'])
        return None

    @staticmethod
    def find_by_username_or_gmail(identifier):
        conn = get_mysql_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM users WHERE username = %s OR gmail = %s', (identifier, identifier))
        user = cursor.fetchone()
        conn.close()
        if user:
            return User(user['id'], user['company_name'], user['gmail'], user['username'], user['password'])
        return None

    @staticmethod
    def find_by_username(username):
        conn = get_mysql_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        user = cursor.fetchone()
        conn.close()
        if user:
            return User(user['id'], user['company_name'], user['gmail'], user['username'], user['password'])
        return None

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

# --- Login Route ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        identifier = request.form['identifier']  # username or gmail
        password = request.form['password']
        user = User.find_by_username_or_gmail(identifier)
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('index'))
        return render_template('login.html', error='Invalid username/gmail or password')
    return render_template('login.html')

# --- Logout Route ---
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# --- Register Route (backend only, no UI) ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        company_name = request.form['company_name']
        gmail = request.form['gmail']
        username = request.form['username']
        password = request.form['password']
        if User.find_by_username(username):
            return render_template('login.html', error='Username already exists', register=True)
        password_hash = generate_password_hash(password)
        conn = get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO users (company_name, gmail, username, password) VALUES (%s, %s, %s, %s)',
            (company_name, gmail, username, password_hash)
        )
        conn.commit()
        conn.close()
        user = User.find_by_username(username)
        login_user(user)
        return redirect(url_for('profile'))
    return render_template('login.html', register=True)

# --- Protect Dashboard ---
@app.route('/')
@login_required
def index():
    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('SELECT * FROM user_profiles WHERE user_id = %s', (current_user.id,))
    profile = cursor.fetchone()
    conn.close()
    return render_template('index.html', profile=profile)

# --- Unified Prediction API ---
@app.route('/api/predict/disruption', methods=['POST'])
def predict_disruption():
    data = request.json
    # Get city names
    mfg = data.get('manufacturing_plant', 'mumbai')
    wh = data.get('warehouse_location', 'mumbai')
    seller = data.get('seller_address', 'mumbai')
    dest = data.get('delivery_destination', 'mumbai')
    print(f"Cities: mfg={mfg}, wh={wh}, seller={seller}, dest={dest}")
    # Get coordinates
    points = [mfg, wh, seller, dest]
    coords = [CITY_COORDS.get(city, CITY_COORDS['mumbai']) for city in points]
    print(f"Coords: {coords}")
    ors_coords = [[c[1], c[0]] for c in coords]  # ORS expects [lon, lat]
    print(f"ORS coords: {ors_coords}")
    # Try ORS API for driving distance
    distance = ors_route_distance(ors_coords)
    distance_type = 'driving (ORS)'
    print(f"ORS distance: {distance}")
    if distance is None:
        # Fallback: sum of straight-line distances between each leg
        distance = sum(haversine(coords[i], coords[i+1]) for i in range(len(coords)-1))
        distance_type = 'straight-line (haversine)'
        print(f"Haversine fallback distance: {distance}")
    # Final fallback
    if distance is None or not isinstance(distance, (int, float)) or distance <= 0:
        distance = 100
        distance_type = 'default (hardcoded)'
        print(f"Default fallback distance: {distance}")
    print(f"Final distance: {distance} km (type: {distance_type})")
    # Fetch weather for all locations
    mfg_weather = fetch_weather(mfg)
    wh_weather = fetch_weather(wh)
    seller_weather = fetch_weather(seller)
    dest_weather = fetch_weather(dest)
    def get_rain(w):
        if 'rain' in w:
            return w['rain']
        elif 'rainfall' in w:
            return w['rainfall']
        else:
            return 0
    weather = {
        'manufacturing_plant': {
            'city': mfg,
            'temperature': mfg_weather.get('temperature', 0),
            'humidity': mfg_weather.get('humidity', 0),
            'condition': mfg_weather.get('condition', ''),
            'rainfall': get_rain(mfg_weather)
        },
        'warehouse': {
            'city': wh,
            'temperature': wh_weather.get('temperature', 0),
            'humidity': wh_weather.get('humidity', 0),
            'condition': wh_weather.get('condition', ''),
            'rainfall': get_rain(wh_weather)
        },
        'seller': {
            'city': seller,
            'temperature': seller_weather.get('temperature', 0),
            'humidity': seller_weather.get('humidity', 0),
            'condition': seller_weather.get('condition', ''),
            'rainfall': get_rain(seller_weather)
        },
        'destination': {
            'city': dest,
            'temperature': dest_weather.get('temperature', 0),
            'humidity': dest_weather.get('humidity', 0),
            'condition': dest_weather.get('condition', ''),
            'rainfall': get_rain(dest_weather)
        }
    }
    # News for all nodes
    news_nodes = [
        (data.get('product_name', 'Product'), mfg),
        (data.get('product_name', 'Product'), wh),
        (data.get('product_name', 'Product'), seller),
        (data.get('product_name', 'Product'), dest)
    ]
    news_list = []
    seen_headlines = set()
    for prod, city in news_nodes:
        news_item = fetch_news(prod, city)
        if news_item['headline'] not in seen_headlines:
            news_list.append(news_item)
            seen_headlines.add(news_item['headline'])
    # Simulate traffic for the seller to destination segment
    traffic_level = int(data.get('traffic_level', 2))
    weather_severity = dest_weather.get('severity', 1)
    seller_coords = CITY_COORDS.get(seller, CITY_COORDS['mumbai'])
    dest_coords = CITY_COORDS.get(dest, CITY_COORDS['mumbai'])
    # Use ORS API for driving distance between seller and destination
    seller_dest_ors_coords = [[seller_coords[1], seller_coords[0]], [dest_coords[1], dest_coords[0]]]
    seller_dest_distance = ors_route_distance(seller_dest_ors_coords)
    if seller_dest_distance is None:
        seller_dest_distance = haversine(seller_coords, dest_coords)
    vehicle = select_vehicle(data)
    traffic = fetch_traffic(dest, seller_dest_distance, traffic_level, weather_severity, vehicle)
    # Calculate actual lead time based on expected, weather, and traffic
    expected_lead_time = float(data.get('planned_lead_time', 48))
    current_weather_delay_factor = 0.10 if dest_weather['severity'] == 2 else 0.0
    current_traffic_delay_factor = 0.15 if traffic['level'] == 3 else (0.05 if traffic['level'] == 2 else 0.0)
    random_noise = random.uniform(-0.03, 0.03)
    base_speed = VEHICLE_SPEEDS.get(vehicle, 60)
    base_time_hr = seller_dest_distance / base_speed if base_speed > 0 else 0
    actual_lead_time = base_time_hr * (1 + current_weather_delay_factor + current_traffic_delay_factor + random_noise)
    actual_lead_time = max(actual_lead_time, expected_lead_time)
    # Inventory Stockout
    inv_input = {
        'current_stock': float(data.get('current_stock', 1000)),
        'reorder_point': float(data.get('reorder_point', 500)),
        'sales_rate': float(data.get('sales_rate', 100))
    }

    # Use ML model if available, else fallback to our function
    if models.get('inventory_stockout'):
        try:
            X = np.array([[inv_input['current_stock'], inv_input['reorder_point'], inv_input['sales_rate']]])
            pred = models['inventory_stockout'].predict(X)[0]
            risk = float(getattr(models['inventory_stockout'], 'predict_proba', lambda x: [[0.3,0.7]])(X)[0][1])
        except Exception:
            pred, risk = predict_inventory_stockout(inv_input['current_stock'], inv_input['reorder_point'], inv_input['sales_rate'])
    else:
        pred, risk = predict_inventory_stockout(inv_input['current_stock'], inv_input['reorder_point'], inv_input['sales_rate'])

    inventory_stockout = {'prediction': int(pred), 'risk': float(risk)}
    # Lead Time Anomaly
    lt_input = {
        'planned_lead_time': expected_lead_time,
        'actual_lead_time': actual_lead_time
    }
    if models['lead_time_anomaly']:
        try:
            X = np.array([[float(lt_input['planned_lead_time']), float(lt_input['actual_lead_time'])]])
            pred = models['lead_time_anomaly'].predict(X)[0]
            risk = float(getattr(models['lead_time_anomaly'], 'predict_proba', lambda x: [[0.3,0.7]])(X)[0][1])
        except Exception:
            pred, risk = predict_lead_time_anomaly_fallback(lt_input)
    else:
        pred, risk = predict_lead_time_anomaly_fallback(lt_input)
    lead_time_anomaly = {'prediction': int(pred), 'risk': float(risk)}
    # Product Condition
    pc_input = {
        'temperature': dest_weather['temperature'],
        'humidity': dest_weather['humidity'],
        'handling_delay': data.get('handling_delay', 0)
    }
    if models['product_condition']:
        try:
            X = np.array([[float(pc_input['temperature']), float(pc_input['humidity']), float(pc_input['handling_delay'])]])
            pred = models['product_condition'].predict(X)[0]
            risk = float(getattr(models['product_condition'], 'predict_proba', lambda x: [[0.3,0.7]])(X)[0][1])
        except Exception:
            pred, risk = predict_product_condition_fallback(pc_input)
    else:
        pred, risk = predict_product_condition_fallback(pc_input)
    product_condition = {'prediction': int(pred), 'risk': float(risk)}
    # Route Delay
    rd_input = {
        'traffic_level': traffic['level'],
        'weather_severity': dest_weather['severity'],
        'distance': distance
    }
    if models['route_delay']:
        try:
            X = np.array([[float(rd_input['traffic_level']), float(rd_input['weather_severity']), float(rd_input['distance'])]])
            pred = models['route_delay'].predict(X)[0]
            risk = float(getattr(models['route_delay'], 'predict_proba', lambda x: [[0.3,0.7]])(X)[0][1])
        except Exception:
            pred, risk = predict_route_delay_fallback(rd_input)
    else:
        pred, risk = predict_route_delay_fallback(rd_input)
    route_delay = {'prediction': int(pred), 'risk': float(risk)}
    # Optimization
    preds = {
        'inventory_stockout': inventory_stockout,
        'lead_time_anomaly': lead_time_anomaly,
        'product_condition': product_condition,
        'route_delay': route_delay
    }
    optimizations = get_optimizations(preds)
    # --- Current Plan Calculation ---
    current_points = [mfg, wh, seller, dest]
    current_coords = [CITY_COORDS.get(city, CITY_COORDS['mumbai']) for city in current_points]
    current_ors_coords = [[c[1], c[0]] for c in current_coords]
    current_distance = ors_route_distance(current_ors_coords)
    if current_distance is None:
        current_distance = sum(haversine(current_coords[i], current_coords[i+1]) for i in range(len(current_coords)-1))
    current_vehicle = select_vehicle(data)
    # Traffic/weather for current plan
    current_traffic_level = int(data.get('traffic_level', 2))
    current_weather_severity = dest_weather.get('severity', 1)
    current_seller_coords = CITY_COORDS.get(seller, CITY_COORDS['mumbai'])
    current_dest_coords = CITY_COORDS.get(dest, CITY_COORDS['mumbai'])
    current_seller_dest_distance = haversine(current_seller_coords, current_dest_coords)
    current_traffic = fetch_traffic(dest, current_seller_dest_distance, current_traffic_level, current_weather_severity, current_vehicle)
    # Lead time for current plan (use user input for expected, and route/vehicle for actual)
    current_expected_lead_time = float(data.get('planned_lead_time', 48))
    current_base_speed = VEHICLE_SPEEDS.get(current_vehicle, 60)
    current_base_time_hr = current_distance / current_base_speed if current_base_speed > 0 else 0
    current_noise_hr = random.uniform(-1, 1)  # at most +/- 1 hour
    current_actual_lead_time = max(current_base_time_hr * (1 + current_weather_delay_factor + current_traffic_delay_factor) + current_noise_hr, current_base_time_hr)
    # --- Optimized Plan Calculation ---
    # Simulate: if route_delay risk > 0.5, change warehouse; if product_condition risk > 0.5, change vehicle
    opt_wh = wh
    opt_vehicle = current_vehicle
    if route_delay['risk'] > 0.5:
        opt_wh = 'pune' if wh != 'pune' else 'delhi'
    if product_condition['risk'] > 0.5:
        if int(data.get('temperature_sensitive', 0)) < 2:
            opt_vehicle = 'Refrigerated Truck'
        else:
            opt_vehicle = 'Frozen Truck'
    opt_points = [mfg, opt_wh, seller, dest]
    opt_coords = [CITY_COORDS.get(city, CITY_COORDS['mumbai']) for city in opt_points]
    opt_ors_coords = [[c[1], c[0]] for c in opt_coords]
    opt_distance = ors_route_distance(opt_ors_coords)
    if opt_distance is None:
        opt_distance = sum(haversine(opt_coords[i], opt_coords[i+1]) for i in range(len(opt_coords)-1))
    opt_traffic = fetch_traffic(dest, opt_distance, current_traffic_level, current_weather_severity, opt_vehicle)
    opt_expected_lead_time = float(data.get('planned_lead_time', 48))
    opt_base_speed = VEHICLE_SPEEDS.get(opt_vehicle, 60)
    opt_base_time_hr = opt_distance / opt_base_speed if opt_base_speed > 0 else 0
    opt_noise_hr = random.uniform(-1, 1)
    opt_weather_delay_factor = 0.10 if dest_weather['severity'] == 2 else 0.0
    opt_traffic_delay_factor = 0.15 if opt_traffic['level'] == 3 else (0.05 if opt_traffic['level'] == 2 else 0.0)
    opt_actual_lead_time = max(opt_base_time_hr * (1 + opt_weather_delay_factor + opt_traffic_delay_factor) + opt_noise_hr, opt_base_time_hr)
    # --- Edge Case: If optimized plan is not better or is the same, keep current plan as optimized ---
    # if (opt_actual_lead_time >= current_actual_lead_time) or (opt_points == current_points):
    #     opt_points = current_points
    #     opt_distance = current_distance
    #     opt_vehicle = current_vehicle
    #     opt_expected_lead_time = current_expected_lead_time
    #     opt_actual_lead_time = current_actual_lead_time
    # --- Response ---
    # Lead Time Anomaly for current plan
    lt_input_current = {
        'planned_lead_time': current_expected_lead_time,
        'actual_lead_time': current_actual_lead_time
    }
    if models['lead_time_anomaly']:
        try:
            X = np.array([[float(lt_input_current['planned_lead_time']), float(lt_input_current['actual_lead_time'])]])
            pred = models['lead_time_anomaly'].predict(X)[0]
            risk = float(getattr(models['lead_time_anomaly'], 'predict_proba', lambda x: [[0.3,0.7]])(X)[0][1])
        except Exception:
            pred, risk = predict_lead_time_anomaly_fallback(lt_input_current)
    else:
        pred, risk = predict_lead_time_anomaly_fallback(lt_input_current)
    lead_time_anomaly_current = {'prediction': int(pred), 'risk': float(risk)}
    # Lead Time Anomaly for optimized plan
    lt_input_opt = {
        'planned_lead_time': opt_expected_lead_time,
        'actual_lead_time': opt_actual_lead_time
    }
    if models['lead_time_anomaly']:
        try:
            X = np.array([[float(lt_input_opt['planned_lead_time']), float(lt_input_opt['actual_lead_time'])]])
            pred = models['lead_time_anomaly'].predict(X)[0]
            risk = float(getattr(models['lead_time_anomaly'], 'predict_proba', lambda x: [[0.3,0.7]])(X)[0][1])
        except Exception:
            pred, risk = predict_lead_time_anomaly_fallback(lt_input_opt)
    else:
        pred, risk = predict_lead_time_anomaly_fallback(lt_input_opt)
    lead_time_anomaly_opt = {'prediction': int(pred), 'risk': float(risk)}
    # --- Response ---
    # When preparing values for jsonify, round all float values to 2 decimal places
    def fmt(val):
        return round(val, 2) if isinstance(val, float) else val
    # Format current plan
    current_cost = calculate_transport_cost(current_distance, current_vehicle, current_actual_lead_time)
    current_plan = {
        'route': current_points,
        'distance': fmt(current_distance),
        'vehicle': current_vehicle,
        'expected_lead_time': fmt(current_expected_lead_time),
        'actual_lead_time': fmt(current_actual_lead_time),
        'lead_time_anomaly': {
            'prediction': int(lead_time_anomaly_current['prediction']),
            'risk': fmt(lead_time_anomaly_current['risk'])
        },
        'cost': fmt(current_cost)
    }
    # Format optimized plan
    optimized_cost = calculate_transport_cost(opt_distance, opt_vehicle, opt_actual_lead_time)
    optimized_plan = {
        'route': opt_points,
        'distance': fmt(opt_distance),
        'vehicle': opt_vehicle,
        'expected_lead_time': fmt(opt_expected_lead_time),
        'actual_lead_time': fmt(opt_actual_lead_time),
        'lead_time_anomaly': {
            'prediction': int(lead_time_anomaly_opt['prediction']),
            'risk': fmt(lead_time_anomaly_opt['risk'])
        },
        'cost': fmt(optimized_cost)
    }
    # Format risk outputs
    inventory_stockout = {'prediction': int(inventory_stockout['prediction']), 'risk': fmt(inventory_stockout['risk'])}
    product_condition = {'prediction': int(product_condition['prediction']), 'risk': fmt(product_condition['risk'])}
    route_delay = {'prediction': int(route_delay['prediction']), 'risk': fmt(route_delay['risk'])}
    return jsonify({
        'current_plan': current_plan,
        'optimized_plan': optimized_plan,
        'inventory_stockout': inventory_stockout,
        'product_condition': product_condition,
        'route_delay': route_delay,
        'optimizations': optimizations,
        'weather': weather,
        'news': news_list,
        'traffic': traffic,
        'vehicle': current_vehicle,
        'optimized_vehicle': opt_vehicle,
        'expected_lead_time': fmt(current_expected_lead_time),
        'actual_lead_time': fmt(current_actual_lead_time),
        'distance': fmt(current_distance),
        'distance_type': distance_type
    })

# --- Endpoint to fetch simulation logs ---
@app.route('/api/logs', methods=['GET'])
def get_logs():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT timestamp, input_data, predictions FROM simulation_logs ORDER BY id DESC LIMIT 50')
    logs = [
        {'timestamp': row[0], 'input_data': row[1], 'predictions': row[2]}
        for row in c.fetchall()
    ]
    conn.close()
    return jsonify({'logs': logs})

def haversine(coord1, coord2):
    # Calculate the great-circle distance between two points
    R = 6371  # Earth radius in km
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def ors_route_distance(coords):
    # coords: list of [lon, lat] pairs (ORS expects lon, lat)
    try:
        url = 'https://api.openrouteservice.org/v2/directions/driving-car'
        headers = {'Authorization': ORS_API_KEY, 'Content-Type': 'application/json'}
        body = {"coordinates": coords}
        r = requests.post(url, json=body, headers=headers, timeout=10)
        data = r.json()
        if 'routes' in data and data['routes']:
            return data['routes'][0]['summary']['distance'] / 1000  # meters to km
    except Exception as e:
        print(f"[WARN] ORS API failed: {e}")
    return None

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('SELECT * FROM user_profiles WHERE user_id = %s', (current_user.id,))
    profile = cursor.fetchone()
    if request.method == 'POST':
        # Handle multi-select fields as comma-separated strings
        manufacturing_plant = ','.join(request.form.getlist('manufacturing_plant'))
        seller = ','.join(request.form.getlist('seller'))
        warehouse = ','.join(request.form.getlist('warehouse'))
        capacity = request.form['capacity']
        cpg_product_type = ','.join(request.form.getlist('cpg_product_type'))
        vehicles = ','.join(request.form.getlist('vehicles'))  # Multi-select
        if profile:
            cursor.execute(
                'UPDATE user_profiles SET manufacturing_plant=%s, seller=%s, warehouse=%s, capacity=%s, cpg_product_type=%s, vehicles=%s WHERE user_id=%s',
                (manufacturing_plant, seller, warehouse, capacity, cpg_product_type, vehicles, current_user.id)
            )
        else:
            cursor.execute(
                'INSERT INTO user_profiles (user_id, manufacturing_plant, seller, warehouse, capacity, cpg_product_type, vehicles) VALUES (%s, %s, %s, %s, %s, %s, %s)',
                (current_user.id, manufacturing_plant, seller, warehouse, capacity, cpg_product_type, vehicles)
            )
        conn.commit()
        conn.close()
        return redirect(url_for('index'))
    conn.close()
    return render_template('profile.html', profile=profile)

@app.route('/statistics')
@login_required
def statistics():
    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('SELECT * FROM user_profiles WHERE user_id = %s', (current_user.id,))
    profile = cursor.fetchone()
    conn.close()
    return render_template('statistics.html', profile=profile)

def predict_inventory_stockout(current_stock, reorder_point, sales_rate):
    """
    Predicts the risk of inventory stockout.
    Returns (prediction, risk_score):
      - prediction: 1 if stockout risk is high, 0 otherwise
      - risk_score: float between 0 and 1
    """
    # Heuristic: risk increases as stock approaches or drops below reorder point
    # and as sales rate increases
    if reorder_point <= 0:
        return 0, 0.0  # No reorder point set, assume no risk

    # Calculate days until stockout
    days_until_stockout = (current_stock - reorder_point) / (sales_rate + 1e-6)
    # Normalize risk: if days_until_stockout < 0, risk is 1; if > 10, risk is 0
    risk = max(0.0, min(1.0, (10 - days_until_stockout) / 10))
    prediction = int(risk > 0.5)
    return prediction, risk

if __name__ == '__main__':
    app.run(debug=True) 