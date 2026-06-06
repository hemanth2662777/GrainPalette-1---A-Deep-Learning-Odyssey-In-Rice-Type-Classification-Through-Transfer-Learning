from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# ===== FIXED MODEL PATH =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "rice_model.h5")

print("Model Path:", MODEL_PATH)

try:
    model = load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully")
except Exception as e:
    print("Model loading failed:", e)
    model = None
class_names = ['arborio', 'basmati', 'ipsala', 'jasmine', 'karacadag']

# ===== USERS FILE PATH =====
USERS_FILE = os.path.join(BASE_DIR, "users.json")

# 🔧 Utility: Load users from JSON
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

# ✅ Root route: Redirect to login or home
@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

# ✅ Home page (only if logged in)
@app.route('/home', methods=['GET', 'POST'])
def home():
    if 'username' not in session:
        return redirect(url_for('login'))

    prediction = None
    confidence = None
    image_path = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded'

        file = request.files['file']
        img_path = os.path.join('static', file.filename)
        file.save(img_path)

        # Preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0

        # Predict
        preds = model.predict(x)
        pred_class = class_names[np.argmax(preds)]
        confidence = np.max(preds) * 100

        return render_template('home.html',
                               prediction=pred_class,
                               confidence=round(confidence, 2),
                               image_path=img_path)

    return render_template('home.html')

# ✅ Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        users = load_users()
        username = request.form['username']
        password = request.form['password']

        if username in users:
            return render_template('register.html', error="Username already exists!")

        users[username] = password
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f)

        return redirect(url_for('login'))

    return render_template('register.html')

# ✅ Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        users = load_users()
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid credentials!")

    return render_template('login.html')

# ✅ Logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# ✅ Run app
if __name__ == '__main__':
    app.run(debug=True)
