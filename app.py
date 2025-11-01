# 1. importer les librairies nécessaires :
import os
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, Response
from werkzeug.utils import secure_filename

# 2. importer Flask-Login et Flask-SQLAlchemy pour gérer les utilisateurs :
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, login_user, logout_user,
    login_required, current_user, UserMixin
)

# 3. importer pour hasher et vérifier les mots de passe :
from werkzeug.security import generate_password_hash, check_password_hash

# 4. importer Keras pour charger le modèle d’IA :
from tensorflow.keras.models import load_model


# ============================================================
# CONFIGURATION
# ============================================================

# 5. créer une application Flask :
app = Flask(__name__)

# 6. définir une clé secrète pour sécuriser les sessions :
app.config['SECRET_KEY'] = 'replace_this_with_random_secret'

# 7. définir un dossier où les images uploadées seront stockées :
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# 8. configurer la base de données SQLite :
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'

# 9. désactiver les notifications inutiles de SQLAlchemy :
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 10. créer automatiquement le dossier d’upload s’il n’existe pas :
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 11. initialiser la base de données :
db = SQLAlchemy(app)

# 12. initialiser le gestionnaire de connexion (login) :
login_manager = LoginManager(app)
login_manager.login_view = 'login'


# ============================================================
# BASE DE DONNÉES
# ============================================================

# 13. créer un modèle (table) pour les utilisateurs :
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)  # identifiant unique
    username = db.Column(db.String(80), unique=True, nullable=False)  # nom d’utilisateur
    password_hash = db.Column(db.String(128), nullable=False)  # mot de passe hashé
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # date de création

    # 14. fonction pour hasher le mot de passe :
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    # 15. fonction pour vérifier le mot de passe :
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


# 16. créer un modèle (table) pour stocker l’historique des images :
class ImageHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # lien avec utilisateur
    image_filename = db.Column(db.String(200), nullable=False)  # nom de l’image
    emotion = db.Column(db.String(64), nullable=False)  # émotion prédite
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)  # date et heure

    # relation avec utilisateur (un user peut avoir plusieurs images)
    user = db.relationship('User', backref=db.backref('history', lazy='dynamic'))


# 17. fonction pour charger un utilisateur depuis son ID :
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ============================================================
# CHARGEMENT DU MODÈLE D’IA
# ============================================================

# 18. définir le chemin du modèle entraîné :
# 19. vérifier si le modèle existe et le charger :
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.h5")

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
else:
    model = None
    EMOTIONS = []



def model_predict(img_path):
    if model is None:
        return "NoModel"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.astype("float") / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    preds = model.predict(img)[0]
    label = EMOTIONS[np.argmax(preds)]
    return label

# ============================================================
# ROUTES: AUTH
# ============================================================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username').strip()
        password = request.form.get('password')
        if not username or not password:
            flash("Fill all fields", "warning")
            return render_template('register.html')
        if User.query.filter_by(username=username).first():
            flash("Username already taken", "danger")
            return render_template('register.html')
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for('index'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username').strip()
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))
        flash("Invalid username or password", "danger")
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ============================================================
# ROUTES: APP FUNCTIONALITY
# ============================================================
@app.route('/')
@login_required
def index():
    items = ImageHistory.query.filter_by(user_id=current_user.id).order_by(
        ImageHistory.timestamp.desc()).limit(12).all()
    return render_template('index.html', history=items)


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        flash("No file uploaded", "danger")
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash("No selected file", "warning")
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    emotion = model_predict(filepath)

    # Save in DB
    hist = ImageHistory(user_id=current_user.id,
                        image_filename=filename,
                        emotion=emotion)
    db.session.add(hist)
    db.session.commit()

    return render_template('index.html', prediction=emotion,
                           img_path=url_for('static', filename=f'uploads/{filename}'),
                           history=current_user.history.order_by(ImageHistory.timestamp.desc()).limit(12))


@app.route('/camera')
@login_required
def camera():
    return render_template('camera.html')

def gen_frames():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("⚠️ Impossible d’ouvrir la caméra !")
        return

    print("✅ Caméra détectée, prédiction live lancée...")

    frame_count = 0
    label = "..."  # Valeur par défaut

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        frame_count += 1
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            # 👉 Faire la prédiction une fois toutes les 10 images
            if frame_count % 10 == 0:
                preds = model.predict(roi, verbose=0)[0]
                label = EMOTIONS[np.argmax(preds)]

            # Afficher le label précédent (pas besoin de recalculer chaque frame)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
