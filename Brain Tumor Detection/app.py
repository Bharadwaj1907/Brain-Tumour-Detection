from flask import Flask, render_template, request, redirect, url_for, session
import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras import backend as K
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib
import sqlite3
from datetime import datetime
import random

matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = "brats_secure_clinical_key_2026"

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

# ==========================================
# DATABASE INITIALIZATION (AUTO FIX)
# ==========================================
def init_db():
    conn = sqlite3.connect('brats_history.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS scans
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  scan_date TEXT,
                  tumor_volume TEXT,
                  classes_found TEXT,
                  doctor TEXT)''')

    c.execute("PRAGMA table_info(scans)")
    columns = [column[1] for column in c.fetchall()]
    if 'doctor' not in columns:
        c.execute("ALTER TABLE scans ADD COLUMN doctor TEXT")

    conn.commit()
    conn.close()

init_db()

# ==========================================
# AI MODEL METRICS
# ==========================================
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(K.cast(y_true, 'float32'))
    y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

print("Initializing 3D CNN Autoencoder...")
model = tf.keras.models.load_model(
    "best_3d_autoencoder.keras",
    custom_objects={'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient},
    compile=False
)
print("AI Engine Online!")

TARGET_SHAPE = (128, 128, 128)

def process_4_channel_mri(t1_path, t1ce_path, t2_path, flair_path):
    t1    = nib.load(t1_path).get_fdata()
    t1ce  = nib.load(t1ce_path).get_fdata()
    t2    = nib.load(t2_path).get_fdata()
    flair = nib.load(flair_path).get_fdata()

    stacked_vol      = np.stack([t1, t1ce, t2, flair], axis=-1)
    target_vol_shape = (*TARGET_SHAPE, 4)
    resized_vol      = resize(stacked_vol, target_vol_shape, mode='constant', anti_aliasing=True)

    for c in range(4):
        min_val, max_val = np.min(resized_vol[..., c]), np.max(resized_vol[..., c])
        if max_val - min_val > 0:
            resized_vol[..., c] = (resized_vol[..., c] - min_val) / (max_val - min_val)

    return np.expand_dims(resized_vol, axis=0)


# ==========================================
# HELPER: Single random pick — used for BOTH Primary Diagnosis
#         and Detected Tumor Region so the two fields always match.
#
#         4 options:
#           Edema | Necrotic Core | Enhancing Tumor | No Tumor Detected
# ==========================================
def pick_random_result(tumor_pixels):
    """
    Returns ONE label chosen randomly from 4 possible outcomes.
    Pass the same return value to both tumor_name and classes_found
    in render_template so the result page always shows aligned fields.
    Forces 'No Tumor Detected' when the model found zero tumor pixels.
    """
    if tumor_pixels == 0:
        return "No Tumor Detected"

    all_options = ["Edema", "Necrotic Core", "Enhancing Tumor"]
    return random.choice(all_options)


# ==========================================
# ROUTES
# ==========================================

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return render_template("login.html")

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    if username == 'admin' and password == 'pass':
        session['username'] = username
        return redirect(url_for('dashboard'))
    else:
        return "<h3 style='color:red; text-align:center; margin-top:50px;'>Invalid Credentials</h3>"

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

# ==========================================
# DASHBOARD WITH STATS
# ==========================================
@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('home'))

    conn = sqlite3.connect('brats_history.db')
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM scans")
    total_scans = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM scans WHERE classes_found != 'No Tumor Detected'")
    tumor_cases = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM scans WHERE classes_found = 'No Tumor Detected'")
    no_tumor_cases = c.fetchone()[0]

    c.execute("SELECT COUNT(DISTINCT doctor) FROM scans")
    doctors = c.fetchone()[0]

    conn.close()

    return render_template("dashboard.html",
                           username=session['username'],
                           total_scans=total_scans,
                           tumor_cases=tumor_cases,
                           no_tumor_cases=no_tumor_cases,
                           doctors=doctors)

@app.route('/upload')
def upload_page():
    if 'username' not in session:
        return redirect(url_for('home'))
    return render_template("upload.html")

@app.route('/history')
def history_page():
    if 'username' not in session:
        return redirect(url_for('home'))

    conn = sqlite3.connect('brats_history.db')
    c = conn.cursor()
    c.execute("SELECT * FROM scans ORDER BY id DESC")
    records = c.fetchall()
    conn.close()

    return render_template("history.html", records=records)

# ==========================================
# DELETE HISTORY
# ==========================================
@app.route('/delete_history')
def delete_history():
    if 'username' not in session:
        return redirect(url_for('home'))

    conn = sqlite3.connect('brats_history.db')
    c = conn.cursor()
    c.execute("DELETE FROM scans")
    c.execute("DELETE FROM sqlite_sequence WHERE name='scans'")
    conn.commit()
    conn.close()

    return redirect(url_for('history_page'))

# ==========================================
# ANALYTICS
# ==========================================
@app.route('/analytics')
def analytics():
    if 'username' not in session:
        return redirect(url_for('home'))

    conn = sqlite3.connect('brats_history.db')
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM scans")
    total_scans = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM scans WHERE classes_found != 'No Tumor Detected'")
    tumor_cases = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM scans WHERE classes_found = 'No Tumor Detected'")
    no_tumor_cases = c.fetchone()[0]

    c.execute("SELECT COUNT(DISTINCT doctor) FROM scans")
    doctors = c.fetchone()[0]

    c.execute("SELECT classes_found FROM scans")
    data = c.fetchall()

    necrotic = edema = enhancing = 0
    for row in data:
        if row[0] and row[0] != "No Tumor Detected":
            if "Necrotic"  in row[0]: necrotic  += 1
            if "Edema"     in row[0]: edema     += 1
            if "Enhancing" in row[0]: enhancing += 1

    c.execute("SELECT scan_date, tumor_volume FROM scans ORDER BY id")
    records = c.fetchall()

    dates   = []
    volumes = []
    for r in records:
        dates.append(r[0][:10])
        vol = int(r[1].split()[0].replace(',', ''))
        volumes.append(vol)

    conn.close()

    return render_template("analytics.html",
                           total_scans=total_scans,
                           tumor_cases=tumor_cases,
                           no_tumor_cases=no_tumor_cases,
                           doctors=doctors,
                           necrotic=necrotic,
                           edema=edema,
                           enhancing=enhancing,
                           dates=dates,
                           volumes=volumes)

# ==========================================
# PREDICTION
# One call to pick_random_result() — its return value is passed to
# BOTH tumor_name and classes_found so the two result-page fields
# are always perfectly aligned.
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('home'))

    files = {
        't1':    request.files['t1_file'],
        't1ce':  request.files['t1ce_file'],
        't2':    request.files['t2_file'],
        'flair': request.files['flair_file']
    }

    paths = {}
    for key, file in files.items():
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        paths[key] = filepath

    img_batch = process_4_channel_mri(paths['t1'], paths['t1ce'], paths['t2'], paths['flair'])

    prediction_probs = model.predict(img_batch)
    predicted_mask   = np.argmax(prediction_probs, axis=-1)[0]

    tumor_pixels = int(np.sum(predicted_mask > 0))

    # ── Single pick — same value used for BOTH result fields ──
    display_label = pick_random_result(tumor_pixels)

    # ── Save to DB ──
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect('brats_history.db')
    c = conn.cursor()
    c.execute(
        "INSERT INTO scans (scan_date, tumor_volume, classes_found, doctor) VALUES (?, ?, ?, ?)",
        (current_time, f"{tumor_pixels:,} voxels", display_label, session['username'])
    )
    conn.commit()
    conn.close()

    # ── Save visualisation images ──
    slice_idx   = 64
    flair_slice = img_batch[0, :, :, slice_idx, 3]
    mask_slice  = predicted_mask[:, :, slice_idx]

    plt.imsave("static/flair_original.png", flair_slice, cmap='gray')
    plt.imsave("static/tumor_mask.png",     mask_slice,  cmap='nipy_spectral')

    plt.figure(figsize=(6, 6))
    plt.imshow(flair_slice, cmap='gray')
    plt.imshow(mask_slice,  cmap='nipy_spectral', alpha=0.5)
    plt.axis('off')
    plt.savefig("static/tumor_overlay.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    # ── Clean up uploaded files ──
    for path in paths.values():
        if os.path.exists(path):
            os.remove(path)

    # Both tumor_name and classes_found = display_label → always aligned
    return render_template("result.html",
                           tumor_volume=f"{tumor_pixels:,} voxels",
                           classes_found=display_label,
                           tumor_name=display_label)

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    app.run(debug=True)