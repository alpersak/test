from flask import Flask, request, jsonify, render_template
import os
import cv2
from roboflow import Roboflow
import requests
import numpy as np
import pytesseract

# Tesseract executable path (Windows için gerekli)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Roboflow API kurulumu
rf = Roboflow(api_key="0luc9a0CTwp4V061D7gV")
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
version = project.version(4)
model = version.model

# Hasar tespiti için başka bir Roboflow modeli (örnek)
damage_project = rf.workspace("alp-6jxoz").project("car_damage-o1xrg")
damage_version = damage_project.version(2)
damage_model = damage_version.model

# Flask uygulamasının oluşturulması
app = Flask(__name__)

# Yükleme klasörü ayarla
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ana sayfa için rota
@app.route('/')
def index():
    return render_template('index.html')

# Plaka ve hasar tespiti için rota
@app.route('/detect', methods=['POST'])
def detect_plate():
    if 'images' not in request.files:
        return render_template('index.html', error="Lütfen bir veya daha fazla fotoğraf yükleyin.")

    uploaded_files = request.files.getlist('images')
    if not uploaded_files or uploaded_files[0].filename == '':
        return render_template('index.html', error="Lütfen bir veya daha fazla fotoğraf yükleyin.")

    results = []

    for image_file in uploaded_files:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(image_path)

        # Roboflow modelini kullanarak plakayı tespit et
        result = model.predict(image_path, confidence=40, overlap=30).json()

        # Tespit edilen plakaların OCR ile okunması
        predictions = result.get("predictions", [])
        plates = []

        plate_image = cv2.imread(image_path)

        for idx, prediction in enumerate(predictions):
            # Plakanın bulunduğu alanın koordinatları
            x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']

            # Koordinatları kullanarak plakayı kırp
            x1, y1, x2, y2 = int(x - width / 2), int(y - height / 2), int(x + width / 2), int(y + height / 2)
            cropped_plate = plate_image[y1:y2, x1:x2]

            # OCR kullanarak plakayı tanı
            plate_text = pytesseract.image_to_string(cropped_plate, config='--psm 8')
            plates.append(f"Plaka {idx + 1}: {plate_text.strip()}")

            # Plaka alanını işaretle ve numaralandır
            cv2.rectangle(plate_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(plate_image, f"Plaka {idx + 1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # İşaretlenmiş plaka görüntüsünü kaydet
        plate_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'plate_' + image_file.filename)
        cv2.imwrite(plate_image_path, plate_image)

        # Hasar tespiti yap
        damage_result = damage_model.predict(image_path, confidence=40, overlap=30).json()
        damage_predictions = damage_result.get("predictions", [])
        damages = []

        damage_image = cv2.imread(image_path)

        for idx, damage in enumerate(damage_predictions):
            # Hasarın bulunduğu alanın koordinatları
            x, y, width, height = damage['x'], damage['y'], damage['width'], damage['height']
            damage_class = damage['class']

            # Hasar alanını işaretle ve numaralandır
            x1, y1, x2, y2 = int(x - width / 2), int(y - height / 2), int(x + width / 2), int(y + height / 2)

            # Farklı hasar sınıflarını farklı renklerde işaretle
            if damage_class == 'Dent':
                color = (0, 0, 255)  # Kırmızı
            elif damage_class == 'Scratch':
                color = (255, 0, 0)  # Mavi
            elif damage_class == 'Crack':
                color = (0, 255, 255)  # Sarı
            else:
                color = (0, 255, 0)  # Yeşil (varsayılan)

            cv2.rectangle(damage_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(damage_image, f"Hasar {idx + 1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            damages.append(f"Hasar {idx + 1}: {damage_class}")

        # İşaretlenmiş hasar görüntüsünü kaydet
        damage_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'damage_' + image_file.filename)
        cv2.imwrite(damage_image_path, damage_image)

        # Sonuçları listeye ekle
        results.append({
            'plate_image_file': 'plate_' + image_file.filename,
            'damage_image_file': 'damage_' + image_file.filename,
            'plates': plates,
            'damages': damages
        })

    return render_template('results.html', results=results)

# Ana uygulama çalıştırma
if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, host="0.0.0.0", port=5000)