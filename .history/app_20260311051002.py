from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# 1. LOAD MODEL
# Pastikan nama file sesuai dengan hasil training terbaru kamu
model = load_model("model3.h5")

# 2. DAFTAR KELAS (SESUAIKAN URUTAN ABJAD FOLDER DATASET)
# Urutan ini biasanya: glioma, meningioma, no_tumor, pituitary
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# 3. INFORMASI PENYAKIT
tumor_info = {
    "glioma": {
        "desc": "Glioma adalah jenis tumor yang tumbuh di otak dan sumsum tulang belakang.",
        "note": "Perlu pemeriksaan MRI lebih lanjut dan konsultasi dengan dokter spesialis saraf."
    },
    "meningioma": {
        "desc": "Meningioma adalah tumor yang muncul dari meninges — selaput yang mengelilingi otak.",
        "note": "Sebagian besar bersifat jinak, namun tekanan pada otak perlu dievaluasi medis."
    },
    "pituitary": {
        "desc": "Tumor pituitari adalah pertumbuhan abnormal pada kelenjar di dasar otak.",
        "note": "Dapat memengaruhi hormon tubuh. Disarankan cek ke dokter endokrinologi."
    },
    "notumor": {
        "desc": "Hasil analisis menunjukkan tidak terdeteksi adanya tumor pada citra ini.",
        "note": "Tetap lakukan pemeriksaan rutin dan konsultasikan hasil ini ke ahli radiologi."
    }
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'image' not in request.files:
            return render_template("index.html", error="Tidak ada file yang diunggah")
        
        file = request.files["image"]
        if file.filename == '':
            return render_template("index.html", error="Nama file kosong")

        try:
            # --- PROSES GAMBAR ---
            # Resize ke 160x160 dan pastikan format RGB
            img_pil = Image.open(file).convert('RGB').resize((224, 224))
            img_array = np.array(img_pil)
            
            # Tambah dimensi batch (1, 160, 160, 3)
            img_dims = np.expand_dims(img_array, axis=0)
            
            # Preprocessing khusus MobileNetV2 (Sangat penting agar akurat!)
            img_ready = preprocess_input(img_dims)

            # --- PREDIKSI ---
            preds = model.predict(img_ready)
            result_index = np.argmax(preds)
            result = classes[result_index]
            
            # Ambil skor keyakinan (confidence) dalam persen
            confidence = float(np.max(preds) * 100)

            # --- AMBIL INFO ---
            # Menggunakan .get untuk menghindari KeyError jika nama kelas tidak cocok
            info = tumor_info.get(result, {
                "desc": "Kategori tidak dikenal.",
                "note": "Harap periksa label kelas pada kode."
            })

            return render_template("index.html", 
                                   result=result, 
                                   confidence=f"{confidence:.2f}%",
                                   desc=info["desc"], 
                                   note=info["note"])
        

        except Exception as e:
            return render_template("index.html", error=f"Terjadi kesalahan: {str(e)}")

    return render_template("index.html")

if __name__ == "__main__":
    # Menjalankan aplikasi
    app.run(debug=True, port=5000)