from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import cv2
import numpy as np
import base64
import os
import uuid
import sys
import shutil

# --- KONFIGURASI ---
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")

if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)
    print(f"[INFO] Folder dataset utama dibuat di: {DATASET_PATH}")

# --- HELPER FUNCTIONS ---
def base64_to_cv2(base64_string):
    try:
        if "," in base64_string:
            encoded_data = base64_string.split(',')[1]
        else:
            encoded_data = base64_string
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: raise ValueError("Gagal decode gambar.")
        return img
    except Exception as e:
        print(f"[ERROR DECODE] {e}")
        return None

def clear_deepface_cache():
    """Hapus file pkl di root dataset agar DeepFace reload data baru"""
    print("[CACHE] Membersihkan cache lama...")
    for pkl_file in ["representations_facenet.pkl", "representations_facenet512.pkl", 
                     "representations_vgg_face.pkl", "representations_arcface.pkl"]:
        pkl_path = os.path.join(DATASET_PATH, pkl_file)
        if os.path.exists(pkl_path):
            try: os.remove(pkl_path)
            except Exception as e: print(f"[CACHE ERROR] {e}")

# --- 1. ENDPOINT STATISTIK (Hitung Folder NIM) ---
@app.route('/stats', methods=['GET'])
def get_stats():
    """
    Menghitung total mahasiswa berdasarkan jumlah FOLDER NIM di dalam dataset.
    """
    try:
        if not os.path.exists(DATASET_PATH):
            return jsonify({"status": "success", "total_users": 0})
        
        # Hitung hanya direktori (folder)
        folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]
        
        return jsonify({"status": "success", "total_users": len(folders)})
    except Exception as e:
        return jsonify({"status": "error", "total_users": 0})

# --- 2. ENDPOINT REGISTER (Buat Folder NIM) ---
@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.json
        nim = data.get('nim')
        name = data.get('name')
        images_list = data.get('images', [])
        single_image = data.get('image')
        
        if single_image: images_list.append(single_image)

        if not nim or not name or len(images_list) == 0:
            return jsonify({"status": "error", "message": "Data tidak lengkap"}), 400

        # 1. Buat Path Folder Khusus NIM: dataset/NIM
        user_folder = os.path.join(DATASET_PATH, nim)
        
        # Jika folder belum ada, buat baru. 
        # Jika sudah ada, kita asumsikan user ingin menimpa/menambah (optional: bisa di-block kalau mau strict)
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
            print(f"[REGISTER] Folder baru dibuat: {user_folder}")

        # Bersihkan nama untuk filename
        safe_name = name.replace(" ", "_").replace("/", "").replace("\\", "")
        
        saved_count = 0
        for idx, img_b64 in enumerate(images_list):
            img = base64_to_cv2(img_b64)
            if img is None: continue
            
            # Simpan file DI DALAM folder NIM
            # Nama file: NIM_Nama_Index_UUID.jpg
            filename = f"{nim}_{safe_name}_{idx}_{uuid.uuid4().hex[:6]}.jpg"
            filepath = os.path.join(user_folder, filename)
            
            cv2.imwrite(filepath, img)
            saved_count += 1
            print(f"[SAVED] {filepath}")

        if saved_count > 0:
            clear_deepface_cache()
            return jsonify({"status": "success", "message": f"Berhasil registrasi {name} (NIM: {nim})."})
        else:
            return jsonify({"status": "error", "message": "Gagal menyimpan gambar."}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --- 3. ENDPOINT UPLOAD DATASET (Cek Folder NIM) ---
@app.route('/upload-dataset', methods=['POST'])
def upload_dataset():
    """
    Upload foto tambahan ke folder NIM yang sudah ada.
    """
    try:
        data = request.json
        nim = data.get('nim')
        images_list = data.get('images') 
        
        if not nim or not images_list or not isinstance(images_list, list):
            return jsonify({"status": "error", "message": "Format data salah."}), 400

        # 1. Cek apakah folder NIM ada
        user_folder = os.path.join(DATASET_PATH, nim)
        
        if not os.path.exists(user_folder) or not os.path.isdir(user_folder):
            return jsonify({
                "status": "error", 
                "message": f"NIM {nim} BELUM TERDAFTAR! Folder tidak ditemukan. Harap Register dulu."
            }), 404

        # 2. Cari Nama Mahasiswa dari file yang sudah ada di folder itu (agar konsisten)
        existing_files = os.listdir(user_folder)
        registered_name = "Unknown"
        
        if len(existing_files) > 0:
            # Ambil file pertama untuk contoh: NIM_Nama_...
            first_file = existing_files[0]
            parts = first_file.split('_')
            if len(parts) >= 2:
                registered_name = parts[1] # Ambil nama dari nama file
        else:
            # Folder ada tapi kosong (aneh, tapi bisa terjadi jika manual delete)
            registered_name = "User"

        print(f"[UPLOAD] Menambahkan {len(images_list)} foto ke folder {nim}")
        
        success_count = 0
        for idx, img_b64 in enumerate(images_list):
            img = base64_to_cv2(img_b64)
            if img is None: continue
            
            # Simpan file baru di folder NIM
            filename = f"{nim}_{registered_name}_Bulk_{uuid.uuid4().hex[:6]}.jpg"
            filepath = os.path.join(user_folder, filename)
            cv2.imwrite(filepath, img)
            success_count += 1
        
        if success_count > 0:
            clear_deepface_cache()
            
        return jsonify({
            "status": "success", 
            "message": f"Berhasil upload {success_count} foto ke folder {nim}."
        })

    except Exception as e:
        print(f"[ERROR UPLOAD] {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# --- 4. ENDPOINT VERIFY (Absensi) ---
@app.route('/verify', methods=['POST'])
def verify():
    try:
        data = request.json
        image_b64 = data.get('image')
        model_name = data.get('model', 'Facenet') 
        if model_name.lower() == 'facenet512': model_name = 'Facenet512'
        elif model_name.lower() == 'facenet': model_name = 'Facenet' 

        if not image_b64: return jsonify({"status": "error", "message": "No image"}), 400
        
        # Cek apakah dataset kosong (tidak ada folder)
        if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
             return jsonify({"status": "failed", "message": "Database kosong."}), 200

        temp_filename = os.path.join(BASE_DIR, "temp_verify.jpg")
        img = base64_to_cv2(image_b64)
        if img is None: return jsonify({"status": "error", "message": "Gagal proses gambar"}), 400
        cv2.imwrite(temp_filename, img)

        try:
            # DeepFace recursive search automatically checks subfolders
            dfs = DeepFace.find(
                img_path=temp_filename, 
                db_path=DATASET_PATH, 
                model_name=model_name, 
                distance_metric='cosine', 
                enforce_detection=False, 
                silent=True
            )
        except:
            if os.path.exists(temp_filename): os.remove(temp_filename)
            return jsonify({"status": "failed", "message": "Wajah tidak jelas"}), 200
        
        if os.path.exists(temp_filename): os.remove(temp_filename)

        if len(dfs) > 0 and not dfs[0].empty:
            match = dfs[0].iloc[0] 
            identity_path = match['identity']
            
            # --- PARSING IDENTITAS DARI PATH FOLDER ---
            # Path bisa berupa: dataset/NIM/filename.jpg
            # Kita ambil nama foldernya sebagai NIM
            
            folder_name = os.path.basename(os.path.dirname(identity_path))
            file_name = os.path.basename(identity_path)
            
            # NIM adalah nama foldernya
            nim = folder_name
            
            # Nama orang diambil dari nama file (NIM_Nama_...)
            parts = file_name.split('_')
            name = "Unknown"
            if len(parts) >= 2:
                name = " ".join(parts[1:-1]) # Menggabungkan jika nama ada spasi (ex: Jonathan_Christian)
                # Kasus khusus file pendek: 123_Budi_0_uuid.jpg -> Budi
            
            # Ambil distance
            distance = match.get('distance', match.get('cosine', 0.5))
            confidence = max(0, (1 - (float(distance) / 0.5)) * 100)
            
            return jsonify({
                "status": "success", 
                "data": { 
                    "nim": nim, 
                    "name": name, 
                    "distance": round(float(distance), 4), 
                    "confidence": f"{confidence:.2f}%", 
                    "model": model_name 
                }
            })
        else:
            return jsonify({"status": "failed", "message": "Wajah tidak dikenali"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)