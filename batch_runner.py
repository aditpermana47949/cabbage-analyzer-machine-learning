import pickle
import os
import numpy as np
import pandas as pd
import cv2
from rembg import remove
from PIL import Image
import io
import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tkinter import filedialog  # Mengimpor filedialog
from tkinter import Tk  # Mengimpor Tk

# Fungsi untuk memilih file menggunakan file dialog
def select_file(title):
    Tk().withdraw()  # Menyembunyikan root window tkinter
    file_path = askopenfilename(title=title, filetypes=[("Pickle files", "*.pkl")])
    return file_path

# Fungsi untuk memilih file gambar
def pilih_gambar():
    root = tk.Tk()
    root.withdraw()  # Sembunyikan jendela utama tkinter
    file_path = filedialog.askopenfilename(
        title="Pilih File Gambar",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    return file_path

def selection(image_path, hasil_folder, healthy_color_lab, threshold):
    image = cv2.imread(image_path)  # Membaca gambar input
    if image is None:
        raise FileNotFoundError(f"Gambar {image_path} tidak ditemukan!")
    
    # Pastikan direktori tujuan ada
    os.makedirs(os.path.dirname(hasil_folder), exist_ok=True)
    
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)  # Mengkonversi gambar ke ruang warna LAB
    distance = np.sqrt(np.sum((image_lab - healthy_color_lab) ** 2, axis=2))  # Menghitung jarak warna
    mask = (distance < threshold).astype(np.uint8) * 255  # Membuat mask untuk area kerusakan
    mask_inverted = cv2.bitwise_not(mask)  # Membalikkan mask
    masked_image = cv2.bitwise_and(image, image, mask=mask_inverted)  # Menerapkan mask pada gambar
     
    # Pastikan path memiliki ekstensi valid
    temp_output_path_selection = os.path.join(hasil_folder, os.path.basename(image_path))
    if not temp_output_path_selection.lower().endswith(('.png', '.jpg', '.jpeg')):
        temp_output_path_selection += '.png'
    
    # Menyimpan gambar hasil seleksi sementara
    success = cv2.imwrite(temp_output_path_selection, masked_image)
    if not success:
        raise IOError(f"Gagal menyimpan gambar ke {temp_output_path_selection}")
    
    # Menghitung rata-rata nilai L, a, b pada gambar yang sudah diseleksi
    mean_L, mean_a, mean_b, damaged_area = final(temp_output_path_selection)
    
    return mean_L, mean_a, mean_b, damaged_area

def final(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Membaca gambar
    if img is None:
        print(f"Gambar {img_path} tidak ditemukan!")
        return None, None, None

    # Cek apakah gambar memiliki saluran alpha (transparansi)
    if img.shape[2] == 4:
        bgr_img = img[:, :, :3]  # Mengambil saluran BGR tanpa alpha
        alpha_channel = img[:, :, 3]  # Mengambil saluran alpha
        mask_alpha = alpha_channel > 0  # Membuat mask untuk area yang tidak transparan
        mask_alpha = mask_alpha.astype(np.uint8) * 255
        masked_img = cv2.bitwise_and(bgr_img, bgr_img, mask=mask_alpha)  # Menerapkan mask pada gambar
    else:
        masked_img = img  # Jika tidak ada alpha, gunakan gambar langsung

    # Membuat mask untuk area yang tidak hitam (area selain hitam)
    mask_non_black = np.any(masked_img > 10, axis=2)  # Piksel yang bukan hitam (threshold bisa disesuaikan)
    
    # Menghitung jumlah piksel yang dianalisis (area non-hitam)
    analyzed_pixels = np.count_nonzero(mask_non_black)  # Menghitung jumlah piksel yang tidak hitam

    # Konversi gambar ke LAB untuk analisis warna
    lab_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2LAB)  # Mengkonversi gambar ke LAB
    L, a, b = cv2.split(lab_img)  # Memisahkan saluran LAB

    # Menghitung rata-rata nilai LAB pada area non-hitam (menggunakan mask)
    L_non_black = L[mask_non_black]
    a_non_black = a[mask_non_black]
    b_non_black = b[mask_non_black]

    # Jika ada area non-hitam, hitung rata-rata LAB, jika tidak set rata-rata ke 0
    if L_non_black.size > 0:
        mean_L = np.mean(L_non_black) / 2.55  # Rata-rata L, distorsi skala untuk range 0-100
        mean_a = np.mean(a_non_black) - 128  # Rata-rata a, skala untuk rentang -128 hingga 127
        mean_b = np.mean(b_non_black) - 128  # Rata-rata b, skala untuk rentang -128 hingga 127
    else:
        mean_L, mean_a, mean_b = 0, 0, 0  # Jika tidak ada data, kembalikan 0

    # Mengembalikan rata-rata LAB dan jumlah piksel yang dianalisis
    return mean_L, mean_a, mean_b, analyzed_pixels

def resize_image(input_path, output_path, new_height=1000):
    # Buka gambar
    with Image.open(input_path) as img:
        # Dapatkan ukuran asli gambar
        width, height = img.size
        
        # Hitung rasio proporsional berdasarkan height baru
        new_width = int((new_height / height) * width)
        
        # Resize gambar dengan mempertahankan proporsi
        resized_img = img.resize((new_width, new_height))
        
        # Pastikan direktori output ada
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Simpan gambar yang sudah diresize
        resized_img.save(output_path)

def remove_background(input_path, output_path, shift_edge_pixels):
    # Buat folder "resized" di dalam folder output jika belum ada
    output_folder = os.path.dirname(output_path)
    cropped_folder = os.path.join(output_folder, 'cropped')
    resized_folder = os.path.join(output_folder, 'resized')
    os.makedirs(resized_folder, exist_ok=True)
    
    # Tentukan path gambar yang sudah dicrop
    cropped_image_path = os.path.join(cropped_folder, os.path.basename(input_path))

    # Membuat directory cropped image
    os.makedirs(os.path.dirname(cropped_image_path), exist_ok=True)

    # Tentukan path gambar yang sudah diresize
    resized_image_path = os.path.join(resized_folder, os.path.basename(input_path))
    
    # Resize gambar sebelum diproses
    resize_image(input_path, resized_image_path, new_height=1000)
    
    # Membaca gambar hasil resize
    with open(resized_image_path, 'rb') as input_file:
        input_image = input_file.read()  # Membaca gambar input dalam format biner
    
    # Menghapus background menggunakan rembg
    output_image = remove(input_image)  # Menggunakan rembg untuk menghapus latar belakang
    
    # Membuka hasil gambar sebagai RGBA
    img = Image.open(io.BytesIO(output_image)).convert("RGBA")
    img_np = np.array(img)  # Mengubah gambar menjadi array numpy
    
    # Mengambil saluran alpha
    alpha_channel = img_np[:, :, 3]
    
    # Membuat mask biner dari alpha
    _, binary_mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)
    
    # Membuat kernel untuk operasi erosi
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=shift_edge_pixels)  # Erosi untuk memperbaiki tepi
    
    # Mengubah saluran alpha dengan mask yang telah tererosi
    img_np[:, :, 3] = eroded_mask
    
    # Mengubah array kembali ke gambar
    img_result = Image.fromarray(img_np)
    
    # Mengkonversi gambar ke RGB untuk disimpan
    img_result = img_result.convert("RGB")
    
    # Menyimpan gambar hasil
    img_result.save(cropped_image_path)

    return cropped_image_path  # Mengembalikan path gambar hasil

def process_test_image(folder_path, hasil_folder, healthy_color_lab):
    L = None
    a = None
    b = None
    damaged_area = None
    tr = 98
    shift_edge_pixels = 5  # Nilai pergeseran tepi untuk erosi
    removed_bg_img_path = remove_background(folder_path, hasil_folder, shift_edge_pixels)  # Menghapus latar belakang
    h_L, h_a, h_b = 0, 0, 0
    # Seleksi berdasarkan warna dan hitung rata-rata nilai LAB
    if removed_bg_img_path:
        mean_L, mean_a, mean_b, damaged_area = selection(removed_bg_img_path, hasil_folder, healthy_color_lab, tr)
    
    L = mean_L
    a = mean_a
    b = mean_b

    return L, a, b, damaged_area  # Mengembalikan data yang telah diproses

def test_model_batch():
    # Memilih file model dan healthy color menggunakan file dialog
    model_filename = select_file("Pilih file model yang ingin diuji")
    if not model_filename:
        print("Model file tidak dipilih!")
        return

    healthy_color_filename = model_filename.replace("_model.pkl", "_healthy_color.npy")
    if not os.path.exists(healthy_color_filename):
        print(f"File healthy color untuk model {model_filename} tidak ditemukan!")
        return

    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)
    print(f"Model berhasil dimuat dari {model_filename}")

    healthy_color_lab = np.load(healthy_color_filename, allow_pickle=True)
    print(f"Healthy color berhasil dimuat dari {healthy_color_filename}")

    root_folder = askdirectory(title="Pilih folder utama yang berisi gambar")
    if not root_folder:
        print("Folder tidak dipilih!")
        return

    hasil_folder = os.path.join(root_folder, "result")
    os.makedirs(hasil_folder, exist_ok=True)

    predictions = []
    
    day_folders = sorted(os.listdir(root_folder), key=lambda x: int(x.split()[1]) if x.startswith("Day ") and x.split()[1].isdigit() else float('inf'))

    for day_folder in day_folders:
        day_path = os.path.join(root_folder, day_folder)
        if not os.path.isdir(day_path):
            continue  # Skip jika bukan folder

        print(f"Memproses folder: {day_folder}")
        
        for image_name in sorted(os.listdir(day_path)):
            image_path = os.path.join(day_path, image_name)
            if not image_path.lower().endswith((".jpg", ".jpeg", ".png")):
                continue  # Skip jika bukan gambar

            L, a, b, damaged_area = process_test_image(image_path, hasil_folder, healthy_color_lab)
            
            # Buat DataFrame agar model mengenali fitur
            input_data = pd.DataFrame([[L, a, b, damaged_area]], columns=['L', 'a', 'b', 'luas_kerusakan'])
            
            # Prediksi menggunakan model
            prediction = model.predict(input_data)[0]
            
            # Simpan hasil
            predictions.append([day_folder, image_name, L, a, b, damaged_area, prediction])
            print(f"{image_name} -> Prediksi umur simpan: {prediction} hari")

    # Simpan hasil ke CSV
    df_result = pd.DataFrame(predictions, columns=["Hari", "File", "L", "a", "b", "Luas Kerusakan", "Prediksi Umur Simpan"])
    result_csv_path = os.path.join(hasil_folder, "hasil_prediksi.csv")
    # Simpan dalam format Excel
    result_excel_path = os.path.join(hasil_folder, "hasil_prediksi.xlsx")
    df_result.to_excel(result_excel_path, index=False, engine="openpyxl")
    print(f"Hasil prediksi disimpan di {result_excel_path}")

    print(f"Hasil prediksi disimpan di {result_csv_path}")

if __name__ == "__main__":
    test_model_batch()
