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
from tkinter import filedialog
from tkinter import Tk 

info = ""
proggress = 0

def select_file(title):
    Tk().withdraw()
    file_path = askopenfilename(title=title, filetypes=[("Pickle files", "*.pkl")])
    return file_path

def pilih_gambar():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Pilih File Gambar",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    return file_path

def selection(image_path, hasil_folder, healthy_color_lab, threshold):
    global info, proggress
    proggress = proggress + 1
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Gambar {image_path} tidak ditemukan!")
    
    os.makedirs(os.path.dirname(hasil_folder), exist_ok=True)
    
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    distance = np.sqrt(np.sum((image_lab - healthy_color_lab) ** 2, axis=2))
    mask = (distance < threshold).astype(np.uint8) * 255
    mask_inverted = cv2.bitwise_not(mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask_inverted)
    
    temp_output_path_selection = os.path.join(hasil_folder, os.path.basename(image_path))
    if not temp_output_path_selection.lower().endswith(('.png', '.jpg', '.jpeg')):
        temp_output_path_selection += '.png'
    
    success = cv2.imwrite(temp_output_path_selection, masked_image)
    if not success:
        raise IOError(f"Gagal menyimpan gambar ke {temp_output_path_selection}")
    info = "Damaged area selected"
    mean_L, mean_a, mean_b, damaged_area = final(temp_output_path_selection)
    proggress = proggress + 1
    return mean_L, mean_a, mean_b, damaged_area

def final(img_path):
    global info, proggress
    proggress = proggress + 1
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Membaca gambar
    if img is None:
        print(f"Gambar {img_path} tidak ditemukan!")
        return None, None, None

    if img.shape[2] == 4:
        bgr_img = img[:, :, :3]  # Mengambil saluran BGR tanpa alpha
        alpha_channel = img[:, :, 3]  # Mengambil saluran alpha
        mask_alpha = alpha_channel > 0  # Membuat mask untuk area yang tidak transparan
        mask_alpha = mask_alpha.astype(np.uint8) * 255
        masked_img = cv2.bitwise_and(bgr_img, bgr_img, mask=mask_alpha)  # Menerapkan mask pada gambar
    else:
        masked_img = img  # Jika tidak ada alpha, gunakan gambar langsung

    mask_non_black = np.any(masked_img > 10, axis=2)  # Piksel yang bukan hitam (threshold bisa disesuaikan)
    
    analyzed_pixels = np.count_nonzero(mask_non_black)  # Menghitung jumlah piksel yang tidak hitam

    lab_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2LAB)  # Mengkonversi gambar ke LAB
    L, a, b = cv2.split(lab_img)  # Memisahkan saluran LAB

    L_non_black = L[mask_non_black]
    a_non_black = a[mask_non_black]
    b_non_black = b[mask_non_black]

    if L_non_black.size > 0:
        mean_L = np.mean(L_non_black) / 2.55  # Rata-rata L, distorsi skala untuk range 0-100
        mean_a = np.mean(a_non_black) - 128  # Rata-rata a, skala untuk rentang -128 hingga 127
        mean_b = np.mean(b_non_black) - 128  # Rata-rata b, skala untuk rentang -128 hingga 127
    else:
        mean_L, mean_a, mean_b = 0, 0, 0  # Jika tidak ada data, kembalikan 0

    info = "Analyzing..."
    proggress = proggress + 1
    return mean_L, mean_a, mean_b, analyzed_pixels

def resize_image(input_path, output_path, new_height=1000):
    global info, proggress
    proggress = proggress + 1
    with Image.open(input_path) as img:
        width, height = img.size
        
        new_width = int((new_height / height) * width)
        
        resized_img = img.resize((new_width, new_height))
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        resized_img.save(output_path)
        info = f"Image resized: {output_path}"

def remove_background(input_path, output_path, shift_edge_pixels):
    global info, proggress
    proggress = proggress + 1
    output_folder = os.path.dirname(output_path)
    cropped_folder = os.path.join(output_folder, 'cropped')
    resized_folder = os.path.join(output_folder, 'resized')
    os.makedirs(resized_folder, exist_ok=True)
    
    cropped_image_path = os.path.join(cropped_folder, os.path.basename(input_path))

    os.makedirs(os.path.dirname(cropped_image_path), exist_ok=True)

    resized_image_path = os.path.join(resized_folder, os.path.basename(input_path))
    
    resize_image(input_path, resized_image_path, new_height=1000)
    
    with open(resized_image_path, 'rb') as input_file:
        input_image = input_file.read() 
    
    output_image = remove(input_image)  

    img = Image.open(io.BytesIO(output_image)).convert("RGBA")
    img_np = np.array(img)  # Mengubah gambar menjadi array numpy
    
    alpha_channel = img_np[:, :, 3]
    
    _, binary_mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=shift_edge_pixels)  # Erosi untuk memperbaiki tepi
    
    img_np[:, :, 3] = eroded_mask
    
    img_result = Image.fromarray(img_np)
    
    img_result = img_result.convert("RGB")
    
    img_result.save(cropped_image_path)

    proggress = proggress + 1

    info = "Background removed"

    return cropped_image_path  # Mengembalikan path gambar hasil

def process_test_image(folder_path, hasil_folder, healthy_color_lab):
    global info, proggress
    proggress = proggress + 1
    info = "Procceeding image"
    L = None
    a = None
    b = None
    damaged_area = None
    tr = 98
    shift_edge_pixels = 5  # Nilai pergeseran tepi untuk erosi
    removed_bg_img_path = remove_background(folder_path, hasil_folder, shift_edge_pixels)  # Menghapus latar belakang
    h_L, h_a, h_b = 0, 0, 0
    proggress = proggress + 1
    if removed_bg_img_path:
        mean_L, mean_a, mean_b, damaged_area = selection(removed_bg_img_path, hasil_folder, healthy_color_lab, tr)
    
    L = mean_L
    a = mean_a
    b = mean_b

    return L, a, b, damaged_area  # Mengembalikan data yang telah diproses

def test_model(model_filename, image_path):
    global info, proggress
    proggress = 0
    proggress = proggress + 1

    healthy_color_filename = model_filename.replace("_model.pkl", "_healthy_color.npy")
    
    if not os.path.exists(healthy_color_filename):
        info = f"healthy color not found: {model_filename} !"
        return
    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)
    info = f"Model has been loaded: {model_filename}"

    healthy_color_lab = np.load(healthy_color_filename, allow_pickle=True)
    info = f"Healthy color loaded: {healthy_color_filename}"
    
    if image_path:
        folder_path = os.path.dirname(image_path)
        hasil_folder = os.path.join(folder_path, "result")
        if not os.path.exists(hasil_folder):
            os.makedirs(hasil_folder)
        L, a, b, damaged_area = process_test_image(image_path, hasil_folder, healthy_color_lab)
        print(L)
        print(a)
        print(b)
        print(damaged_area)

        info = f"Picture has been analyzed: {L}, {a}, {b}, {damaged_area}"
        proggress = proggress + 1

        input_data = pd.DataFrame({'L': [L], 'a': [a], 'b': [b], 'damaged_area': [damaged_area]})
        input_data.rename(columns={'damaged_area': 'luas_kerusakan'}, inplace=True)
        prediction = model.predict(input_data)

        info = "Analyze completed!"
    else:
        info = "File not found!"
    proggress = proggress + 1
    return prediction[0], L, a, b, damaged_area, healthy_color_lab

def get_info():
    pr = (proggress/12) * 100
    return info, pr
