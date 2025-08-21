import pickle
import cv2
import numpy as np
import os
import pandas as pd
from rembg import remove
from PIL import Image
import io
import re
from tkinter import Tk
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askdirectory
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tkinter import ttk
import threading

def selection(image_path, temp_output_path_selection, healthy_color_lab, threshold):
    image = cv2.imread(image_path)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    distance = np.sqrt(np.sum((image_lab - healthy_color_lab) ** 2, axis=2))
    mask = (distance < threshold).astype(np.uint8) * 255
    mask_inverted = cv2.bitwise_not(mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask_inverted)
    cv2.imwrite(temp_output_path_selection, masked_image)
    mean_L, mean_a, mean_b, damaged_area = final(temp_output_path_selection)
    return mean_L, mean_a, mean_b, damaged_area

def color_picker(path, model_name, folder):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    mask_nonblack = np.all(image != [0, 0, 0], axis=-1)
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab_img)
    mean_L = np.mean(L[mask_nonblack]) / 2.55
    mean_a = np.mean(a[mask_nonblack]) - 128
    mean_b = np.mean(b[mask_nonblack]) - 128
    healthy_color_lab = np.array([(mean_L * 2.55), (mean_a + 128), (mean_b + 128)])
    if not os.path.exists(folder):
        os.makedirs(folder)

    healthy_color_filename = os.path.join(folder, f"{model_name}_healthy_color.npy")
    print("berhasil simpan colorlab")
    print(healthy_color_lab)
    np.save(healthy_color_filename, healthy_color_lab)
    print("color picker")
    print(f"mean_L: {mean_L}")
    print(f"mean_a: {mean_a}")
    print(f"mean_b: {mean_b}")
    return mean_L, mean_a, mean_b

def resize_image(input_path, output_path, new_height=1000):
    with Image.open(input_path) as img:
        width, height = img.size
        new_width = int((new_height / height) * width)
        resized_img = img.resize((new_width, new_height))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        resized_img.save(output_path)

def remove_background(input_path, output_path, shift_edge_pixels):
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
    img_np = np.array(img)
    
    alpha_channel = img_np[:, :, 3]
    _, binary_mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=shift_edge_pixels)
    
    img_np[:, :, 3] = eroded_mask
    img_result = Image.fromarray(img_np)
    
    img_result = img_result.convert("RGB")
    img_result.save(cropped_image_path)

    return cropped_image_path

def final(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Gambar {img_path} tidak ditemukan!")
        return None, None, None

    if img.shape[2] == 4:
        bgr_img = img[:, :, :3]
        alpha_channel = img[:, :, 3]
        mask_alpha = alpha_channel > 0
        mask_alpha = mask_alpha.astype(np.uint8) * 255
        masked_img = cv2.bitwise_and(bgr_img, bgr_img, mask=mask_alpha)
    else:
        masked_img = img

    mask_non_black = np.any(masked_img > 10, axis=2)
    analyzed_pixels = np.count_nonzero(mask_non_black)

    lab_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab_img)

    L_non_black = L[mask_non_black]
    a_non_black = a[mask_non_black]
    b_non_black = b[mask_non_black]

    if L_non_black.size > 0:
        mean_L = np.mean(L_non_black) / 2.55
        mean_a = np.mean(a_non_black) - 128
        mean_b = np.mean(b_non_black) - 128
    else:
        mean_L, mean_a, mean_b = 0, 0, 0

    return mean_L, mean_a, mean_b, analyzed_pixels

def select_input_folder():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder_path = askdirectory(title="Pilih folder yang berisi gambar")
    root.attributes('-topmost', False)
    root.destroy()
    return folder_path

def select_output_folder():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder_path = askdirectory(title="Pilih folder untuk menyimpan output")
    root.attributes('-topmost', False)
    root.destroy()
    return folder_path

def load_previous_lab(temp_file_data, current_day, sample_id):
    try:
        if not os.path.exists(temp_file_data):
            return None, None, None  # Jika file tidak ada, return None
        
        data = pd.read_csv(temp_file_data)
        if data.empty:
            return None, None, None  # Jika file kosong, return None
        
        if current_day == 0:
            return None, None, None
        
        # Pastikan kolom yang dibutuhkan ada
        required_columns = {'sample_id', 'L', 'a', 'b', 'umur_simpan'}
        if not required_columns.issubset(data.columns):
            print("Kolom yang dibutuhkan tidak ditemukan dalam data.")
            return None, None, None

        # Ambil hanya data dengan sample_id yang cocok
        sample_data = data[data['sample_id'] == sample_id]

        # Ambil data dari hari sebelum current_day
        previous_sample = sample_data[sample_data['umur_simpan'] < current_day].sort_values(by='umur_simpan', ascending=False)

        if not previous_sample.empty:
            latest_entry = previous_sample.iloc[0]  # Ambil entri terbaru dari hari sebelumnya
            print(f"Loaded previous lab data for sample {sample_id} from day {latest_entry['umur_simpan']}")
            return latest_entry['L'], latest_entry['a'], latest_entry['b']
        
        return None, None, None  # Jika tidak ada data sebelumnya

    except Exception as e:
        print(f"Error membaca file {temp_file_data}: {e}")
        return None, None, None


def save_current_lab(sample_id, mean_L, mean_a, mean_b, hari, temp_file_data):
    temp_data = pd.DataFrame({'sample_id': [sample_id], 'L': [mean_L], 'a': [mean_a], 'b': [mean_b], 'umur_simpan': [hari]})
    
    # Jika file sudah ada, tambahkan data tanpa menimpa (mode append)
    if os.path.exists(temp_file_data):
        temp_data.to_csv(temp_file_data, mode='a', index=False, header=False)
    else:
        temp_data.to_csv(temp_file_data, index=False)
    
    print(f"Saved current lab data for sample {sample_id} on day {hari}")

def process_images_in_folder(folder_path, temp_folder, model_name, folder):
    # Data untuk menyimpan hasil

    data = {'L': [], 'a': [], 'b': [], 'luas_kerusakan': [], 'umur_simpan': []}


    data_delta_E = {'delta_E': [], 'luas_kerusakan': [], 'umur_simpan': []}

    temp_file_data = os.path.join(temp_folder, "temp_data.csv")


    
    healthy_color_lab = []
    tr = 95

    day_folders = [f for f in os.listdir(folder_path) if f.startswith('Day')]
    day_folders.sort(key=lambda x: int(x.split()[1]))

    total_tasks = sum(
        len([f for f in os.listdir(os.path.join(folder_path, folder)) if f.endswith('.JPG')])
        for folder in day_folders
    )

    # Membuat window untuk progress bar
    root = tk.Tk()
    root.title("Processing Images")
    root.geometry("400x150")
    root.attributes("-topmost", True)  # Pop-up di depan
    root.resizable(False, False)  # Mencegah maximize dan resize
    root.update()  # Memastikan perubahan langsung diterapkan
    root.focus_force()  # Fokus pada window pop-up

    # Fungsi untuk mencegah window ditutup
    def disable_event():
        pass

    # Mengatur agar tombol close (X) tidak berfungsi
    root.protocol("WM_DELETE_WINDOW", disable_event)

    label = ttk.Label(root, text="Processing images, please wait...")
    label.pack(pady=10)

    progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
    progress.pack(pady=20)
    progress["maximum"] = total_tasks

    def run_processing():
        task_count = 0

        # Proses gambar
        for day_folder in day_folders:
            day_path = os.path.join(folder_path, day_folder)
            day_output_folder = os.path.join(temp_folder, day_folder)
            os.makedirs(day_output_folder, exist_ok=True)

            hari = int(day_folder.split()[1])
            first_file = next((f for f in os.listdir(day_path) if f.endswith('.JPG')), None)

            if first_file:
                if first_file.startswith('A'):
                    file_pattern = re.compile(r'Ax(\d+)\.JPG$')
                elif first_file.startswith('B'):
                    file_pattern = re.compile(r'Bx(\d+)\.JPG$')
                else:
                    continue
            else:
                continue

            files = [f for f in os.listdir(day_path) if file_pattern.search(f)]
            files.sort(key=lambda f: int(file_pattern.search(f).group(1)))

            for filename in files:
                task_count += 1
                progress["value"] = task_count
                root.after(0, root.update_idletasks)  # Memperbarui progress bar di thread utama

                match = file_pattern.search(filename)
                if match:
                    sampel = int(match.group(1))
                    L_prev, a_prev, b_prev = load_previous_lab(temp_file_data, hari, sampel)
                    img_path = os.path.join(day_path, filename)

                    day_selection_folder = os.path.join(day_output_folder, 'selection')
                    os.makedirs(day_selection_folder, exist_ok=True)

                    temp_output_path = os.path.join(day_output_folder, f"temp_{filename}")
                    temp_output_path_selection = os.path.join(day_selection_folder, f"temp_s_{filename}")

                    try:
                        shift_edge_pixels = 5
                        removed_bg_img_path = remove_background(img_path, temp_output_path, shift_edge_pixels)
                        h_L, h_a, h_b = 0, 0, 0
                        if hari == 0 and sampel == 1:
                            for_color_picker = removed_bg_img_path
                            h_L, h_a, h_b = color_picker(for_color_picker, model_name, folder)
                            healthy_color_lab = np.array([(h_L * 2.55), (h_a + 128), (h_b + 128)])
                            mean_L, mean_a, mean_b, damaged_area = selection(removed_bg_img_path, temp_output_path_selection, healthy_color_lab, tr)

                            if mean_L is not None:
                                data['L'].append(mean_L)
                                data['a'].append(mean_a)
                                data['b'].append(mean_b)
                                data['luas_kerusakan'].append(damaged_area)
                                data['umur_simpan'].append(hari)
                                save_current_lab(sampel, mean_L, mean_a, mean_b, hari, temp_file_data)
                            continue

                        if healthy_color_lab is not None:
                            mean_L, mean_a, mean_b, damaged_area = selection(removed_bg_img_path, temp_output_path_selection, healthy_color_lab, tr)

                            if mean_L is not None:

                                if hari != 0:
                                    delta_E = np.sqrt((mean_L - L_prev)**2 + (mean_a - a_prev)**2 + (mean_b - b_prev)**2)
                                    data_delta_E['delta_E'].append(delta_E)
                                    data_delta_E['luas_kerusakan'].append(damaged_area)
                                    data_delta_E['umur_simpan'].append(hari)

                                data['L'].append(mean_L)
                                data['a'].append(mean_a)
                                data['b'].append(mean_b)
                                data['luas_kerusakan'].append(damaged_area)
                                data['umur_simpan'].append(hari)
                                save_current_lab(sampel, mean_L, mean_a, mean_b, hari, temp_file_data)

                    except Exception as e:
                        print(f"Error saat memproses gambar {filename}: {e}")

        # Meminta thread utama untuk menutup jendela setelah selesai
        root.after(0, root.destroy)

    # Jalankan pemrosesan di thread terpisah
    threading.Thread(target=run_processing, daemon=True).start()

    # Menjalankan mainloop untuk menjaga pop-up tetap responsif
    root.mainloop()

    return data, data_delta_E, healthy_color_lab

if __name__ == "__main__":
    healthy_color = None

    input_folder = select_input_folder()
    temp_folder = select_output_folder()

    model_name = input("Masukkan nama model: ")

    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    if input_folder and temp_folder:
        folder = temp_folder
        dataset, data_delta_E, healthy_color_lab = process_images_in_folder(input_folder, temp_folder, model_name, folder)
        print("Dataset berhasil dibuat:")
        print(dataset)

        df = pd.DataFrame(dataset)

        X = df[['L', 'a', 'b', 'luas_kerusakan']]
        y = df['umur_simpan']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        df2 = pd.DataFrame(data_delta_E)

        X2 = df2[['delta_E', 'luas_kerusakan']]
        y2 = df2['umur_simpan']

        X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

        model2 = LinearRegression()
        model2.fit(X_train_2, y_train_2)

        df3 = pd.DataFrame(dataset)

        X3 = df3[['a', 'b', 'luas_kerusakan']]
        y3 = df3['umur_simpan']

        X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X3, y3, test_size=0.2, random_state=42)

        model3 = LinearRegression()
        model3.fit(X_train_3, y_train_3)

        plt.figure(figsize=(16, 8))

        plt.subplot(2, 2, 1)
        plt.scatter(y_train, X_train['L'], color='blue', label='L')
        plt.xlabel('Umur Simpan (Hari)')
        plt.ylabel('Nilai L')
        plt.title('Nilai L terhadap Umur Simpan')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.scatter(y_train, X_train['a'], color='green', label='a')
        plt.xlabel('Umur Simpan (Hari)')
        plt.ylabel('Nilai a')
        plt.title('Nilai a terhadap Umur Simpan')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.scatter(y_train, X_train['b'], color='red', label='b')
        plt.xlabel('Umur Simpan (Hari)')
        plt.ylabel('Nilai b')
        plt.title('Nilai b terhadap Umur Simpan')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.scatter(y_train, X_train['luas_kerusakan'], color='purple', label='luas_kerusakan')
        plt.xlabel('Umur Simpan (Hari)')
        plt.ylabel('Luas Kerusakan')
        plt.title('Luas Kerusakan terhadap Umur Simpan')
        plt.grid(True)

        plt.tight_layout()  # Pastikan dipanggil sebelum plt.show()
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.scatter(y_train_2, X_train_2['delta_E'], color='orange', label='delta_E')
        plt.xlabel('Umur Simpan (Hari)')
        plt.ylabel('Delta E')
        plt.title('Delta E terhadap Umur Simpan')
        plt.grid(True)

        plt.show()


        print("DataFrame:")
        print(df)

        output_csv = os.path.join(temp_folder, "hasil_data.csv")
        df.to_csv(output_csv, index=False)
        print(f"Data berhasil disimpan ke {output_csv}")

        if not os.path.exists(folder):
            os.makedirs(folder)

        model_filename = os.path.join(folder, f"{model_name}_model_L.pkl")
        model_filename_2 = os.path.join(folder, f"{model_name}_model_delta_E.pkl")
        model_filename_3 = os.path.join(folder, f"{model_name}_model.pkl")

        with open(model_filename, 'wb') as model_file:
            pickle.dump(model, model_file)
        print(f"Model berhasil disimpan ke {model_filename}")

        with open(model_filename_2, 'wb') as model_file_2:
            pickle.dump(model2, model_file_2)
        print(f"Model berhasil disimpan ke {model_filename_2}")

        with open(model_filename_3, 'wb') as model_file_3:
            pickle.dump(model3, model_file_3)
        print(f"Model berhasil disimpan ke {model_filename_3}")