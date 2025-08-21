import pickle
import cv2
import numpy as np
import os
import pandas as pd
from rembg import remove
from PIL import Image
import io
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

count = 0
info = ""
folder_count = 0

def selection(image_path, temp_output_path_selection, healthy_color_lab, threshold):
    global info
    image = cv2.imread(image_path)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    distance = np.sqrt(np.sum((image_lab - healthy_color_lab) ** 2, axis=2))
    mask = (distance < threshold).astype(np.uint8) * 255
    mask_inverted = cv2.bitwise_not(mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask_inverted)
    cv2.imwrite(temp_output_path_selection, masked_image)
    mean_L, mean_a, mean_b, damaged_area = final(temp_output_path_selection)

    info = "Selected damage area image: " + str(count)
    return mean_L, mean_a, mean_b, damaged_area

def color_picker(path, model_name, folder):
    global info
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
    np.save(healthy_color_filename, healthy_color_lab)
    info = "Healthy color has been saved"
    return mean_L, mean_a, mean_b

def resize_image(input_path, output_path, new_height=1000):
    global info
    with Image.open(input_path) as img:
        width, height = img.size
        new_width = int((new_height / height) * width)
        resized_img = img.resize((new_width, new_height))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        resized_img.save(output_path)

    info = "Resized image: " + str(count)

def remove_background(input_path, output_path, shift_edge_pixels):
    global info
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

    info = "Removed background image: " + str(count)

    return cropped_image_path

def final(img_path):
    global info
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None, None, None
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
    
    info = "L, a, b, damaged area has been saved for image: " + str(count)
    return mean_L, mean_a, mean_b, analyzed_pixels

def load_previous_lab(temp_file_data, current_day, sample_id):
    try:
        if not os.path.exists(temp_file_data):
            return None, None, None
        data = pd.read_csv(temp_file_data)
        if data.empty:
            return None, None, None
        if current_day == 0:
            return None, None, None

        sample_data = data[data['sample_id'] == sample_id]
        previous_sample = sample_data[sample_data['umur_simpan'] < current_day].sort_values(by='umur_simpan', ascending=False)
        if not previous_sample.empty:
            latest_entry = previous_sample.iloc[0]
            return latest_entry['L'], latest_entry['a'], latest_entry['b']
        return None, None, None
    except Exception:
        return None, None, None

def save_current_lab(sample_id, mean_L, mean_a, mean_b, hari, temp_file_data):
    temp_data = pd.DataFrame({'sample_id': [sample_id], 'L': [mean_L], 'a': [mean_a], 'b': [mean_b], 'umur_simpan': [hari]})
    if os.path.exists(temp_file_data):
        temp_data.to_csv(temp_file_data, mode='a', index=False, header=False)
    else:
        temp_data.to_csv(temp_file_data, index=False)

def process_images_in_folder(folder_path, temp_folder, model_name, folder):
    global count, info, folder_count

    count = 0
    info = ""
    folder_count = 0

    data = {'L': [], 'a': [], 'b': [], 'luas_kerusakan': [], 'umur_simpan': []}
    temp_file_data = os.path.join(temp_folder, "temp_data.csv")
    
    healthy_color_lab = []
    tr = 95

    day_folders = [f for f in os.listdir(folder_path) if f.startswith('Day')]
    day_folders.sort(key=lambda x: int(x.split()[1]))

    for day_folder in day_folders:
        folder_count = folder_count + 1
        day_path = os.path.join(folder_path, day_folder)
        day_output_folder = os.path.join(temp_folder, day_folder)
        os.makedirs(day_output_folder, exist_ok=True)

        hari = int(day_folder.split()[1])
        first_file = next((f for f in os.listdir(day_path) if f.endswith('.JPG')), None)
        if not first_file:
            continue

        if first_file.startswith('A'):
            file_pattern = re.compile(r'Ax(\d+)\.JPG$')
        elif first_file.startswith('B'):
            file_pattern = re.compile(r'Bx(\d+)\.JPG$')
        else:
            continue

        files = [f for f in os.listdir(day_path) if file_pattern.search(f)]
        files.sort(key=lambda f: int(file_pattern.search(f).group(1)))

        for filename in files:
            match = file_pattern.search(filename)
            if not match:
                continue

            sampel = int(match.group(1))
            img_path = os.path.join(day_path, filename)

            day_selection_folder = os.path.join(day_output_folder, 'selection')
            os.makedirs(day_selection_folder, exist_ok=True)

            temp_output_path = os.path.join(day_output_folder, f"temp_{filename}")
            temp_output_path_selection = os.path.join(day_selection_folder, f"temp_s_{filename}")

            try:
                shift_edge_pixels = 5
                count = count + 1
                removed_bg_img_path = remove_background(img_path, temp_output_path, shift_edge_pixels)
                if hari == 0 and sampel == 1:
                    for_color_picker = removed_bg_img_path
                    h_L, h_a, h_b = color_picker(for_color_picker, model_name, folder)
                    healthy_color_lab = np.array([(h_L * 2.55), (h_a + 128), (h_b + 128)])
                
                if healthy_color_lab is not None:
                    mean_L, mean_a, mean_b, damaged_area = selection(removed_bg_img_path, temp_output_path_selection, healthy_color_lab, tr)
                    if mean_L is not None:
                        data['L'].append(mean_L)
                        data['a'].append(mean_a)
                        data['b'].append(mean_b)
                        data['luas_kerusakan'].append(damaged_area)
                        data['umur_simpan'].append(hari)
                        save_current_lab(sampel, mean_L, mean_a, mean_b, hari, temp_file_data)
                info = "Completed analyze image: " + str(count)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
            
    df = pd.DataFrame(data)

    # Fitur dan target
    X = df[['L', 'a', 'b', 'luas_kerusakan']]
    y = df['umur_simpan']

    # Melatih model dengan 100% data
    model = LinearRegression()
    model.fit(X, y)

    # Menyimpan model
    if not os.path.exists(folder):
        os.makedirs(folder)

    model_filename = os.path.join(folder, f"{model_name}_model.pkl")

    try:
        with open(model_filename, "wb") as modelfile:
            pickle.dump(model, modelfile)
            print(f"Model berhasil disimpan ke {model_filename}")
    except Exception as e:
        print(f"Gagal menyimpan model: {e}")

    # Menyimpan data ke CSV
    output_csv = os.path.join(folder, "hasil_data.csv")
    df.to_csv(output_csv, index=False)
    print(f"Data berhasil disimpan ke {output_csv}")

    return data, healthy_color_lab, model_filename

def look_for_count():
    if folder_count != 0:
        average_item = count/folder_count
    else:
        average_item = 0
    return count, info, folder_count, average_item