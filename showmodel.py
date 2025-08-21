import os
import pickle
import pandas as pd

def load_data_for_plots(path):
    temp_folder = os.path.dirname(path)

    # Memuat data dari CSV
    csv_path = os.path.join(temp_folder, "hasil_data.csv")
    if not os.path.exists(csv_path):
        print("Data CSV tidak ditemukan!")
        return None, None

    df = pd.read_csv(csv_path)
    print("Data berhasil dimuat!")
    
    X_train = df[['L', 'a', 'b', 'luas_kerusakan']]
    y_train = df['umur_simpan']
    
    return X_train, y_train