import os

def rename_files_in_folder(folder_path):
    try:
        # Ambil daftar semua file dalam folder
        files = os.listdir(folder_path)

        # Sortir file berdasarkan nama untuk memastikan urutan yang benar
        files.sort()

        # Inisialisasi penghitung untuk A_ dan Ax
        counter_a = 1
        counter_ax = 1

        for file_name in files:
            # Dapatkan jalur lengkap file
            old_file_path = os.path.join(folder_path, file_name)

            # Pastikan itu file (bukan folder)
            if os.path.isfile(old_file_path):
                # Dapatkan ekstensi file
                file_extension = os.path.splitext(file_name)[1]  # Mengambil ekstensi (misalnya, .jpg, .png)


                # Tentukan nama baru berdasarkan urutan
                if counter_a <= 5:
                    new_name = f"A_{counter_a}{file_extension}"
                    counter_a += 1
                else:
                    new_name = f"Ax{counter_ax}{file_extension}"
                    counter_ax += 1

                # Jalur lengkap file baru
                new_file_path = os.path.join(folder_path, new_name)

                # Ganti nama file
                os.rename(old_file_path, new_file_path)

        print("Proses mengganti nama file selesai.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

# Masukkan jalur folder yang ingin diubah nama filenya
folder_path = input("Masukkan path folder: ")
rename_files_in_folder(folder_path)
