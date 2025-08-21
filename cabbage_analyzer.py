import dearpygui.dearpygui as dpg
import training as tr
import testing as ts
import parameter_model as pm
import showmodel as sm
import os
import threading
import pandas as pd
from tkinter import filedialog, Tk
import time

training_input_folder = ""
training_output_folder = ""
current_model_name = ""
training_dataset = None
healthy_lab_color = None
testing_model_path = ""
testing_image_path = None
progress = 0
total_images = 0
count = 0
X_train = None
y_train = None

def select_input_folder(sender, app_data):
    global training_input_folder
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory()
    if folder:
        training_input_folder = folder
        short_path = shorten_path(folder)
        dpg.configure_item("input_folder_path", show=True)
        dpg.set_value("input_folder_path", short_path)
        validate_analysis_parameters()
    root.destroy()

def select_output_folder(sender, app_data):
    global training_output_folder
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory()
    if folder:
        training_output_folder = folder
        dpg.configure_item("output_folder_path", show=True)
        dpg.set_value("output_folder_path", shorten_path(training_output_folder))
        validate_analysis_parameters()
    root.destroy()

def validate_analysis_parameters():
    global current_model_name
    if training_input_folder and training_output_folder and current_model_name:
        dpg.configure_item("start_analysis_button", show=True)
        dpg.enable_item("start_analysis_button")
    else:
        dpg.configure_item("start_analysis_button", show=False)
        dpg.disable_item("start_analysis_button")

def handle_model_name_input(sender, app_data):
    global current_model_name
    current_model_name = app_data
    validate_analysis_parameters()

def count_images_in_folder(folder_path):
    total_images = 0
    for day_folder in os.listdir(folder_path):
        day_path = os.path.join(folder_path, day_folder)
        if os.path.isdir(day_path) and day_folder.startswith("Day"):
            total_images += len([f for f in os.listdir(day_path) if f.endswith(".JPG")])
    return total_images


def update_progress_status(total):
    while True:
        count, info, folder_count, average_item = tr.look_for_count()
        percentage = (count / total)
        percentage_string = str(round(percentage * 100)) + " %"
        progress_text = percentage_string + " - " + info + " - Folder: " + str(folder_count) + " - Average per folder: " + str(average_item)
        dpg.set_value("progress_bar", percentage)
        
        dpg.set_value("progress_text", progress_text)

        if percentage >= 100:
            dpg.set_value("progress_text", "Complete")
            break
        time.sleep(0.3)

def create_scatter_plot(x, y, label, tag):
    with dpg.plot(label=label, height=300, width=400, tag=tag):
        dpg.add_plot_axis(dpg.mvXAxis, label="Umur Simpan (Hari)")
        yaxis = dpg.add_plot_axis(dpg.mvYAxis, label=label)
        dpg.add_scatter_series(x, y, parent=yaxis, label=label)

def create_regression_model(result_data):
    global X_train, y_train
    df = pd.DataFrame(result_data)
    
    # Untuk visualisasi, gunakan semua data
    X_train = df[['L', 'a', 'b', 'luas_kerusakan']]
    y_train = df['umur_simpan']
    
    # Visualisasi dengan semua data
    dpg.delete_item("plot_container", children_only=True)
    
    with dpg.group(horizontal=True, parent="plot_container"):
        create_scatter_plot(y_train.tolist(), X_train['L'].tolist(), "Nilai L", "plot_L")
        create_scatter_plot(y_train.tolist(), X_train['a'].tolist(), "Nilai a", "plot_a")
    
    with dpg.group(horizontal=True, parent="plot_container"):
        create_scatter_plot(y_train.tolist(), X_train['b'].tolist(), "Nilai b", "plot_b")
        create_scatter_plot(y_train.tolist(), X_train['luas_kerusakan'].tolist(), "Luas Kerusakan", "plot_damage")
    
    return X_train.shape, y_train.unique()

def testing_create_regression_model(path):
    
    X_train, y_train = sm.load_data_for_plots(path)

    dpg.delete_item("prediction_plot_container", children_only=True)
    
    with dpg.group(horizontal=True, parent="prediction_plot_container"):
        create_scatter_plot(y_train.tolist(), X_train['L'].tolist(), "Nilai L", "plot_L")
        create_scatter_plot(y_train.tolist(), X_train['a'].tolist(), "Nilai a", "plot_a")
    
    with dpg.group(horizontal=True, parent="prediction_plot_container"):
        create_scatter_plot(y_train.tolist(), X_train['b'].tolist(), "Nilai b", "plot_b")
        create_scatter_plot(y_train.tolist(), X_train['luas_kerusakan'].tolist(), "Luas Kerusakan", "plot_damage")
    
    return X_train.shape, y_train.unique()

def run_analysis():
    global training_dataset, healthy_lab_color, total_images, progress, count

    progress = 0
    total_images = 0
    count = 0

    dpg.set_value("progress_text", "Starting analysis...")
    dpg.configure_item("progress_bar", show=True)
    dpg.configure_item("progress_text", show=True)
    dpg.disable_item("start_analysis_button")
    dpg.set_item_label("start_analysis_button", "Analyzing...")

    total_images = count_images_in_folder(training_input_folder)
    thread = threading.Thread(target=update_progress_status, args=(total_images,))
    thread.start()

    result_data, healthy_color, model_filename = tr.process_images_in_folder(training_input_folder, training_output_folder, current_model_name, training_output_folder)

    training_dataset = result_data
    healthy_lab_color = healthy_color

    a, b = create_regression_model(result_data)

    if model_filename is not None:
        get_result_info(model_filename, a, b)
    
    dpg.enable_item("start_analysis_button")
    dpg.set_item_label("start_analysis_button", "Analyze")

def get_result_info(model_filename, a, b):
    count, info, folder_count, avg_per_folder = tr.look_for_count()
    
    coefficients, intercept = pm.get_info(model_filename)
    
    info_text = f"""
    Total data analyzed: {count}
    Total folders scanned: {folder_count}
    Average data per folder: {avg_per_folder}
    X shape: {a}
    Unique y values: {b}
    Regression coefficients (weights): {coefficients}
    Intercept (bias): {intercept:.4f}
    """
    
    dpg.configure_item("result_info", show=True)
    dpg.set_value("result_info", info_text)

def select_model_file(sender, app_data):
    global testing_model_path
    root = Tk()
    root.withdraw()
    model_path = filedialog.askopenfilename(title="Select Model File", filetypes=[("Pickle files", "*.pkl")])
    if model_path:
        testing_model_path = model_path
        model_filename = os.path.basename(testing_model_path)
        dpg.set_value("selected_model", model_filename)
        info = f"Selected model: {model_filename}"
        dpg.set_value("prediction_model_name", info)
        validate_testing_parameters()
    root.destroy()

def select_test_image(sender, app_data):
    global testing_image_path
    root = Tk()
    root.withdraw()
    testing_image_path = filedialog.askopenfilename(
        title="Select Image File", 
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if testing_image_path:
        image_filename = os.path.basename(testing_image_path)
        dpg.set_value("selected_image", image_filename)
        validate_testing_parameters()
    root.destroy()

def validate_testing_parameters():
    if testing_model_path and testing_image_path:
        dpg.enable_item("predict_button")
    else:
        dpg.disable_item("predict_button")

def run_prediction():
    global testing_model_path, testing_image_path

    thread = threading.Thread(target=prediction_update_progress_status)
    thread.start()

    prediction_result, L, a, b, damaged, healthy = ts.test_model(testing_model_path, testing_image_path)
    prediction = f"Day {round(prediction_result, 2)}"
    dpg.configure_item("prediction_result", show=True)
    dpg.set_value("prediction_result", prediction)
    info = f"""
    L*: {round(L, 2)}
    a*: {round(a, 2)}
    b*: {round(b, 2)}
    Damaged area: {damaged} (pixels)
    Healthy color used: {healthy}
    """
    dpg.configure_item("prediction_label", show=True)
    dpg.configure_item("prediction_info", show=True)
    dpg.set_value("prediction_info", info)
    dpg.configure_item("prediction_model_name", show=True)

    testing_create_regression_model(testing_model_path)

def prediction_update_progress_status():
    dpg.configure_item("prediction_progress_bar", show=True)
    dpg.configure_item("prediction_progress_text", show=True)
    while True:
        inf, pr = ts.get_info()

        progress = pr / 100

        info = f"{round(pr)}% - {inf}"

        dpg.set_value("prediction_progress_bar", progress)
        
        dpg.set_value("prediction_progress_text", info)

        if progress >= 100:
            dpg.set_value("prediction_progress_text", "Complete")
            break
        time.sleep(0.2)

def shorten_path(
    path, 
    max_length=40, 
    min_chars=5, 
    head=3,
    tail=3
):
    if len(path) <= max_length:
        return path
    
    parts = [p for p in path.split('/') if p]
    if len(parts) <= (head + tail):
        return path
    
    first_part = '/'.join(parts[:head])
    last_part = '/'.join(parts[-tail:])
    shortened = f"{first_part}/.../{last_part}"
    
    if len(shortened) > max_length:
        remaining_space = max_length - (len(".../") + 1)
        first_allowed = max(min_chars, remaining_space // 2)
        last_allowed = max(min_chars, remaining_space - first_allowed)
        
        first_part = first_part[:first_allowed]
        last_part = last_part[-last_allowed:] if last_allowed < len(last_part) else last_part
        shortened = f"{first_part}.../{last_part}"
    
    return shortened

dpg.create_context()

with dpg.theme() as light_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (255, 255, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (255, 255, 255))
        dpg.add_theme_color(dpg.mvThemeCol_Border, (64,64,64))
        dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 0, 0))
        dpg.add_theme_color(dpg.mvThemeCol_Button, (189, 189, 189))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (189, 189, 189))
        dpg.add_theme_color(dpg.mvThemeCol_TabActive, (20, 252, 229))
        dpg.add_theme_color(dpg.mvThemeCol_Tab, (189, 189, 189))
        dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (255, 255, 255))
        dpg.add_theme_color(dpg.mvThemeCol_PlotLines, (20, 252, 229))


with dpg.font_registry():
    font_regular = dpg.add_font("Roboto-Regular.ttf", 16)

dpg.bind_theme(light_theme)

dpg.bind_font(font_regular)

with dpg.window(tag="main_window", label="Cabbage Analyzer", width=1920, height=1080, 
               no_collapse=True, no_move=True, no_resize=True):
    with dpg.tab_bar():
        with dpg.tab(label="Training"):
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text("Select data folder and name your model:")
                    dpg.add_button(label="Select Data Folder", callback=select_input_folder, height=40, width=300)
                    dpg.add_button(label="Select Output Folder", callback=select_output_folder, height=40, width=300)
                    dpg.add_spacer(height=10)
                    dpg.add_text("", show=False, tag="input_folder_path")
                    dpg.add_text("", show=False, tag="output_folder_path")
                    dpg.add_spacer(height=10)
                    dpg.add_input_text(label="Model Name", tag="model_name_input", 
                                    callback=handle_model_name_input, width=300)
                    dpg.add_button(label="Analyze", tag="start_analysis_button", 
                                callback=run_analysis, enabled=False, show=False, width=300, height=40)
                dpg.add_spacer(width=20)
                with dpg.group():
                    dpg.add_child_window(tag="plot_container", width=900, height=900)
                with dpg.group():
                    dpg.add_text("", tag="result_info", show=False)
            with dpg.group():
                dpg.add_text("", tag="progress_text", show=False)
                dpg.add_progress_bar(tag="progress_bar", default_value=0.0, width=1900, show=False)
        
        with dpg.tab(label="Testing"):
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text("Select model and test image:")
                    dpg.add_button(label="Select Model", callback=select_model_file, width=300, height=40)
                    dpg.add_button(label="Select Image", callback=select_test_image, width=300, height=40)
                with dpg.group():
                    dpg.add_text("")
                    dpg.add_input_text(label="Model Path", default_value="", tag="selected_model", 
                                    readonly=True, width=300, height=40)
                    dpg.add_input_text(label="Image Path", default_value="", tag="selected_image", 
                                    readonly=True, width=300, height=40)
            dpg.add_button(label="Predict", tag="predict_button", callback=run_prediction, 
                                enabled=False, width=300, height=40)
            dpg.add_separator()
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text("Prediction result:", tag="prediction_label", show=False)
                    dpg.add_text("", tag="prediction_result", show=False)
                    dpg.add_text("")
                    dpg.add_text("", tag="prediction_info", show=False)
                    dpg.add_text("", tag="prediction_model_name", show=False)
                with dpg.group():
                    dpg.add_child_window(tag="prediction_plot_container", width=900, height=700)
            dpg.add_text("", tag="prediction_progress_text", show=False)
            dpg.add_progress_bar(tag="prediction_progress_bar", default_value=0.0, width=1900, show=False)
dpg.create_viewport(title='Cabbage Analyzer', width=1920, height=1080, resizable=False)
dpg.setup_dearpygui()
dpg.show_viewport()

def handle_application_exit():
    dpg.destroy_context()
    exit()

dpg.set_exit_callback(handle_application_exit)
dpg.set_primary_window("main_window", True)
dpg.start_dearpygui()