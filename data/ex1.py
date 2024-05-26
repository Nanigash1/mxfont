import os

folder_path = "ttfs/train"  # Replace with the actual path to your folder
file_names_to_delete = ['ALS Hauss Variable GX 0.907', 'ALS Hauss Variable GX 0.907', 'ApocLC-Regular.220108-2142', 'ApocLC-Regular.220108-2142', 'CYR_CT_OP_Emberly Regular [wdth,wght]-VF', 'CYR_CT_OP_Emberly Regular [wdth,wght]-VF', 'Raleway-Italic[wght]', 'Raleway-Italic[wght]', 'Raleway[wght]', 'Raleway[wght]']

for filename in file_names_to_delete:
    for ext in ['.ttf', '.txt']:
        file_path = os.path.join(folder_path, filename + ext)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")