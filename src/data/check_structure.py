import os

def check_existing_file(file_path):
    """Check if a file already exists. If it does, ask if we want to overwrite it."""
    if os.path.isfile(file_path):
        while True:
            response = input(f"File {os.path.basename(file_path)} already exists. Do you want to overwrite it? (y/n): ")
            if response.lower() == 'y':
                return True
            elif response.lower() == 'n':
                return False
            else:
                print("Invalid response. Please enter 'y' or 'n'.")
    else:
        return True

def check_existing_folder(folder_path):
    """Check if a folder exists. If not, create it."""
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
        return False
    return True
