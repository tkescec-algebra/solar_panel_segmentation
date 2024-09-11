import os

def extract_filename(path):
    filename_with_extension = os.path.basename(path)
    filename_without_extension, _ = os.path.splitext(filename_with_extension)
    return filename_without_extension