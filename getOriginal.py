import os

write_path = "originalData"
directory_path = "DataOriginal"

# Iterate through all files in the directory
for filename in os.listdir(directory_path):
    lines = []
    if os.path.isfile(os.path.join(directory_path, filename)):
        with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as f:
            lines = f.readlines()
    
        with open(write_path, "a", encoding="utf-8") as f:
            f.writelines(lines)
