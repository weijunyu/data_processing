import subprocess
import os


for file in os.listdir():
    file_name, file_ext = os.path.splitext(file)
    if file_ext == ".unscaled":
        subprocess.run(['svm-scale.exe', '-s',
                        'range_' + file_name,
                        file,
                        '>',
                        file_name + ".scaled"],
                       shell=True,  # Necessary for the > operator
                       stdout=subprocess.PIPE)
