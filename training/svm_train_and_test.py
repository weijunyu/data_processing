import subprocess


# Hand position training and testing
scaling_process = subprocess.run(['svm-scale.exe', '-s',
                                  'range', 'hand_2p_unscaled.train', '>',
                                  'hand_2p_scaled.train'],
                                 shell=True,  # Necessary for the > operator
                                 stdout=subprocess.PIPE)

training_process = subprocess.run(['svm-train.exe',
                                   '-v', '10',  # 10 fold cross validation
                                   '-g', '10',  # Gamma in kernel function
                                   '-c', '10',  # Cost
                                   'hand_2p_scaled.train'],
                                  stdout=subprocess.PIPE)

print(training_process.stdout.decode('utf-8'))
