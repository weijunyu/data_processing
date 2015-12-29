import subprocess


# Hand position training and testing
training_process = subprocess.run(['svm-train.exe', 'hand_2p.train'],
                                  stdout=subprocess.PIPE)
testing_process = subprocess.run(['svm-predict.exe', 'hand_2p.test',
                                  'hand_2p.train.model', 'hand_2p.output'],
                                 stdout=subprocess.PIPE)
print(testing_process.stdout)