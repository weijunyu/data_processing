import os


# Hand position training
os.system("svm-train.exe hand_2p.train")
result = str(os.system(
    "svm-predict.exe hand_2p.test hand_2p.train.model hand_2p.output"))

