import subprocess
import pprint
import numpy


# Hand position training and testing
# scaling_process = subprocess.run(['svm-scale.exe', '-s',
#                                   'range', 'hand_2p_unscaled.train', '>',
#                                   'hand_2p_scaled.train'],
#                                  shell=True,  # Necessary for the > operator
#                                  stdout=subprocess.PIPE)
#
# training_process = subprocess.run(['svm-train.exe',
#                                    '-v', '10',  # 10 fold cross validation
#                                    '-g', '10',  # Gamma in kernel function
#                                    '-c', '10',  # Cost
#                                    'hand_2p_scaled.train'],
#                                   stdout=subprocess.PIPE)

scaling_process = subprocess.run(['svm-scale.exe', '-s',
                                  'range', 'tap_occurrence_unscaled.train', '>',
                                  'tap_occurrence_scaled.train'],
                                 shell=True,  # Necessary for the > operator
                                 stdout=subprocess.PIPE)

results = []
for c_power in numpy.linspace(10.9, 11.5, 15):
    for gamma_power in numpy.linspace(-9.6, -8.9, 15):
        cost = 2 ** c_power
        gamma = 2 ** gamma_power
        training_process = subprocess.run(
                ['svm-train.exe',
                 '-v', '10',  # 10 fold cross validation
                 '-g', str(gamma),  # Gamma in kernel function
                 '-c', str(cost),  # Cost
                 'tap_occurrence_scaled.train'],
                stdout=subprocess.PIPE
        )

        result_string = str(
                training_process.stdout.decode('utf-8')
        ).splitlines()
        percentage = float(result_string[-1].split(" ")[-1][:-1])
        results.append([c_power, gamma_power, percentage])

percentage_list = [result[2] for result in results]
max_percentage = max(percentage_list)
max_percentage_index = percentage_list.index(max_percentage)

pprint.pprint(results)
pprint.pprint(results[max_percentage_index])
