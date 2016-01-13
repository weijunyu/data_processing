import subprocess


scaling_process_1 = subprocess.run(['svm-scale.exe', '-s',
                                    'range_tap_occurrence',
                                    'tap_occurrence_unscaled.train',
                                    '>',
                                    'tap_occurrence_scaled.train'],
                                   shell=True,  # Necessary for the > operator
                                   stdout=subprocess.PIPE)

scaling_process_2 = subprocess.run(['svm-scale.exe', '-s',
                                    'range_tap_occurrence_new',
                                    'tap_occurrence_unscaled_new_features.train',
                                    '>',
                                    'tap_occurrence_scaled_new_features.train'],
                                   shell=True,  # Necessary for the > operator
                                   stdout=subprocess.PIPE)

scaling_process_3 = subprocess.run(['svm-scale.exe', '-s',
                                    'range_hand_2p',
                                    'hand_2p_unscaled.train',
                                    '>',
                                    'hand_2p_scaled.train'],
                                   shell=True,  # Necessary for the > operator
                                   stdout=subprocess.PIPE)

scaling_process_4 = subprocess.run(['svm-scale.exe', '-s',
                                    'range_hand_5p',
                                    'hand_5p_unscaled.train',
                                    '>',
                                    'hand_5p_scaled.train'],
                                   shell=True,  # Necessary for the > operator
                                   stdout=subprocess.PIPE)

scaling_process_5 = subprocess.run(['svm-scale.exe', '-s',
                                    'range_hand_5p_new_features',
                                    'hand_5p_unscaled_new_features.train',
                                    '>',
                                    'hand_5p_scaled_new_features.train'],
                                   shell=True,  # Necessary for the > operator
                                   stdout=subprocess.PIPE)