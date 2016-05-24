import random


def generate_test_file(file_name):
    with open(file_name + ".train", 'r') as file:
        content = file.readlines()
        num_of_lines = len(content)
        test_size = int(num_of_lines / 10)
        test_file_lines = []
        for i in range(test_size):
            # Get random line
            test_line_index = random.randint(0, num_of_lines-1)
            # Remove it from the list of lines
            test_line = content.pop(test_line_index)
            # Add it to the test file
            test_file_lines.append(test_line)
            num_of_lines = len(content)
        with open(file_name + "_dec.test", 'w') as test_file:
            test_file.writelines(test_file_lines)
        with open(file_name + "_dec.train", 'w') as train_file:
            train_file.writelines(content)


generate_test_file("tap_occurrence_scaled")
