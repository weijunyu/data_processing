# Program starts here ----------------------------------------------------------
from training_functions import clean_logs
from training_functions import make_tap_occurrence_data
from training_functions import make_hand_data_5p
from training_functions import make_left_hand_location_data
from training_functions import make_right_hand_location_data
from training_functions import make_tap_occurrence_data_combined, make_hand_data_5p_combined, \
    make_left_hand_location_data_combined, make_right_hand_location_data_combined

clean_logs()

# make_tap_occurrence_data("tap_occurrence")
# make_hand_data_5p("hand_5p")
# make_left_hand_location_data("location_lhand")
# make_right_hand_location_data("location_rhand")

make_tap_occurrence_data_combined("tap_occurrence")
make_hand_data_5p_combined("hand_5p")
make_left_hand_location_data_combined("location_lhand")
make_right_hand_location_data_combined("location_rhand")

# make_tap_occurrence_data_new("tap_occurrence_new_features")
# make_tap_occurrence_data_combined("tap_occurrence_combined")

# make_hand_data_2p("hand_2p")

# make_hand_data_5p_no_angles("hand_5p_no_angles")
# make_hand_data_5p_new("hand_5p_new_features")
# make_hand_data_5p_new_no_angles("hand_5p_new_no_angles")
# make_hand_data_5p_angle("hand_5p_angles_only")
# make_hand_data_5p_combined("hand_5p_combined")

# make_left_hand_location_data_new("location_lhand_new_features")
# make_left_hand_location_data_combined("location_lhand_combined")

# make_right_hand_location_data_new("location_rhand_new_features")
# make_right_hand_location_data_combined("location_rhand_combined")

from training_functions import get_positive_tap_samples
from training_functions import get_negative_tap_samples_full
from training_functions import LEFT_HAND, RIGHT_HAND

# negative_taps = get_negative_tap_samples_full(LEFT_HAND)
# positive_taps = get_positive_tap_samples(LEFT_HAND)
#
# i = 1
# for sensor in positive_taps:
#     print(i)
#     i += 1
#     for location in sensor:
#         print(len(location))