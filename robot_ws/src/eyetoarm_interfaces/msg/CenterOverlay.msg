# B center overlay image + the same source-frame keys as the /sort_part goal (C request 2026-11-20).
# C joins this image and a SortPart goal by: source_id, image_seq, capture_generation, generation,
# capture_stamp_ns, timestamp_base (all copied unchanged from the A frame/observation, same values as the goal).
# Grasp/place coordinates are NOT here: they stay in the /sort_part goal.

string kind                  # accepted | status | marker (marker = run end, keys empty/0)
string run_id                # A run id
string source_id             # same as SortPart goal
int64 image_seq              # same as SortPart goal
int64 capture_generation     # same as SortPart goal
int64 generation             # same as SortPart goal
int64 job_seq                # same as SortPart goal
int64 capture_stamp_ns       # same as SortPart goal; also image.header.stamp
string timestamp_base        # SIM_TIME, same as SortPart goal
string calibration_id        # same as SortPart goal
sensor_msgs/Image image      # bgr8 annotated frame, header.frame_id = optical frame
