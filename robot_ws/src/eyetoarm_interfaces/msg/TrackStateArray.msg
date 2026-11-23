# A world tracker snapshot after one link frame (/a/track_states). Units/frame/time as SortPart.action.
# kind=reset: drop every track of earlier messages. kind=marker: run end (tracks empty).

string kind                      # frame | reset | marker
string run_id                    # A run id
string source_id                 # same as SortPart goal
int64 image_seq                  # link frame image_seq (0 for reset/marker)
int64 generation                 # A processing generation (-1 if unknown)
int64 tracker_epoch              # A tracker epoch (+1 on reset/scope change)
int64 capture_stamp_ns           # capture time of the link frame (SIM_TIME, 0 for reset/marker)
string timestamp_base            # SIM_TIME
string frame_id                  # base_link
int64 publish_stamp_ns           # A SIM clock at publish (latency = publish - capture)
float64 belt_velocity_mm_per_s   # A belt common velocity along +Y (NaN if not estimated)
string reset_reason              # kind=reset only
bool handoff_on                  # A gripper handoff enabled
TrackState[] tracks              # confirmed tracks only
