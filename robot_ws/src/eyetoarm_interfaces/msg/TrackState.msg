# One confirmed A world track in a TrackStateArray. Units/frame/time as SortPart.action
# (position mm, SIM_TIME ns, frame base_link = world). Unknown float values are NaN.

string part_track_id             # same string as SortPart.part_track_id
string part_class                # BOLT_A | BOLT_B (A projection-height kind, not the B decision); empty if unknown
string state                     # VISIBLE | PREDICTED | OCCLUDED | GRIPPER
float64[3] position_mm           # center at capture_stamp_ns (predicted if not VISIBLE), z = projection plane
float64[3] velocity_mm_per_s     # velocity used for the prediction (x = 0: belt along +Y)
string velocity_source           # TRACK | BELT_COMMON | NONE
float64 axis_yaw_deg             # bolt axis yaw in base_link, folded to [-90, 90)
float64 length_mm                # axis length
float64[3] endpoint_a_mm         # center - length/2 along axis_yaw_deg
float64[3] endpoint_b_mm         # center + length/2 along axis_yaw_deg
int64 last_hit_stamp_ns          # SIM_TIME of the last position update (ms resolution)
string last_hit_camera           # link | gripper
bool confirmed                   # always true in this version
