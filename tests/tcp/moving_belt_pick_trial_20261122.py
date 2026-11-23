import sys, time
from collections import deque
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'scripts'))
import numpy as np
import rclpy
from gz.msgs10.double_pb2 import Double
from control_msgs.action import ParallelGripperCommand
from control_msgs.msg import JointTrajectoryControllerState
from sensor_msgs.msg import JointState
from pick_bolt_demo import (
    Arm, PartTracker, forward, load_chain, solve_ik, quaternion_matrix, transform,
    GRIPPER_JOINT, GRASP_OFFSET_FROM_MODEL_ORIGIN, SHANK_RADIUS,
    TIP_CLEARANCE, FINGER_REACH, GRASP_HEIGHT_OFFSET,
)

class StampedTracker(PartTracker):
    def __init__(self, world, part):
        self.history = deque(maxlen=2000)
        super().__init__(world, part)

    def _on_poses(self, message):
        for pose in message.pose:
            if pose.name == self._part:
                p, q, stamp = pose.position, pose.orientation, message.header.stamp
                with self._lock:
                    self._pose = (np.array([p.x,p.y,p.z]), (q.w,q.x,q.y,q.z),
                                  stamp.sec + stamp.nsec * 1e-9)
                    self.history.append(self._pose)
                break

    def motion(self, window=.38):
        current = self.pose()
        with self._lock:
            old = next((item for item in reversed(self.history)
                        if item[2] <= current[2] - window), current)
        dt = current[2] - old[2]
        if not .05 <= dt <= 1.0:
            return current, np.zeros(2)
        return current, (grasp(current)[0][:2] - grasp(old)[0][:2]) / dt

part = sys.argv[1]
bins = {'part_00': (0.72,-0.05), 'part_01': (0.72,0.17)}
tracker = StampedTracker('sort_cell', part)
pub = tracker._node.advertise('/model/conveyor/link/link/track_cmd_vel', Double)
rclpy.init()
node = rclpy.create_node('moving_belt_pick_trial')
arm = Arm(node)
controller_samples = []
node.create_subscription(
    JointTrajectoryControllerState,
    '/joint_trajectory_controller/controller_state',
    lambda message: controller_samples.append((
        message.joint_names,
        tuple(message.error.positions),
        message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
    )),
    100,
)
chain = load_chain()
home = arm.joints()
down = np.array([0.,0.,-1.])
speed = float(sys.argv[2]) if len(sys.argv)>2 else .012
height_offset = float(sys.argv[3]) if len(sys.argv)>3 else 0.

def grasp(sample):
    p, quat, _ = sample
    axis = quaternion_matrix(*quat) @ np.array([0.,0.,-1.])
    length = np.linalg.norm(axis[:2])
    if length < 1e-5:
        raise RuntimeError('bolt axis is vertical')
    flat = axis[:2] / length
    pb = p + axis * GRASP_OFFSET_FROM_MODEL_ORIGIN[part]
    z = (pb[2] - SHANK_RADIUS[part] + TIP_CLEARANCE + FINGER_REACH
         + GRASP_HEIGHT_OFFSET[part] + height_offset)
    return pb, z, np.array([-flat[1],flat[0],0.])

def belt(speed):
    for _ in range(4):
        pub.publish(Double(data=speed))
        time.sleep(.08)

def sample_after(seconds):
    old = tracker.pose()
    end = time.monotonic() + 25
    while time.monotonic() < end:
        current = tracker.pose()
        if current[2] - old[2] >= seconds:
            return old, current
        time.sleep(.03)
    raise RuntimeError('Gazebo pose timestamps did not advance')

def follow(points, seconds):
    sample_start = len(controller_samples)
    error = arm.follow(points, seconds)
    samples = controller_samples[sample_start:]
    if samples:
        names = samples[-1][0]
        max_by_joint = {
            name: max(abs(sample[1][index]) for sample in samples)
            for index, name in enumerate(names)
            if all(len(sample[1]) > index for sample in samples)
        }
        print(f'ARM_DIAG duration_sim_s={seconds:.3f} samples={len(samples)} '
              f'max_error_by_joint_rad={max_by_joint} final_error_rad={error:.6f}', flush=True)
    else:
        print('ARM_DIAG controller_state_samples=0; checking /joint_trajectory_controller/controller_state',
              flush=True)
    if error > .012:
        raise RuntimeError(f'arm tracking error {error:.4f} rad, goal={points[-1]}, actual={arm.joints()}')

try:
    arm.grip(0.)
    initial_tool = forward(chain,home)
    if initial_tool[2,3] < .20:
        initial_tool[2,3] = .22
        follow([solve_ik(chain,initial_tool,home)],1.2)
        home = arm.joints()
    print(f'{part} CONFIG belt_command_mps={speed} height_offset_mm={height_offset*1000:.1f}',flush=True)
    belt(speed)
    a,b = sample_after(1.5)
    vel = (b[0][1]-a[0][1])/(b[2]-a[2])
    print(f'{part} BELT_MEASURED sim={a[2]:.3f}->{b[2]:.3f} y={a[0][1]:.5f}->{b[0][1]:.5f} velocity={vel:.5f} m/s',flush=True)
    if abs(vel) < .001:
        raise RuntimeError('bolt did not move with the belt')
    pb,z,closing = grasp(tracker.pose())
    options = []
    for sign in (1.,-1.):
        c=closing*sign
        R=np.column_stack([c,np.cross(down,c),down])
        try:
            q=solve_ik(chain, transform(R,[pb[0],pb[1]+vel*3,z+.065]),home)
            if np.max(np.abs(q)) < 6.2 and abs(q[2]) < 3.1:
                options.append((np.linalg.norm(q-home),R,q))
        except RuntimeError:
            pass
    _, R, above = min(options,key=lambda x:x[0])
    follow([above],3.)
    arm.grip(.62 if part == 'part_00' else .55, settle=.1)
    def trajectory(start_xyz,end_xyz,seed,R,steps=24):
        result=[]
        initial=forward(chain,seed)[:3,0]
        angle=np.arctan2(initial[0]*R[1,0]-initial[1]*R[0,0],np.dot(initial[:2],R[:2,0]))
        yaw=np.arctan2(initial[1],initial[0])
        for i in range(1,steps+1):
            xyz=np.asarray(start_xyz)+(np.asarray(end_xyz)-start_xyz)*i/steps
            heading=yaw+angle*i/steps
            closing=np.array([np.cos(heading),np.sin(heading),0.])
            pose_rotation=np.column_stack([closing,np.cross(down,closing),down])
            seed=solve_ik(chain,transform(pose_rotation,xyz),seed)
            if np.max(np.abs(seed)) >= 6.2 or abs(seed[2]) >= 3.1:
                raise RuntimeError(f'joint limit in Cartesian path: {seed}')
            result.append(seed)
        return result
    def track_step(clearance, phase, seconds=.32):
        sample, velocity = tracker.motion()
        pb, z, closing = grasp(sample)
        if not (.455 < pb[0] < .645 and abs(pb[1]) < .54 and .097 < pb[2] < .12):
            raise RuntimeError(f'bolt left the pick region: {pb}')
        if np.dot(closing, R[:,0]) < 0:
            closing = -closing
        angle=np.degrees(np.arccos(np.clip(np.dot(closing,R[:,0]),-1.,1.)))
        if angle > 45 and clearance < .014:
            raise RuntimeError(f'bolt yaw changed {angle:.1f} deg near contact')
        if angle > 30:
            seconds=max(seconds,.65)
        orientation = np.column_stack([closing,np.cross(down,closing),down])
        q = arm.joints()
        start = forward(chain,q)[:3,3]
        end = np.r_[pb[:2] + np.clip(velocity, -.05, .05) * (seconds+.045), z+clearance]
        follow(trajectory(start,end,q,orientation,10),seconds)
        actual = forward(chain,arm.joints())[:3,3]
        latest = tracker.pose()
        observed, goal_z, _ = grasp(latest)
        error = actual - np.r_[observed[:2],goal_z+clearance]
        print(f'{part} {phase} sim={latest[2]:.3f} bolt_xy={latest[0][:2]} '
              f'vel_xy={velocity} error_xyz_mm={np.round(error*1000,2)}',flush=True)
        return orientation, error

    for clearance in (.050,.036,.024,.014,.007,0.):
        R, error = track_step(clearance,'DESCEND')
    for _ in range(4):
        if np.linalg.norm(error[:2]) <= .0035:
            break
        R, error = track_step(0.,'ALIGN')
    if np.linalg.norm(error[:2]) > .0035:
        raise RuntimeError('gripper missed the moving grasp point')
    grip=ParallelGripperCommand.Goal()
    grip.command=JointState(name=[GRIPPER_JOINT],position=[.8])
    pending=arm.gripper.send_goal_async(grip)
    result = None
    for _ in range(5):
        R, error = track_step(0.,'CLOSE')
        if pending.done() and result is None:
            handle = pending.result()
            if not handle.accepted:
                raise RuntimeError('gripper goal rejected')
            result = handle.get_result_async()
        if result is not None and result.done():
            print(f'{part} GRIPPER status={result.result().status} stalled={result.result().result.stalled}',flush=True)
            break
    before_lift=tracker.pose()
    current_q=arm.joints()
    current=forward(chain,current_q)[:3,3]
    probe = current + np.array([0.,0.,.020])
    follow(trajectory(current,probe,current_q,R,10),.55)
    _, observed_lift=sample_after(.25)
    dz=observed_lift[0][2]-before_lift[0][2]
    relative_shift = np.linalg.norm(observed_lift[0]-before_lift[0]- (forward(chain,arm.joints())[:3,3]-current))
    print(f'{part} LIFT dz_mm={dz*1000:.1f} relative_shift_mm={relative_shift*1000:.1f} '
          f'knuckle={arm.state[GRIPPER_JOINT]:.4f}',flush=True)
    if dz < .014 or relative_shift > .008:
        arm.grip(0.,settle=.1)
        q=arm.joints()
        tip=forward(chain,q)[:3,3]
        follow(trajectory(tip,tip+np.array([0.,0.,.055]),q,R,10),.8)
        raise RuntimeError('moving grasp did not lift the bolt')
    bx,by=bins[part]
    q=arm.joints()
    tip=forward(chain,q)[:3,3]
    lifted=tip.copy()
    lifted[2]=max(.24,tip[2]+.075)
    follow(trajectory(tip,lifted,q,R),1.2)
    over=np.array([bx,by,lifted[2]])
    q=arm.joints()
    follow(trajectory(forward(chain,q)[:3,3],over,q,R,30),3.)
    on_bin=tracker.pose()[0]
    print(f'{part} CARRY xyz={on_bin}',flush=True)
    if abs(on_bin[0]-bx)>.04 or abs(on_bin[1]-by)>.06 or on_bin[2]<.10:
        raise RuntimeError('bolt did not reach the bin while held')
    arm.grip(0.)
    _, final=sample_after(1.5)
    p=final[0]
    passed=abs(p[0]-bx)<.053 and abs(p[1]-by)<.073 and .001<p[2]<.08
    print(f'{part} DROP xyz={p} bin_confirmed={passed}',flush=True)
    if not passed:
        raise RuntimeError('released bolt did not settle inside the bin')
finally:
    belt(0.)
    node.destroy_node()
    rclpy.try_shutdown()
