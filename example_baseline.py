import math
import random
import time

import numpy as np
import pybullet as p
import pybullet_data

from panda_robot import PandaRobot
from helper_functions import animate_path, execute_rrt_star, get_arm_joint_limits, sample_feasible_goals, merge_paths, set_arm_configuration, configuration_feasible, benchmark_control_modes

INCLUDE_GRIPPER = True
SAMPLING_RATE = 1e-3 # 1000 Hz
NUM_GOALS = 5 # number of goals to plan for
MAX_ITER_PER_SEGMENT = 120 # max number of iterations per segment
STEER_STEP = 0.25  # rad; max extension per RRT* edge
GOAL_BIAS = 0.15 # goal bias
GOAL_REACH_EPS = 0.2  # rad (L2) to treat goal as reached
EDGE_DISCRETIZATION_STEPS = 24 # number of discretization steps per edge
NEAR_RADIUS_SCALE = 1.8 # near radius scale



def create_goal_ball(radius=0.1, rgba=(1, 0, 0, 0.8), position=[0.5, 0, 0.8], orientation=[0, 0, 0, 1]):
    """Create a visual goal ball"""
    vis_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=rgba
    )
    ball_id = p.createMultiBody(
        baseMass=0.0,                    # static
        baseCollisionShapeIndex=-1,      # <- no collision
        baseVisualShapeIndex=vis_id,
        basePosition=position,
        baseOrientation=orientation,
    )
    return ball_id

def fk_ee_position(panda_robot, q_arm_7):
    set_arm_configuration(panda_robot, q_arm_7)  # from helper_functions
    ee_link = panda_robot.dof  # matches IK in PandaRobot
    pos = p.getLinkState(panda_robot.robot_id, ee_link)[4]
    return pos

# ----------------------------------------------------------------------------------------------------------------------
def main():
    """Main function to run the RRT* algorithm and animate the path."""
    # Set random seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Connect to PyBullet and set up the environment
    p.connect(p.GUI)
    p.setTimeStep(SAMPLING_RATE)
    p.setGravity(0, 0, -9.81)

    # Setup plane
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    # Setup box obstacle
    box_half_extents = [0.12, 0.25, 0.18]  # x, y, z half extents
    box_center = [0.45, 0.0, 0.35] # [0.45, 0.0, 0.35]  # x, y, z center
    box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half_extents)
    obstacle_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=box_id, basePosition=box_center)

    # R2D2 as a second static obstacle (URDF from pybullet_data)
    r2d2_id = p.loadURDF(
        "r2d2.urdf",
        basePosition=[0.55, 0.45, 0.0],
        useFixedBase=True,
    )
    obstacle_ids = [obstacle_id, r2d2_id]

    # Setup Panda robot
    panda_robot = PandaRobot(include_gripper=INCLUDE_GRIPPER)
    lowers, uppers = get_arm_joint_limits(panda_robot.robot_id)
    
    # Get start configuration
    q_start, _ = panda_robot.get_position_and_velocity()
    q_start = q_start[:7]   # only the arm joints (first 7)

    # Check if start configuration is feasible
    if not configuration_feasible(panda_robot, q_start, obstacle_ids):
        print("Start configuration in collision; reset to zero.")
        q_start = [0.0] * 7
        set_arm_configuration(panda_robot, q_start)
        panda_robot.set_target_positions(q_start)

    # Sample feasible goals
    random_goals = True 
    if random_goals:
        goals = sample_feasible_goals(panda_robot, lowers, uppers, obstacle_ids, NUM_GOALS)
        marker_ids = []
        colors = [[1, 0, 0, 0.9], [0, 1, 0, 0.9], [0, 0, 1, 0.9], [1, 1, 0, 0.9], [1, 0, 1, 0.9]]

        for i, q_g in enumerate(goals):
            ee_pos = fk_ee_position(panda_robot, q_g)
            mid = create_goal_ball(radius=0.04, rgba=colors[i % len(colors)], position=ee_pos)
            marker_ids.append(mid)
    else: # use ball as goal
        goal_position = [-0.9, 0, 0.7]
        goal_orient = p.getQuaternionFromEuler([0.0, 0.0, 0.0])# [0, 0, 0, 1]
        goal_ball_id = create_goal_ball(position=goal_position)
        #goal_orient = p.getQuaternionFromEuler([math.pi, 0.0, 0.0]) # if in RPY
        ik = panda_robot.calculate_inverse_kinematics(goal_position, goal_orient)
        goals = [list(ik[:7])]  # first 7 are arm joints
        if goals[0] is None:
            print("Could not calculate inverse kinematics for goal ball; disconnecting.")
            p.disconnect()
            return

    print ("Goals: ", goals)
    if len(goals) < NUM_GOALS:
        print(
            "Could only sample {} feasible goals (collision-free). "
            "Try moving the obstacles or increasing max tries.".format(len(goals))
        )

    # RRT* execution
    full_segments = execute_rrt_star(panda_robot, q_start, goals, lowers, uppers, obstacle_ids)

    if not full_segments:
        print("No feasible path; disconnecting.")
        p.disconnect()
        return

    path = merge_paths(full_segments) #merge the paths into a single path
    print("Merged path length: {} waypoints.".format(len(path)))
    set_arm_configuration(panda_robot, path[0]) #set the start configuration for the robot
    panda_robot.set_target_positions(path[0]) #set the start configuration for the robot
    for _ in range(20): #wait for 20 steps to stabilize the robot
        p.stepSimulation()
    animate_path(
        panda_robot,
        path,
        hold_steps_per_vertex=200,
        control_mode="position",
        kp=200.0,
        kd=30.0,
        max_torque=87.0,
        print_tracking_error=True,
    ) #animate the path with a single selected controller
    # benchmark_control_modes(
    #     panda_robot,
    #     path,
    #     hold_steps_per_vertex=200,
    #     kp=200.0,
    #     kd=30.0,
    #     max_torque=87.0,
    # ) #run position, torque_pd, and computed_torque on same path

    p.disconnect()
    print("Simulation end")


if __name__ == "__main__":
    main()
