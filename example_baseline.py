import argparse
import math
import random
import time

import numpy as np
import pybullet as p
import pybullet_data

from panda_robot import PandaRobot
from helper_functions import animate_path, execute_rrt_star, get_arm_joint_limits, sample_feasible_goals, merge_paths, set_arm_configuration, configuration_feasible, benchmark_control_modes
from helper_functions import fk_ee_position
import env

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

def load_obstacles():
    obstacle_ids = []
    # Setup box obstacles
    boxes, obstacles = env.boxes, env.obstacles
    for box in boxes:
        box_half_extents = box["half_extents"]
        box_center = box["position"]
        box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half_extents)
        obstacle_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=box_id, basePosition=box_center)
        obstacle_ids.append(obstacle_id)
    # Setup other obstacles, e.g. r2d2
    for obstacle in obstacles:
        obstacle_id = p.loadURDF(
            obstacle["name"] + ".urdf",
            basePosition=obstacle["position"],
            useFixedBase=True,
        )
        obstacle_ids.append(obstacle_id)    

    return obstacle_ids

# ----------------------------------------------------------------------------------------------------------------------
def main():
    """Main function to run the RRT* algorithm and animate the path."""
    parser = argparse.ArgumentParser(description="RRT* baseline: Franka Panda in PyBullet.")
    parser.add_argument(
        "--no-random-goals",
        action="store_false",
        dest="random_goals",
        default=True,
        help="Use a fixed IK goal ball instead of randomly sampled feasible goals.",
    )
    parser.add_argument(
        "--mode",
        choices=["position", "torque_pd", "computed_torque"],
        default="position",
        help="Control mode to use for the robot."
    )
    parser.add_argument(
        "--algo",
        choices=["rrt_star"],
        default="rrt_star",
        help="Algorithm to use for the robot."
    )
    parser.add_argument(
        "--stats_csv_path",
        default="rrt_star_stats.csv",
        help="Path to save the statistics of the algorithm."
    )

    args = parser.parse_args()
    random_goals = args.random_goals
    mode = args.mode
    algo = args.algo
    stats_csv_path = args.stats_csv_path

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

    # Load obstacles
    obstacle_ids = load_obstacles()

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
    if random_goals:
        goals = sample_feasible_goals(panda_robot, lowers, uppers, obstacle_ids, NUM_GOALS)
        marker_ids = []
        colors = [[1, 0, 0, 0.9], [0, 1, 0, 0.9], [0, 0, 1, 0.9], [1, 1, 0, 0.9], [1, 0, 1, 0.9]]

        for i, q_g in enumerate(goals):
            state = fk_ee_position(panda_robot, q_g)
            mid = create_goal_ball(radius=0.04, rgba=colors[i % len(colors)], position=state[4], orientation=state[5])
            marker_ids.append(mid)
        
        if len(goals) < NUM_GOALS:
            print(
                "Could only sample {} feasible goals (collision-free). "
                "Try moving the obstacles or increasing max tries.".format(len(goals))
            )

    else: # use ball as goal
        goals = []
        for goal in env.goals:
            goal_position = goal["position"]
            goal_orient = goal["orientation"]
            goal_ball_id = create_goal_ball(position=goal_position)
            ik = panda_robot.calculate_inverse_kinematics(goal_position, goal_orient)
            goals.append(list(ik[:7]))  # first 7 are arm joints
            if goals[0] is None:
                print("Could not calculate inverse kinematics for goal ball; disconnecting.")
                p.disconnect()
                return

    # RRT* execution or other algorithms
    if algo == "rrt_star":
        full_segments = execute_rrt_star(panda_robot, q_start, goals, 
            lowers, uppers, obstacle_ids, stats_csv_path=stats_csv_path)
    else:
        print("Algorithm not supported.")
        p.disconnect()
        return

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
        control_mode=mode,
        kp=200.0,
        kd=30.0,
        max_torque=87.0,
        print_tracking_error=True,
    ) 
    #animate the path with a single selected controller
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
