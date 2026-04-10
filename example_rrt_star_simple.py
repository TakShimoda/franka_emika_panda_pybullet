"""
Minimal RRT* in joint space for the Panda arm, then playback in PyBullet.

Samples five random collision-free joint configurations, plans between consecutive
targets (start -> q1 -> ... -> q5) with a short RRT* segment planner, and animates
with POSITION_CONTROL like example_visualize_movement.py.

This is intentionally small (few iterations per segment) so it runs quickly; increase
MAX_ITER_PER_SEGMENT and STEER_STEP for harder queries.
"""
import math
import random
import time

import numpy as np
import pybullet as p
import pybullet_data

from panda_robot import PandaRobot

INCLUDE_GRIPPER = True
SAMPLING_RATE = 1e-3 # 1000 Hz
NUM_GOALS = 5 # number of goals to plan for
MAX_ITER_PER_SEGMENT = 120 # max number of iterations per segment
STEER_STEP = 0.25  # rad; max extension per RRT* edge
GOAL_BIAS = 0.15 # goal bias
GOAL_REACH_EPS = 0.2  # rad (L2) to treat goal as reached
EDGE_DISCRETIZATION_STEPS = 24 # number of discretization steps per edge
NEAR_RADIUS_SCALE = 1.8 # near radius scale


def get_arm_joint_limits(robot_id, num_arm_joints=7):
    """Get the joint limits for the Panda arm."""
    lowers = [p.getJointInfo(robot_id, j)[8] for j in range(num_arm_joints)]
    uppers = [p.getJointInfo(robot_id, j)[9] for j in range(num_arm_joints)]
    return lowers, uppers

def set_arm_configuration(panda_robot, q_arm):
    """Set arm joints with resetJointState (kinematic pose for collision queries)."""
    rid = panda_robot.robot_id
    for i in range(7):
        p.resetJointState(rid, i, float(q_arm[i]))
    if panda_robot.dof == 9:
        p.resetJointState(rid, 7, 0.0)
        p.resetJointState(rid, 8, 0.0)


def configuration_feasible(panda_robot, q_arm, obstacle_ids):
    set_arm_configuration(panda_robot, q_arm)
    for oid in obstacle_ids:
        pts = p.getClosestPoints(panda_robot.robot_id, oid, distance=0.05)
        for pt in pts:
            if pt[8] < 0.002:
                return False
    return True


def edge_feasible(panda_robot, q_from, q_to, obstacle_ids, steps=EDGE_DISCRETIZATION_STEPS):
    q0, q1 = np.asarray(q_from, dtype=float), np.asarray(q_to, dtype=float)
    for alpha in np.linspace(0.0, 1.0, steps + 1):
        q = (1.0 - alpha) * q0 + alpha * q1 #interpolate between q_from and q_to
        if not configuration_feasible(panda_robot, q.tolist(), obstacle_ids): #check if the interpolated configuration is feasible
            return False
    return True


def distance_q(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def steer(from_q, to_q, step_size):
    """Steer from one configuration to another by a given step size."""
    d = distance_q(from_q, to_q)
    if d < 1e-9: #if the distance is very small, return the from_q configuration
        return list(from_q)
    if d <= step_size: #if the distance is less than or equal to the step size, return the to_q configuration
        return list(to_q)
    s = step_size / d #calculate the step size
    return [from_q[i] + s * (to_q[i] - from_q[i]) for i in range(7)] #return the new configuration


def nearest_index(nodes, q):
    """Find the nearest node to the given configuration."""
    best_i, best_d = 0, float("inf")
    for i, n in enumerate(nodes):
        d = distance_q(n["q"], q)
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def near_indices(nodes, q_new, radius):
    return [i for i, n in enumerate(nodes) if distance_q(n["q"], q_new) < radius]


def rrt_star_segment(panda_robot, q_start, q_goal, lowers, uppers, obstacle_ids):
    """
    Plan from q_start to (near) q_goal. Returns list of configurations from start
    to goal along the tree, or None.
    """
    def sample_uniform():
        return [random.uniform(lo, hi) for lo, hi in zip(lowers, uppers)]

    nodes = [{"q": list(q_start), "parent": None, "cost": 0.0}]

    for _ in range(MAX_ITER_PER_SEGMENT):
        # Sample a random goal with a bias towards the goal
        if random.random() < GOAL_BIAS:
            q_rand = list(q_goal)
        else:
            q_rand = sample_uniform()
            # Check if the sampled goal is feasible
            if not configuration_feasible(panda_robot, q_rand, obstacle_ids):
                continue

        # Find the nearest node to the sampled goal
        i_near = nearest_index(nodes, q_rand) #nearest node index
        q_near = nodes[i_near]["q"] #nearest node configuration
        q_new = steer(q_near, q_rand, STEER_STEP) #new configuration

        if not edge_feasible(panda_robot, q_near, q_new, obstacle_ids): #check if the edge is feasible
            continue

        # Calculate the near radius
        n = len(nodes)
        gamma = NEAR_RADIUS_SCALE
        radius = min(gamma * STEER_STEP, gamma * (math.log(n + 1) / (n + 1)) ** (1.0 / 7.0)) # from Karaman and Frazzoli 2011

        # Find the near indices
        near_is = near_indices(nodes, q_new, radius)
        if not near_is:
            near_is = [i_near]

        # Initialize the best parent and cost
        best_parent = i_near
        best_cost = nodes[i_near]["cost"] + distance_q(nodes[i_near]["q"], q_new)
        # Iterate through the near indices and find the best parent and cost
        for i in near_is:
            c = nodes[i]["cost"] + distance_q(nodes[i]["q"], q_new) #cost to the new node
            if c < best_cost and edge_feasible(panda_robot, nodes[i]["q"], q_new, obstacle_ids): #check if the edge is feasible
                best_cost = c
                best_parent = i

        # Create the new node
        new_node = {"q": q_new, "parent": best_parent, "cost": best_cost}
        new_idx = len(nodes)
        nodes.append(new_node)

        # Iterate through the near indices and update the parent and cost (re-wiring)
        for i in near_is:
            if i == best_parent: #if the node is the best parent, skip
                continue
            alt = best_cost + distance_q(q_new, nodes[i]["q"])
            if alt < nodes[i]["cost"] and edge_feasible(panda_robot, q_new, nodes[i]["q"], obstacle_ids):
                nodes[i]["parent"] = new_idx
                nodes[i]["cost"] = alt

        # Check if the new node is close to the goal
        if distance_q(q_new, q_goal) < GOAL_REACH_EPS:
            if edge_feasible(panda_robot, q_new, q_goal, obstacle_ids):
                goal_idx = len(nodes)
                nodes.append(
                    {
                        "q": list(q_goal),
                        "parent": new_idx,
                        "cost": best_cost + distance_q(q_new, q_goal),
                    }
                )
                return extract_path(nodes, goal_idx)

    i_best = nearest_index(nodes, q_goal) #nearest node index to the goal
    if distance_q(nodes[i_best]["q"], q_goal) < GOAL_REACH_EPS * 2.5: #check if the nearest node is close to the goal
        if edge_feasible(panda_robot, nodes[i_best]["q"], q_goal, obstacle_ids):
            goal_idx = len(nodes)
            nodes.append(
                {
                    "q": list(q_goal),
                    "parent": i_best,
                    "cost": nodes[i_best]["cost"] + distance_q(nodes[i_best]["q"], q_goal),
                }
            )
            return extract_path(nodes, goal_idx)
    return None


def extract_path(nodes, goal_idx):
    """Extract the path from the nodes."""
    path = []
    idx = goal_idx
    # Trace back the path from the goal to the start
    while idx is not None:
        path.append(nodes[idx]["q"])
        idx = nodes[idx]["parent"]
    path.reverse() #reverse the path to start at the start and end at the goal
    return path


def sample_feasible_goals(panda_robot, lowers, uppers, obstacle_ids, count, max_tries=400):
    """Sample feasible goals from the joint limits."""
    goals = []
    for _ in range(max_tries):
        if len(goals) >= count:
            break
        q = [random.uniform(lo, hi) for lo, hi in zip(lowers, uppers)]
        if configuration_feasible(panda_robot, q, obstacle_ids):
            goals.append(q)
    return goals


def merge_paths(segments):
    """Merge the paths into a single path."""
    if not segments:
        return []
    out = list(segments[0])
    for seg in segments[1:]:
        if seg and out and np.allclose(seg[0], out[-1], atol=1e-4): #check if the first configuration of the next segment is close to the last configuration of the current segment
            out.extend(seg[1:])
        else:
            out.extend(seg) #if not, add the entire segment to the path
    return out


def animate_path(panda_robot, path, hold_steps_per_vertex=400):
    """Hold at each waypoint then interpolate to the next (same control style as visualize example)."""
    period = 1.0 / SAMPLING_RATE
    counter_seconds = -1
    total_steps = 0

    for w in range(len(path) - 1):
        q0, q1 = np.asarray(path[w], float), np.asarray(path[w + 1], float)
        # Hold at each waypoint
        for _ in range(hold_steps_per_vertex):
            if total_steps % int(period) == 0: #print the time every second (or 1000 steps)
                counter_seconds += 1
                print("Passed time in simulation: {:>4} sec".format(counter_seconds))
            panda_robot.set_target_positions(path[w])
            p.stepSimulation() #perform the simulation step
            time.sleep(SAMPLING_RATE)
            total_steps += 1

        # Interpolate to the next waypoint
        interp_steps = max(80, int(distance_q(path[w], path[w + 1]) / 0.02))
        for alpha in np.linspace(0.0, 1.0, interp_steps + 1)[1:]: #interpolate between the current and next waypoint
            if total_steps % int(period) == 0:
                counter_seconds += 1
                print("Passed time in simulation: {:>4} sec".format(counter_seconds))
            q = ((1.0 - alpha) * q0 + alpha * q1).tolist()
            panda_robot.set_target_positions(q)
            p.stepSimulation()
            time.sleep(SAMPLING_RATE)
            total_steps += 1

    # Hold at the last waypoint
    for _ in range(hold_steps_per_vertex // 2):
        panda_robot.set_target_positions(path[-1]) #set the last waypoint for the robot
        p.stepSimulation()
        time.sleep(SAMPLING_RATE)

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

    # Setup Panda robot
    panda_robot = PandaRobot(include_gripper=INCLUDE_GRIPPER)
    lowers, uppers = get_arm_joint_limits(panda_robot.robot_id)
    obstacle_ids = [obstacle_id, r2d2_id]
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
    goals = sample_feasible_goals(panda_robot, lowers, uppers, obstacle_ids, NUM_GOALS)
    if len(goals) < NUM_GOALS:
        print(
            "Could only sample {} feasible goals (collision-free). "
            "Try moving the obstacles or increasing max tries.".format(len(goals))
        )

    # Plan a path for each goal with RRT*
    full_segments = []
    q_from = q_start
    for gi, q_g in enumerate(goals):
        print("Planning RRT* segment {} -> goal {} ...".format(gi, gi + 1))
        path_seg = rrt_star_segment(panda_robot, q_from, q_g, lowers, uppers, obstacle_ids)
        if path_seg is None:
            print("  Segment failed; skipping this goal.")
            continue
        full_segments.append(path_seg)
        q_from = q_g #update the start configuration for the next goal

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
    animate_path(panda_robot, path, hold_steps_per_vertex=100) #animate the path

    p.disconnect()
    print("Simulation end")


if __name__ == "__main__":
    main()
