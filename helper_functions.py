"""
Helper functions for the Franka Emika Panda 7-DOF robot.
"""

import csv
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

# ========== ROBOT FUNCTIONS ==========

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

def fk_ee_position(panda_robot, q_arm_7):
    """Get the end effector position and orientation given the arm joint angles"""
    set_arm_configuration(panda_robot, q_arm_7) 
    ee_link = panda_robot.dof  # matches IK in PandaRobot
    state = p.getLinkState(panda_robot.robot_id, ee_link)
    return state

def distance_q(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

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


# ========== RRT* FUNCTIONS ==========

def execute_rrt_star(panda_robot, q_start, goals, lowers, uppers, obstacle_ids, stats_csv_path=None):
    """
    Execute the RRT* algorithm.

    If ``stats_csv_path`` is set, append one CSV row per planning attempt with columns:
    ``segment_index``, ``success``, ``iterations``, ``accepted_expansions``,
    ``num_tree_nodes``, ``cumulative_path_cost`` (joint-space path length; empty if failed).
    """
    full_segments = []
    csv_rows = []
    q_from = q_start
    for gi, q_g in enumerate(goals):
        print("Planning RRT* segment {} -> goal {} ...".format(gi, gi + 1))
        seg_stats = {} if stats_csv_path is not None else None
        path_seg = rrt_star_segment(
            panda_robot, q_from, q_g, lowers, uppers, obstacle_ids, stats_out=seg_stats
        )
        if stats_csv_path is not None:
            csv_rows.append({"segment_index": gi, **seg_stats})
        if path_seg is None:
            print("  Segment failed; skipping this goal.")
            continue
        full_segments.append(path_seg)
        q_from = q_g #update the start configuration for the next goal

    if stats_csv_path is not None and csv_rows:
        fieldnames = [
            "segment_index",
            "success",
            "iterations",
            "accepted_expansions",
            "num_tree_nodes",
            "cumulative_path_cost",
        ]
        with open(stats_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in csv_rows:
                out = dict(row)
                if out.get("cumulative_path_cost") is None:
                    out["cumulative_path_cost"] = ""
                else:
                    out["cumulative_path_cost"] = "{:.8f}".format(out["cumulative_path_cost"])
                out["success"] = "1" if out.get("success") else "0"
                writer.writerow(out)

    return full_segments

def edge_feasible(panda_robot, q_from, q_to, obstacle_ids, steps=EDGE_DISCRETIZATION_STEPS):
    q0, q1 = np.asarray(q_from, dtype=float), np.asarray(q_to, dtype=float)
    for alpha in np.linspace(0.0, 1.0, steps + 1):
        q = (1.0 - alpha) * q0 + alpha * q1 #interpolate between q_from and q_to
        if not configuration_feasible(panda_robot, q.tolist(), obstacle_ids): #check if the interpolated configuration is feasible
            return False
    return True

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


def _fill_rrt_segment_stats(stats_out, success, iterations, nodes, goal_idx, accepted_expansions):
    """Populate ``stats_out`` in place (caller passes a dict)."""
    cost = nodes[goal_idx]["cost"] if success and goal_idx is not None else None
    stats_out["success"] = bool(success)
    stats_out["iterations"] = int(iterations)
    stats_out["accepted_expansions"] = int(accepted_expansions)
    stats_out["num_tree_nodes"] = int(len(nodes))
    stats_out["cumulative_path_cost"] = float(cost) if cost is not None else None


def rrt_star_segment(
    panda_robot, q_start, q_goal, lowers, uppers, obstacle_ids, stats_out=None
):
    """
    Plan from q_start to (near) q_goal. Returns list of configurations from start
    to goal along the tree, or None.

    If ``stats_out`` is a dict, it is filled with:
    ``success``, ``iterations`` (main loop count), ``accepted_expansions`` (new tree
    vertices added in the loop), ``num_tree_nodes`` (total nodes including start and
    goal if any), ``cumulative_path_cost`` (sum of Euclidean joint-space edge lengths
    along the returned path, or None if planning failed).
    """
    def sample_uniform():
        return [random.uniform(lo, hi) for lo, hi in zip(lowers, uppers)]

    nodes = [{"q": list(q_start), "parent": None, "cost": 0.0}]
    accepted_expansions = 0

    for iteration in range(1, MAX_ITER_PER_SEGMENT + 1):
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
        accepted_expansions += 1

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
                if stats_out is not None:
                    _fill_rrt_segment_stats(
                        stats_out, True, iteration, nodes, goal_idx, accepted_expansions
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
            if stats_out is not None:
                _fill_rrt_segment_stats(
                    stats_out,
                    True,
                    MAX_ITER_PER_SEGMENT,
                    nodes,
                    goal_idx,
                    accepted_expansions,
                )
            return extract_path(nodes, goal_idx)
    if stats_out is not None:
        _fill_rrt_segment_stats(
            stats_out, False, MAX_ITER_PER_SEGMENT, nodes, None, accepted_expansions
        )
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



# ========== ANIMATION FUNCTIONS ==========

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

def animate_path(
    panda_robot,
    path,
    hold_steps_per_vertex=400,
    control_mode="position",
    use_gravity_compensation=None,
    kp=200.0,
    kd=30.0,
    max_torque=87.0,
    print_tracking_error=True,
):
    """
    Hold at each waypoint then interpolate to the next.

    control_mode:
      - "position": PyBullet POSITION_CONTROL
      - "torque_pd": tau = tau_g(q) + Kp * e + Kd * e_dot
      - "computed_torque": tau = ID(q, qd, qdd_des + Kd*e_dot + Kp*e)

    use_gravity_compensation is kept for backward compatibility:
      True -> "torque_pd", False -> "position"
    """
    print("Animating path with control mode: ", control_mode)
    if use_gravity_compensation is not None:
        control_mode = "torque_pd" if use_gravity_compensation else "position"
    if control_mode not in ("position", "torque_pd", "computed_torque"):
        raise ValueError("control_mode must be one of: position, torque_pd, computed_torque")

    period = 1.0 / SAMPLING_RATE
    counter_seconds = -1
    total_steps = 0
    err_accum = 0.0
    err_count = 0

    if control_mode in ("torque_pd", "computed_torque"):
        # Explicitly disable default motors before applying torques.
        p.setJointMotorControlArray(
            bodyUniqueId=panda_robot.robot_id,
            jointIndices=panda_robot.joints,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0 for _ in panda_robot.joints],
        )

    def apply_control(q_des, qd_des, qdd_des):
        nonlocal err_accum, err_count
        q, qd = panda_robot.get_position_and_velocity()
        q_arm, qd_arm = q[:7], qd[:7]
        e = [q_des[i] - q_arm[i] for i in range(7)]
        e_dot = [qd_des[i] - qd_arm[i] for i in range(7)]
        err_accum += float(np.linalg.norm(np.asarray(e)))
        err_count += 1

        if control_mode == "position":
            panda_robot.set_target_positions(q_des)
            return

        if control_mode == "torque_pd":
            tau_g = panda_robot.calculate_inverse_dynamics(q_arm, [0.0] * 7, [0.0] * 7)
            tau_pd = [kp * e[i] + kd * e_dot[i] for i in range(7)]
            tau = [tau_g[i] + tau_pd[i] for i in range(7)]
        else:
            qdd_cmd = [qdd_des[i] + kd * e_dot[i] + kp * e[i] for i in range(7)]
            tau = panda_robot.calculate_inverse_dynamics(q_arm, qd_arm, qdd_cmd)

        tau = [max(-max_torque, min(max_torque, t)) for t in tau]
        if panda_robot.dof == 9:
            tau = tau + [0.0, 0.0]
        panda_robot.set_torques(tau)

    for w in range(len(path) - 1):
        q0, q1 = np.asarray(path[w], float), np.asarray(path[w + 1], float)
        # Hold at each waypoint
        for _ in range(hold_steps_per_vertex):
            if total_steps % int(period) == 0: #print the time every second (or 1000 steps)
                counter_seconds += 1
                print("Passed time in simulation: {:>4} sec".format(counter_seconds))
                if control_mode in ("torque_pd", "computed_torque") and print_tracking_error and err_count > 0:
                    print("Mean joint tracking error (L2): {:.5f} rad".format(err_accum / err_count))
                    err_accum = 0.0
                    err_count = 0
            apply_control(path[w], [0.0] * 7, [0.0] * 7)
            p.stepSimulation() #perform the simulation step
            time.sleep(SAMPLING_RATE)
            total_steps += 1

        # Interpolate to the next waypoint
        interp_steps = max(80, int(distance_q(path[w], path[w + 1]) / 0.02))
        segment_time = interp_steps * SAMPLING_RATE
        qd_segment = ((q1 - q0) / max(segment_time, 1e-9)).tolist()
        for alpha in np.linspace(0.0, 1.0, interp_steps + 1)[1:]: #interpolate between the current and next waypoint
            if total_steps % int(period) == 0:
                counter_seconds += 1
                print("Passed time in simulation: {:>4} sec".format(counter_seconds))
                if control_mode in ("torque_pd", "computed_torque") and print_tracking_error and err_count > 0:
                    print("Mean joint tracking error (L2): {:.5f} rad".format(err_accum / err_count))
                    err_accum = 0.0
                    err_count = 0
            q = ((1.0 - alpha) * q0 + alpha * q1).tolist()
            apply_control(q, qd_segment, [0.0] * 7)
            p.stepSimulation()
            time.sleep(SAMPLING_RATE)
            total_steps += 1

    # Hold at the last waypoint
    for _ in range(hold_steps_per_vertex // 2):
        apply_control(path[-1], [0.0] * 7, [0.0] * 7) #set the last waypoint for the robot
        p.stepSimulation()
        time.sleep(SAMPLING_RATE)

    mean_tracking_error = (err_accum / err_count) if err_count > 0 else float("nan")
    if print_tracking_error:
        print(
            "Final mean joint tracking error for {}: {:.5f} rad".format(
                control_mode, mean_tracking_error
            )
        )
    return {
        "control_mode": control_mode,
        "mean_joint_tracking_error_l2_rad": mean_tracking_error,
        "samples": err_count,
    }


def benchmark_control_modes(
    panda_robot,
    path,
    hold_steps_per_vertex=200,
    modes=("position", "torque_pd", "computed_torque"),
    kp=200.0,
    kd=30.0,
    max_torque=87.0,
):
    """Run multiple control modes on the same path and print a compact error table."""
    if not path:
        return []

    results = []
    print("\n=== Control-mode benchmark ===")
    for mode in modes:
        set_arm_configuration(panda_robot, path[0])
        panda_robot.set_target_positions(path[0])
        for _ in range(40):
            p.stepSimulation()
            time.sleep(SAMPLING_RATE)

        print("\nRunning mode: {}".format(mode))
        metrics = animate_path(
            panda_robot,
            path,
            hold_steps_per_vertex=hold_steps_per_vertex,
            control_mode=mode,
            kp=kp,
            kd=kd,
            max_torque=max_torque,
            print_tracking_error=False,
        )
        results.append(metrics)

    print("\nMode                Mean joint error [rad]   Samples")
    print("----------------------------------------------------")
    for r in results:
        print(
            "{:<19} {:>10.5f}            {:>7}".format(
                r["control_mode"], r["mean_joint_tracking_error_l2_rad"], r["samples"]
            )
        )
    return results