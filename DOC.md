Python files:
- **example_rrt_star_simple.py** - working RRT* example
- **helper_functions.py** - all helper functions
- **example_baseline.py** - main function to execute baseline

PyBullet functions:
- **getClosestPoints(robot id, obstacle, distance)**
- **resetJointState(robot id, joint id, angle)**

Panda Robot functions:
- set_target_positions(joint angle)

Custom functions:
- **animate_path** - animate the path segment
- **configuration_feasible** - is configuration feasible? i.e. Not in collision
    - used to check for valid start configuration as well as goals and edges
- **distance_q** - Euclidean distance between two sets of joint angles
    - used in nearest_index to get nearest node in tree to new sampled node configuration
- **extract_path** - returns the path from nodes, starting at the goal
- **get_arm_joint_limits** - get the arm's lower/upper limits
- **merge_paths** - merges all path segments from RRT* (each one going to a separate goal)
- **near_indices** - given a radius from sampled node, find all nodes within the radius
- **nearest_index** - finds nearest configuration to sampled configuration
- **sample_feasible_goals** - samples random goals which are the goals in RRT*
- **set_arm_configuration**
- **steer** - steer from closest node to the new sampled node (straight line)

Workflow (RRT*)
1) Initialize GUI, robot, and obstacles. Includes calculating joint limits of robot
    - get_arm_joint_limits
2) Check if initial pose is feasible (collision-free)
    - configuration_feasible
3) Sample feasible goals. Try to sample 5 goals if possible
    - sample_feasible_goals
4) Perform RRT* for each goal; then combine each path segment
5) Animate the path segments