import pybullet as p
import pybullet_data
import numpy as np
import time


from assets.ycb_objects import getURDFPath
from utils import camera, image
from utils.control import get_movej_trajectory
import pybullet_industrial as pi

class UR5PickEnviornment:
    def __init__(self, gui=True):
        # 0 load environment
        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(1.5,45,-45,[0,0,0])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._plane_id = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.8)

        # 1 load UR5 robot
        self.robot_id = p.loadURDF(
            "assets/franka_panda/panda.urdf", basePosition = [0, 0, 0], baseOrientation = p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

        # Set maximum joint velocity. Maximum joint velocity taken from:
        # https://s3-eu-central-1.amazonaws.com/franka-de-uploads/uploads/Datasheet-EN.pdf
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=0, maxJointVelocity=150 * (np.pi / 180))
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=1, maxJointVelocity=150 * (np.pi / 180))
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=2, maxJointVelocity=150 * (np.pi / 180))
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=3, maxJointVelocity=150 * (np.pi / 180))
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=4, maxJointVelocity=180 * (np.pi / 180))
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=5, maxJointVelocity=180 * (np.pi / 180))
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=6, maxJointVelocity=180 * (np.pi / 180))

        # Set DOF for supplied gripper
        self.dof = p.getNumJoints(self.robot_id) - 1
        self.joints = range(self.dof)


        # Robot home joint configuration (over tote 1)
        #11, additional 2 joints for gripper
        self.robot_home_joint_config = [
            0, 0, 0, -0.35*np.pi, 0, 0.5*np.pi, 0, 0, 0, 0, 0]
        self.robot_tote2_joint_config = [
            np.pi, 0, 0, -0.35*np.pi, 0, 0.5*np.pi, 0, 0, 0, 0, 0]
    
        # 2 load tote
        # 3D workspace for tote 1
        self._workspace1_bounds = np.array([
            [0.38, 0.62],  # 3x2 rows: x,y,z cols: min,max
            [-0.22, 0.22],
            [0.00, 0.5]
        ])
        # 3D workspace for tote 2
        self._workspace2_bounds = np.copy(self._workspace1_bounds)
        self._workspace2_bounds[0, :] = - self._workspace2_bounds[0, ::-1]        # Load totes and fix them to their position
        # Load totes and fix them to their position
        self._tote1_position = (
            self._workspace1_bounds[:, 0] + self._workspace1_bounds[:, 1]) / 2
        self._tote1_position[2] = 0.01
        self._tote1_body_id = p.loadURDF(
            "assets/tote/toteA_large.urdf", self._tote1_position, p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)

        self._tote2_position = (
            self._workspace2_bounds[:, 0] + self._workspace2_bounds[:, 1]) / 2
        self._tote2_position[2] = 0.01
        self._tote2_body_id = p.loadURDF(
            "assets/tote/toteA_large.urdf", self._tote2_position, p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)

        # 3. load gripper
        self.robot_end_effector_link_index = 11

        # Distance between tool tip and end-effector joint
        #0.04 + 0.0584 from urdf
        self._tool_tip_to_ee_joint = np.array([0, 0, -0.0984])

        # Set friction coefficients for gripper fingers
        for i in range(9, 11, 1):
            p.changeDynamics(self.robot_id, i, lateralFriction=1.0, spinningFriction=0.001,
                             rollingFriction=0.001, frictionAnchor=True)
        
        self.set_joints(self.robot_home_joint_config)

        # 4. load camera
        self.camera = camera.Camera(
            image_size=(128, 128),
            near=0.01,
            far=10.0,
            fov_w=80
        )
        
        # Camera at end effector
        ee_link_state = p.getLinkState(self.robot_id, self.robot_end_effector_link_index)
        camera_target_position = (self._workspace1_bounds[:, 0] + self._workspace1_bounds[:, 1]) / 2
        camera_target_position[2] = 0
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition = ee_link_state[0] + np.array([0, 0.1, 0]),
            cameraTargetPosition=camera_target_position,
            cameraUpVector = np.array([0, 1, 0])
        )
        # Camera at base
        base_link_state = p.getLinkState(self.robot_id, 0)
        self.base_view_matrix = p.computeViewMatrix(
            cameraEyePosition = base_link_state[0] + np.array([0, 0.05, 0]),
            cameraTargetPosition=camera_target_position,
            cameraUpVector = np.array([0, 1, 0])
        )

        # 5. prepare loading objects
        self.object_ids = list()
    

    def get_camera_target_distance(self):
        """
        Get the distance from camera to its target point in PyBullet environment.
        
        Args:
            env: Your UR5PickEnvironment instance
            
        Returns:
            distance: Euclidean distance from camera to target
            camera_info: Dictionary with camera parameters
        """
        ee_link_state = p.getLinkState(self.robot_id, self.robot_end_effector_link_index)
        
        camera_position = np.array(ee_link_state[0]) + np.array([0, 0.05, 0])
        
        camera_target = (self._workspace1_bounds[:, 0] + self._workspace1_bounds[:, 1]) / 2
        camera_target[2] = 0  # Floor level
        
        distance = np.linalg.norm(camera_position - camera_target)
        
        dx = camera_position[0] - camera_target[0]
        dy = camera_position[1] - camera_target[1]
        dz = camera_position[2] - camera_target[2]
        
        horizontal_distance = np.sqrt(dx**2 + dy**2)
        
        camera_info = {
            'camera_position': camera_position,
            'camera_target': camera_target,
            'total_distance': distance,
            'horizontal_distance': horizontal_distance,
            'vertical_distance': dz,
            'dx': dx,
            'dy': dy,
            'dz': dz
        }
        
        return distance, camera_info

    

    def get_position_and_velocity(self):
        """"""
        joint_states = p.getJointStates(self.robot_id, self.joints)
        joint_pos = [state[0] for state in joint_states]
        joint_vel = [state[1] for state in joint_states]
        return joint_pos, joint_vel
    
    def camera_to_world(self, cam_coords):
        pose = camera.cam_view2pose(self.view_matrix)
        world_coords = cam_coords @ pose[:3,:3].T + pose[:3,3]
        return world_coords
    
    def pixel_to_world(self, img_x, img_y, depth):
        """
        CV Coordinte Convension
        """
        intrinsics = self.camera.intrinsic_matrix
        fx = intrinsics[0,0]
        fy = intrinsics[1, 1]
        cx, cy = intrinsics[:2,2]
        x = (img_x - cx) * depth / fx
        y = (img_y - cy) * depth / fy
        cam_coords = np.array([x,y,depth], dtype=np.float32)
        world_coords = self.camera_to_world(cam_coords)
        return world_coords

    def image_pose_to_pick_pose(self, coord, angle, depth_obs, min_z=0.022):
        print(depth_obs)
        depth = depth_obs[coord[::-1]]
        world_coord = self.pixel_to_world(*coord, depth)
        world_coord[-1] = max(min_z, world_coord[-1]-0.05)
        world_angle = ((-angle-90)/180)*np.pi
        return world_coord, world_angle

    def ob_pos_to_ee_pos(self, obj_pos, obj_oreintation, obj_id):
        # In this assignment we assume the object pose is the grasp pose.  
        # we only covert the pose to a top down grasp angle.  
        world_coord =  obj_pos
        rot_r = p.getEulerFromQuaternion(obj_oreintation)
        world_angle = rot_r[0]
        return world_coord, world_angle

    def load_ycb_objects(self, name_list, seed=None):
        rs = np.random.RandomState(seed=seed)
        self.name_list = name_list
        for name in name_list:
            urdf_path = getURDFPath(name)
            position, orientation = self.get_random_pose(rs)
            obj_id = p.loadURDF(urdf_path, 
                position, p.getQuaternionFromEuler(orientation))
            self.object_ids.append(obj_id)

        self.step_simulation(1e3)

    def observe(self, image_id):
        base_rgb_obs, base_depth_obs, base_mask_obs = camera.make_obs(self.camera, self.base_view_matrix)
        ee_rgb_obs, ee_depth_obs, ee_mask_obs = camera.make_obs(self.camera, self.view_matrix)
        # save images
        image.write_rgb(ee_rgb_obs.astype(np.uint8), 'output_images/ee_rgb_{}.png'.format(image_id))
        image.write_depth(ee_depth_obs, 'output_images/ee_depth_{}.png'.format(image_id))
        image.write_mask(ee_mask_obs, 'output_images/ee_mask_{}.png'.format(image_id))
        image.write_rgb(base_rgb_obs.astype(np.uint8), 'output_images/base_rgb_{}.png'.format(image_id))
        image.write_depth(base_depth_obs, 'output_images/base_depth_{}.png'.format(image_id))
        image.write_mask(base_mask_obs, 'output_images/base_mask_{}.png'.format(image_id))
        # save npy files
        np.save('output_npy/ee_rgb_{}.npy'.format(image_id), ee_rgb_obs)
        np.save('output_npy/ee_depth_{}.npy'.format(image_id), ee_depth_obs)
        np.save('output_npy/ee_mask_{}.npy'.format(image_id), ee_mask_obs)
        np.save('output_npy/base_rgb_{}.npy'.format(image_id), base_rgb_obs)
        np.save('output_npy/base_depth_{}.npy'.format(image_id), base_depth_obs)
        np.save('output_npy/base_mask_{}.npy'.format(image_id), base_mask_obs)
        #We use these for vision
        return ee_rgb_obs, ee_depth_obs, ee_mask_obs
    
    
    def get_random_pose(self, rs):
        low = self._workspace1_bounds[:,0].copy()
        low[-1] += 0.2
        high = self._workspace1_bounds[:,1].copy()
        high[-1] += 0.2
        position = rs.uniform(low, high, size=3)
        orientation = rs.uniform(-np.pi, np.pi,size=3)
        return position, orientation

    def reset_objects(self, seed=None):
        rs = np.random.RandomState(seed=seed)
        for obj_id in self.object_ids:
            position, orientation = self.get_random_pose(rs)
            p.resetBasePositionAndOrientation(
                obj_id, position, p.getQuaternionFromEuler(orientation))
        self.step_simulation(1e3)
    
    def remove_objects(self):
        for obj_id in self.object_ids:
            p.removeBody(obj_id)
        self.object_ids = list()
    
    def get_object_pose(self, obj_id): 
        position, orientation = p.getBasePositionAndOrientation(obj_id)
        return position, orientation

    def set_joints(self, target_joint_state, steps=1e2):
        assert len(self.joints) == len(target_joint_state)
        for joint, value in zip(self.joints, target_joint_state):
            p.resetJointState(self.robot_id, joint, value)
        if steps > 0:
            self.step_simulation(steps)

    def num_object_in_tote1(self):
        num_in = 0
        low = self._workspace1_bounds[:,0].copy()
        low -= 0.2
        high = self._workspace1_bounds[:,1].copy()
        high += 0.2

        for object_id in self.object_ids:
            pos, _ = p.getBasePositionAndOrientation(object_id)
            pos = np.array(pos)
            is_in = (low < pos).all()
            is_in &= (pos < high).all()
            if is_in:
                num_in += 1
        return num_in

    def move_joints(self, target_joint_state, target_gripper_position_provided, target_gripper_position, acceleration=10, speed=3.0):
        """
            Move robot arm to specified joint configuration by appropriate motor control
        """
        dt = 1./240
        q_current = np.array([x[0] for x in p.getJointStates(self.robot_id, self.joints)])
        q_target = target_joint_state
        # Special case for gripper
        if target_gripper_position_provided:
            q_target[9] = target_gripper_position
            q_target[10] = target_gripper_position
        else:
            q_target[9] = q_current[9]
            q_target[10] = q_current[10]
        q_traj = get_movej_trajectory(q_current, q_target, 
            acceleration=acceleration, speed=speed)
        qdot_traj = np.gradient(q_traj, dt, axis=0)
        p_gain = 1 * np.ones(len(self.joints))
        d_gain = 1 * np.ones(len(self.joints))
        for i in range(len(q_traj)):
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id, 
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL, 
                targetPositions=q_traj[i],
                targetVelocities=qdot_traj[i],
                positionGains=p_gain,
                velocityGains=d_gain
            )
            self.step_simulation(1)

    def move_tool(self, position, orientation, target_gripper_position_provided, target_gripper_position, acceleration=10, speed=3.0):
        """
            Move robot tool (end-effector) to a specified pose
            @param position: Target position of the end-effector link
            @param orientation: Target orientation of the end-effector link
        """
        target_joint_state = np.array([x[0] for x in p.getJointStates(self.robot_id, self.joints)])
        # Calculate inverse kinematics for the target pose
        ik_result = p.calculateInverseKinematics(
            self.robot_id, 
            self.robot_end_effector_link_index, #end effector link index
            position, 
            orientation, 
            maxNumIterations=100
        )

        # Extract joint angles for the joints, should be length 9
        # Gripper values are not included in the ik_result
        for i in range(len(ik_result)):
            target_joint_state[i] = ik_result[i]
        
        print("target_gripper_position", position)
        self.move_joints(target_joint_state, target_gripper_position_provided, target_gripper_position, acceleration=acceleration, speed=speed)

    def robot_go_home(self, isOpen, speed=3.0):
        if isOpen:
            self.move_joints(self.robot_home_joint_config, True, 0.15, speed=speed)
        else:
            self.move_joints(self.robot_home_joint_config, False, 0.0, speed=speed)

    def open_gripper(self):
        #open joint 1
        p.setJointMotorControl2(
            self.robot_id, 9, p.POSITION_CONTROL, targetPosition=0.15, force=1000)
        #open joint 2
        p.setJointMotorControl2(
            self.robot_id, 10, p.POSITION_CONTROL, targetPosition=0.15, force=1000)
        self.step_simulation(6e2)

    def close_gripper(self):
        """
        Close the gripper completely with maximum force, regardless of object position
        """
        print("Starting forceful gripper close...")
        
        # Initial gripper state
        joint_state_9 = p.getJointState(self.robot_id, 9)[0]
        joint_state_10 = p.getJointState(self.robot_id, 10)[0]
        print(f"Initial gripper state - Joint 9: {joint_state_9}, Joint 10: {joint_state_10}")
        
        # Method 1: Direct position control with extremely high force
        # Set target position to fully closed (0.0)
        MAX_FORCE = 1000  # Use maximum allowable force
        p.setJointMotorControl2(self.robot_id, 9, p.POSITION_CONTROL, 
                            targetPosition=0.0, force=MAX_FORCE)
        p.setJointMotorControl2(self.robot_id, 10, p.POSITION_CONTROL, 
                            targetPosition=0.0, force=MAX_FORCE)
        
        # Step simulation for a longer period to ensure closing
        for i in range(3000):
            p.stepSimulation()
            
            # Periodically check and report gripper state
            if i % 50 == 0:
                joint_state_9 = p.getJointState(self.robot_id, 9)[0]
                joint_state_10 = p.getJointState(self.robot_id, 10)[0]
                print(f"Step {i} - Joint 9: {joint_state_9}, Joint 10: {joint_state_10}")
        
        # Method 2: Fallback to velocity control if position control didn't fully close
        # joint_state_9 = p.getJointState(self.robot_id, 9)[0]
        # if joint_state_9 > 0.01:  # If not fully closed
        #     print("Switching to velocity control for final closing...")
            
        #     # Use velocity control with high negative velocity and maximum force
        #     p.setJointMotorControl2(self.robot_id, 9, p.VELOCITY_CONTROL, 
        #                         targetVelocity=-1.0, force=MAX_FORCE)
        #     p.setJointMotorControl2(self.robot_id, 10, p.VELOCITY_CONTROL, 
        #                         targetVelocity=-1.0, force=MAX_FORCE)
            
        #     # Step simulation again
        #     for i in range(200):
        #         p.stepSimulation()
                
        #         # Periodically check and report gripper state
        #         if i % 50 == 0:
        #             joint_state_9 = p.getJointState(self.robot_id, 9)[0]
        #             joint_state_10 = p.getJointState(self.robot_id, 10)[0]
        #             print(f"Velocity control step {i} - Joint 9: {joint_state_9}, Joint 10: {joint_state_10}")
        
        # Final gripper state
        joint_state_9 = p.getJointState(self.robot_id, 9)[0]
        joint_state_10 = p.getJointState(self.robot_id, 10)[0]
        print(f"Final gripper state - Joint 9: {joint_state_9}, Joint 10: {joint_state_10}")

    def check_grasp_success(self):
        #just checks that joint 1 isn't stuck, can replace with a better metric if we figure it out
        return p.getJointState(self.robot_id, 1)[0] < 0.834 - 0.001

    def execute_grasp(self, grasp_position, grasp_angle):
        """
            Execute grasp sequence
            @param: grasp_position: 3d position of place where the gripper jaws will be closed
            @param: grasp_angle: angle of gripper before executing grasp from positive x axis in radians 
        """
        # Adjust grasp_position to account for end-effector length
        grasp_position = grasp_position + self._tool_tip_to_ee_joint
        print("grasp_position", grasp_position)
        gripper_orientation = p.getQuaternionFromEuler([np.pi, 0, grasp_angle])
        pre_grasp_position_over_bin = grasp_position+np.array([0, 0, 0.3])
        pre_grasp_position_over_object = grasp_position+np.array([0, 0, 0.10])
        post_grasp_position = grasp_position+np.array([0, 0, 0.3])
        grasp_success = False
        # ========= PART 2============
        # Implement the following grasp sequence:
        # 1. open gripper
        # 2. Move gripper to pre_grasp_position_over_bin
        # 3. Move gripper to grasp_position
        # 4. Close gripper
        # 5. Move gripper to post_grasp_position
        # 6. Move robot to robot_home_joint_config
        # 7. Detect whether or not the object was grasped and return grasp_success
        # ============================
        # 1. open gripper
        self.open_gripper()
        # 2. Move gripper to pre_grasp_position_over_bin
        self.move_tool(pre_grasp_position_over_bin, gripper_orientation, True, 0.1)
        # 3. Move gripper to pre_grasp_position_over_object
        self.move_tool(pre_grasp_position_over_object, gripper_orientation, True, 0.1)
        # 4. Move gripper to grasp_position
        self.move_tool(pre_grasp_position_over_object, gripper_orientation, True, 0.1)
        ee_link_state = p.getLinkState(self.robot_id, self.robot_end_effector_link_index)
        print("ee_link_state", ee_link_state[0])
        # 5. Close gripper
        self.close_gripper()
        # 6. Move gripper to post_grasp_position
        self.move_tool(post_grasp_position, gripper_orientation, False, 0.0)
        # 7. Move robot to robot_home_joint_config
        self.robot_go_home(False, speed=3.0)
        # 8. Detect whether or not the object was grasped and return grasp_success
        grasp_success = self.check_grasp_success()

        return grasp_success

    def execute_place(self):
        print("execute place")
        self.move_joints(self.robot_tote2_joint_config, False, 0.0, speed=6.0)
        self.open_gripper()
        self.robot_go_home(True, speed=6.0)
        
    def step_simulation(self, num_steps):
        for i in range(int(num_steps)):
            p.stepSimulation()
            if self.robot_id is not None:
                # Constraints for Panda gripper finger synchronization
                joint_positions = np.array([x[0] for x in p.getJointStates(self.robot_id, self.joints)])
                p.setJointMotorControlArray(
                    self.robot_id, [9, 10], p.POSITION_CONTROL,
                    [
                        joint_positions[9],
                        joint_positions[9]
                    ],
                    positionGains=np.ones(2)
                )
        #update camera position and orientation
        ee_link_state = p.getLinkState(self.robot_id, self.robot_end_effector_link_index)
        camera_target_position = (self._workspace1_bounds[:, 0] + self._workspace1_bounds[:, 1]) / 2
        camera_target_position[2] = 0
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition = ee_link_state[0] + np.array([0, 0.05, 0]),
            cameraTargetPosition=camera_target_position,
            cameraUpVector = np.array([0, 1, 0])
        )
        # Camera at base
        base_link_state = p.getLinkState(self.robot_id, 0)
        self.base_view_matrix = p.computeViewMatrix(
            cameraEyePosition = base_link_state[0] + np.array([0, 0.05, 0]),
            cameraTargetPosition=camera_target_position,
            cameraUpVector = np.array([0, 1, 0])
        )

    

