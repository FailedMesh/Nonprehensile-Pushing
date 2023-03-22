import time
import pybullet as p
import pybullet_data
import numpy as np
import cameras as cameras
#from constants import WORKSPACE_LIMITS
import os
import copy
# from time import time

import pybullet_utils.bullet_client as bc

class Environment:
    
    def __init__(self, ur5_pos = [0, 0], gui=True, use_gripper = True, time_step=1 / 480. ):

        """Creates environment with PyBullet.

        Args:
        gui: show environment with PyBullet's built-in display viewer
        time_step: PyBullet physics simulation step speed. Default is 1 / 240.
        """

        self.time_step = time_step
        self.gui = gui
        self.obj_ids = {"fixed": [], "rigid": []}
        self.agent_cams = cameras.RealSenseD435.CONFIG
        self.oracle_cams = cameras.Oracle.CONFIG
        #self.bounds = WORKSPACE_LIMITS
        
        # Joint angles for home position (How are these decided???):
        self.home_joints = np.array([0, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi
        
        # Joint angles for IK rest position (How are these decided???):
        self.ik_rest_joints = np.array([0, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        
        # These two are only used during grasp:
        self.drop_joints0 = np.array([0.5, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi
        self.drop_joints1 = np.array([1, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi

        # self.pixel_size = PIXEL_SIZE

        # Start PyBullet.
        self.client_id = bc.BulletClient(p.GUI if gui else p.DIRECT) # p.connect(p.GUI if gui else p.DIRECT) # To spawn multiple instances
        self.client_id.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.client_id.setTimeStep(time_step)
        self.client_id.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        
        # Set camera to correct position and orientation:
        if gui:
            target = self.client_id.getDebugVisualizerCamera()[11]
            self.client_id.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=90,
                cameraPitch=-25,
                cameraTargetPosition=target,
            )

        # Environment Objects:
        self.ur5e = None
        self.table = None
        self.table_height = 0.625
        self.ur5_pos = ur5_pos[:]

        self.use_gripper = use_gripper
        self.client_id.resetSimulation()                # Reset simulation
        self.client_id.setGravity(0, 0, -9.8)           # Set Gravity

        # Temporarily disable rendering to load scene faster.
        self.client_id.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        # ------------- SETUP WORKSPACE ---------------- #

        # Load plane:
        self.create_plane()
        self.client_id.changeVisualShape(self.plane, -1, rgbaColor = np.array([1, 0.7, 0, 0.2]))
        
        # Load table:
        self.create_table()

        # Load workspace:
        #self.create_plane(urdf = "Assets/workspace/workspace.urdf", basePosition = (0.5, 0.0, self.table_height + 0.002))
        
        # Load ur5:
        success = self.create_ur5()

        # ---------------------------------------------- #

        if not success:
            print("Simulation is wrong!")
            exit()

        # Re-enable rendering.
        self.client_id.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        #print(self.obj_ids["rigid"])
        v = [np.linalg.norm(self.client_id.getBaseVelocity(i)[0]) for i in self.obj_ids["rigid"]]
        return all(np.array(v) < 5e-3)

    @property
    # Change this function to a new info function:
    def info(self):
        """Environment info variable with object poses, dimensions, and colors."""

        info = {}  # object id : (position, rotation, dimensions)
        for obj_ids in self.obj_ids.values():
            for obj_id in obj_ids:
                pos, rot = self.client_id.getBasePositionAndOrientation(obj_id)
                dim = self.client_id.getVisualShapeData(obj_id)[0][3]
                info[obj_id] = (pos, rot, dim)
        return info

    # Function used only in a not used function ???
    def add_object_id(self, obj_id, category="rigid"):
        """List of (fixed, rigid) objects in env."""
        self.obj_ids[category].append(obj_id)

    def remove_object_id(self, obj_id, category="rigid"):
        """List of (fixed, rigid) objects in env."""
        self.obj_ids[category].remove(obj_id)

    def wait_static(self, timeout=5):
        
        """Step simulator asynchronously until objects settle."""
        self.client_id.stepSimulation()
        self.client_id.stepSimulation()
        t0 = time.time()
        while (time.time() - t0) < timeout:
            if self.is_static:
                return True
            self.client_id.stepSimulation()
            self.client_id.stepSimulation()
        print(f"Warning: move_joints exceeded {timeout} second timeout. Skipping.")
        return False

    def create_cuboid(self, obj_size, obj_pos, obj_orientation, collision = True):
        
        color = np.array([237, 201, 72, 255])/255
        
        vuid = self.client_id.createVisualShape(p.GEOM_BOX, 
                                                    halfExtents = obj_size/2, 
                                                    rgbaColor=color)
        
        if collision:
        
            cuid = self.client_id.createCollisionShape(p.GEOM_BOX, halfExtents = obj_size/2)
            
            obj_id = self.client_id.createMultiBody(baseMass=0.1, 
                                                            baseCollisionShapeIndex = cuid, 
                                                            baseVisualShapeIndex = vuid, 
                                                            basePosition = obj_pos, 
                                                            baseOrientation = obj_orientation)
        
        else:

            obj_id = self.client_id.createMultiBody(baseMass=0.1, 
                                                            baseVisualShapeIndex = vuid, 
                                                            basePosition = obj_pos, 
                                                            baseOrientation = obj_orientation)
        
        self.add_object_id(obj_id)
        
        return obj_id
    
    def create_plane(self, urdf = "plane.urdf", basePosition=(0, 0, -0.0005)):

        # Load plane from the file system of physics server:
        self.plane = self.client_id.loadURDF(urdf, basePosition = basePosition, useFixedBase=True)

        # Change physical dynamics of the plane:
        self.client_id.changeDynamics(
            self.plane,
            -1,
            lateralFriction=1.1,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )

    def create_table(self):
        
        orientation = self.client_id.getQuaternionFromEuler((0, 0, np.pi/2))

        self.table = self.client_id.loadURDF("table/table.urdf", 
                                        basePosition=(0.4, 0, 0), 
                                        baseOrientation = orientation, 
                                        useFixedBase=True)
        
        # Change physical dynamics of the table:
        self.client_id.changeDynamics(
            self.table,
            -1,
            lateralFriction=1.1,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )
    
    def create_ur5(self):

        # Load UR5e
        self.ur5e = self.client_id.loadURDF("Assets/ur5e/ur5e.urdf", 
                                            basePosition = (self.ur5_pos[0], self.ur5_pos[1], self.table_height), 
                                            useFixedBase=True)
        
        # Loop through all UR5 joints to get its revolute joints and end effector joints:
        self.ur5e_joints = []           # This variable will contain all the revolute joints
        for i in range(self.client_id.getNumJoints(self.ur5e)):
            info = self.client_id.getJointInfo(self.ur5e, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_name == "ee_fixed_joint":
                self.ur5e_ee_id = joint_id                      # Find the end effector fixed joint
            if joint_type == self.client_id.JOINT_REVOLUTE:
                self.ur5e_joints.append(joint_id)               # Store all the revolute joints

        # Enable sensing of the 6-dof joints + torques on the end effector fixed joint:
        self.client_id.enableJointForceTorqueSensor(self.ur5e, self.ur5e_ee_id, 1)

        self.setup_gripper()

        # Move robot to home joint configuration (this configuration is defined in init).
        success = self.go_home()[0]

        self.close_gripper()
        #self.open_gripper()

        return success

    def reset(self, use_gripper=True):
        
        for i in self.obj_ids["rigid"]:
            self.client_id.removeBody(i)
            self.remove_object_id(i)
        
        self.obj_ids = {"fixed": [], "rigid": []}
        self.use_gripper = use_gripper
        #self.client_id.resetSimulation()                # Reset simulation
        self.client_id.setGravity(0, 0, -9.8)           # Set Gravity

        self.go_home()

    def setup_gripper(self):
        
        """Load end-effector: gripper"""

        # Extract just the position of the end effector in the world frame:
        ee_position, _ = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        
        # Load in the robotiq gripper at the end effector position with an angle of -90 around the X-axis:
        self.ee = self.client_id.loadURDF(
            "Assets/ur5e/gripper/robotiq_2f_85.urdf",
            ee_position,
            self.client_id.getQuaternionFromEuler((-np.pi / 2, 0, 0)),)
        
        self.ee_tip_offset = 0.15
        self.gripper_angle_open = 0.03
        self.gripper_angle_close = 0.8
        self.gripper_angle_close_threshold = 0.7
        self.gripper_mimic_joints = {
            "left_inner_finger_joint": -1,
            "left_inner_knuckle_joint": -1,
            "right_outer_knuckle_joint": -1,
            "right_inner_finger_joint": -1,
            "right_inner_knuckle_joint": -1,
        }

        # Loop through all joints to get their info:
        for i in range(self.client_id.getNumJoints(self.ee)):
            
            info = self.client_id.getJointInfo(self.ee, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            
            if joint_name == "finger_joint":
                self.gripper_main_joint = joint_id              # Get the gripper's joint

            elif joint_name == "dummy_center_fixed_joint":
                self.ee_tip_id = joint_id                       # Get end effector's dummy fixed tip joint

            elif (joint_name == "left_inner_finger_pad_joint"
                or joint_name == "right_inner_finger_pad_joint"):
                self.client_id.changeDynamics(self.ee, joint_id, lateralFriction=1)     # If there are padding joints, add friction to the end effector
            
            elif joint_type == p.JOINT_REVOLUTE:
                self.gripper_mimic_joints[joint_name] = joint_id
                # Make every revolute joint static since we are just setting up gripper:
                self.client_id.setJointMotorControl2(
                    self.ee, joint_id, self.client_id.VELOCITY_CONTROL, targetVelocity=0, force=0
                )


        self.ee_constraint = self.client_id.createConstraint(
            parentBodyUniqueId=self.ur5e,
            parentLinkIndex=self.ur5e_ee_id,
            childBodyUniqueId=self.ee,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 1),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.02),
            childFrameOrientation=p.getQuaternionFromEuler((0, -np.pi / 2, 0)),
        )
        
        # Enable sensing of the 6-dof joints + torques on the end effector fixed joint:
        self.client_id.enableJointForceTorqueSensor(self.ee, self.gripper_main_joint, 1)

        # Set up mimic joints in robotiq gripper: left
        c = self.client_id.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["left_inner_finger_joint"],
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self.client_id.changeConstraint(c, gearRatio=1, erp=0.5, maxForce=800)
        c = self.client_id.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["left_inner_knuckle_joint"],
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self.client_id.changeConstraint(c, gearRatio=-1, erp=0.5, maxForce=800)
        # Set up mimic joints in robotiq gripper: right
        c = self.client_id.createConstraint(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            self.ee,
            self.gripper_mimic_joints["right_inner_finger_joint"],
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self.client_id.changeConstraint(c, gearRatio=1, erp=0.5, maxForce=800)
        c = self.client_id.createConstraint(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            self.ee,
            self.gripper_mimic_joints["right_inner_knuckle_joint"],
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self.client_id.changeConstraint(c, gearRatio=-1, erp=0.5, maxForce=800)
        # Set up mimic joints in robotiq gripper: connect left and right
        c = self.client_id.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self.client_id.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=100)

    def render_camera(self, config):
        """Render RGB-D image with specified camera configuration."""

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config["rotation"])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config["position"] + lookdir
        focal_len = config["intrinsics"][0, 0]
        znear, zfar = config["zrange"]
        viewm = p.computeViewMatrix(config["position"], lookat, updir)
        fovh = (config["image_size"][0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = config["image_size"][1] / config["image_size"][0]
        projm = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = self.client_id.getCameraImage(
            width=config["image_size"][1],
            height=config["image_size"][0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=1,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        # Get color image.
        color_image_size = (config["image_size"][0], config["image_size"][1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if config["noise"]:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, config["image_size"]))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (config["image_size"][0], config["image_size"][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = zfar + znear - (2.0 * zbuffer - 1.0) * (zfar - znear)
        depth = (2.0 * znear * zfar) / depth
        if config["noise"]:
            depth += self._random.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)

        return color, depth, segm

    def __del__(self):
        self.client_id.disconnect()

    def get_link_pose(self, body, link):
        result = self.client_id.getLinkState(body, link)
        # Return only the worldLinkFramePosition and worldLinkFrameOrientation:
        return result[4], result[5]

    # ---------------------------------------------------------------------------
    # Robot Movement Functions
    # ---------------------------------------------------------------------------

    def go_home(self):
        return self.move_joints(self.home_joints)

    def move_joints(self, target_joints, speed=0.01, timeout=3):
        
        """Move UR5e to target joint configuration."""
        t0 = time.time()
        
        all_joints = []

        # Loop that moves joints in every step until it reaches:
        while (time.time() - t0) < timeout:

            # Get current joint position of each joint in the joint list defined in reset()
            current_joints = np.array([self.client_id.getJointState(self.ur5e, i)[0] for i in self.ur5e_joints])
            
            # Extract just the position of the end effector tip in the world frame:
            pos, _ = self.get_link_pose(self.ee, self.ee_tip_id)
            
            
            if pos[2] < 0.005:
                print(f"Warning: move_joints tip height is {pos[2]}. Skipping.")
                return False, np.array(all_joints)
            
            diff_joints = target_joints - current_joints    # Calculate the difference between target joint positions and current joint positions
            
            # If the difference between current joints and target joints is less than a threshold, end movement.
            # Only in this case is the movement sucessful.
            if all(np.abs(diff_joints) < 1e-2):
                # give 10 steps of time to stop
                for _ in range(10):
                    self.client_id.stepSimulation()
                return True, np.array(all_joints)

            # --------------------------- #
            # Move if tip height is more than 0.005 (off the ground ???) and distance to target joints is more than  0.002
            # --------------------------- #

            # Find the norm of joint movement:
            norm = np.linalg.norm(diff_joints)

            # Find the unit vector direction of movement for each joint w.r.t total norm
            v = diff_joints / norm if norm > 0 else 0
            
            # Calculate the fraction of total speed used and add it to current joints, to get the next position
            # of joints after one simulation step
            step_joints = current_joints + v * speed

            # Append this to all_joints
            all_joints.append(step_joints)
            self.client_id.setJointMotorControlArray(
                bodyIndex=self.ur5e,
                jointIndices=self.ur5e_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=step_joints,
                positionGains=np.ones(len(self.ur5e_joints)),
            )

            # Step the simulation
            self.client_id.stepSimulation()

        # If the while loop runs out of time then joints could not be brought to target within the given time
        print(f"Warning: move_joints exceeded {timeout} second timeout (Object is probably far away). Skipping.")
        
        return False, np.array(all_joints)

    def move_ee_pose(self, pose, speed=0.01):
        """Move UR5e to target end effector pose."""
        target_joints = self.solve_ik(pose)
        return self.move_joints(target_joints, speed)

    def solve_ik(self, pose):
        # st_time = time.time()

        """Calculate joint configuration with inverse kinematics."""
        joints = self.client_id.calculateInverseKinematics(
            bodyUniqueId=self.ur5e,
            endEffectorLinkIndex=self.ur5e_ee_id,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=[-6.283, -6.283, -3.141, -6.283, -6.283, -6.283],
            upperLimits=[6.283, 6.283, 3.141, 6.283, 6.283, 6.283],
            jointRanges=[12.566, 12.566, 6.282, 12.566, 12.566, 12.566],
            restPoses=np.float32(self.ik_rest_joints).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        joints = np.array(joints, dtype=np.float32)
        # joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        # print(f"IK time: {time.time()-st_time}")
        return joints

    def straight_move(self, pose0, pose1, rot, speed=0.003, max_force=300, detect_force=False):
        
        """Move every 1 cm, keep the move in a straight line instead of a curve. Keep level with rot"""
        step_distance = 0.01  # every 1 cm
        vec = np.float32(pose1) - np.float32(pose0)
        length = np.linalg.norm(vec)
        vec = vec / length
        n_push = np.int32(np.floor(np.linalg.norm(pose1 - pose0) / step_distance))  # every 1 cm
        success = True
        success1 = True

        all_joints = []
        for n in range(n_push):

            cur_joints = []
            target = pose0 + vec * n * step_distance
            success1, cur_joints = self.move_ee_pose((target, rot), speed)
            success &= success1
            if len(cur_joints)==0:
                pass
            elif len(all_joints)==0:
                all_joints = copy.deepcopy(cur_joints)
            else:
                all_joints = np.vstack((all_joints, cur_joints))
            if detect_force:
                force = np.sum(np.abs(np.array(self.client_id.getJointState(self.ur5e, self.ur5e_ee_id)[2])))
                if force > max_force:
                    target = target - vec * 2 * step_distance
                    success1, cur_joints = self.move_ee_pose((target, rot), speed)
                    success &= success1
                    if len(cur_joints)==0:
                        pass
                    elif len(all_joints)==0:
                        all_joints = copy.deepcopy(cur_joints)
                    else:
                        all_joints = np.vstack((all_joints, cur_joints))
                    print(f"Force is {force}, exceed the max force {max_force}")
                    return False, all_joints
        success1, cur_joints = self.move_ee_pose((pose1, rot), speed)

        success &= success1
        if len(cur_joints) == 0:
            pass
        elif len(all_joints) == 0:
            all_joints = copy.deepcopy(cur_joints)
        else:
            all_joints = np.vstack((all_joints, cur_joints))
        return success, all_joints

    def push(self, pose0, pose1, speed=0.0002):
        """Execute pushing primitive.

        Args:
            pose0: SE(3) starting pose.
            pose1: SE(3) ending pose.
            speed: the speed of the planar push.

        Returns:
            success: robot movement success if True.
        """

        # Adjust push start and end positions.
        pos0 = np.array(pose0, dtype=np.float32)
        pos1 = np.array(pose1, dtype=np.float32)
        pos0[2] += self.ee_tip_offset
        pos1[2] += self.ee_tip_offset
        vec = pos1 - pos0
        length = np.linalg.norm(vec)
        if length <= 0.005:
            length = 0.005
        vec = vec / length

        # Align against push direction.
        theta = np.arctan2(vec[1], vec[0])
        rot = p.getQuaternionFromEuler([np.pi / 2, np.pi / 2, theta])

        over0 = (pos0[0], pos0[1], pos0[2] + 0.2)
        over1 = (pos1[0], pos1[1], pos1[2] + 0.2)

        st_time = time.time()
        
        # 1) Move to IK rest position and get joint positions: ???
        success, all_joints = self.move_joints(self.ik_rest_joints)
        #_ = input("At home")

        success1 = True
        cur_joints = []

        # 2) Move to 0.2 metres above push start position, only if moving to IK rest position was successful:
        if success:
            st_time = time.time()
            success1, cur_joints = self.move_ee_pose((over0, rot))
            # Stack new joint positions on previous joint positions:
            if len(cur_joints)>0:
                all_joints = np.vstack((all_joints, cur_joints))
            success &= success1

        # 3) Perform a straight vertical move downwards by 0.2 metres at the push position
        if success:
            st_time = time.time()
            success1, cur_joints = self.straight_move(over0, pos0, rot, detect_force=True)
            if len(cur_joints)>0:
                all_joints = np.vstack((all_joints, cur_joints))
            success &= success1

        #_ = input("Ready to push")
  
        # 4) Perform the push from push start position to push end position
        if success:
            st_time = time.time()
            success1, cur_joints = self.straight_move(pos0, pos1, rot, speed, detect_force=True)
            if len(cur_joints)>0:
                all_joints = np.vstack((all_joints, cur_joints))
            success &= success1

        # 5) Perform a straight vertical move upwards 0.2 metres from push end position
        if success:
            st_time = time.time()
            success1, cur_joints = self.straight_move(pos1, over1, rot)
            if len(cur_joints)>0:
                all_joints = np.vstack((all_joints, cur_joints))
            success &= success1
        
        st_time = time.time()

        # 6) Go back home:
        success1, cur_joints = self.go_home()
        
        if len(cur_joints)>0:
                all_joints = np.vstack((all_joints, cur_joints))
        success &= success1

        return success, all_joints

    def get_push_start_and_end(self, push_dir):
        
        '''Given the yaw, we get the best push point, that prioritizes the change in position 
        '''

        targetYaw = self.env.client_id.getEulerFromQuaternion(self.target_obj_orientation)[2]
        target_cartesian_pos = np.array(self.target_obj_pos, dtype=float).reshape((3, 1))

        l = self.target_obj_size[0]
        b = self.target_obj_size[1]

        targetCornersStandard = np.array([
            [l/2, -l/2, -l/2, l/2],
            [b/2, b/2, -b/2, -b/2]
        ], dtype=float)

        targetMidpointsStandard = np.array([
            [l/2, 0, -l/2, 0],
            [0, b/2, 0, -b/2]
        ], dtype=float)

        pushPoints = np.concatenate([targetCornersStandard, targetMidpointsStandard], axis = 1)
        
        R = np.array([
            [np.cos(targetYaw), -1*np.sin(targetYaw)],
            [np.sin(targetYaw), np.cos(targetYaw)]
        ], dtype=float)

        pushPointswrtCOM = R @ pushPoints
        pushPoints_normalized = pushPointswrtCOM/np.linalg.norm(pushPointswrtCOM, axis = 0)

        push_vec = np.array([np.cos(push_dir), np.sin(push_dir)])

        best_corner = None
        max_neg_dist = 100  
        for i in range(pushPointswrtCOM.shape[1]):
            dist = pushPoints_normalized[:, i].T @ push_vec
            if dist < max_neg_dist:
                max_neg_dist = dist
                best_corner = pushPointswrtCOM[:, i].T  

        best_corner = target_cartesian_pos[0:2].T + best_corner

        desired_push_start = np.zeros(shape=(3, ))
        desired_push_start[0:2] = best_corner - (GRIPPER_PUSH_RADIUS + 0.01)*push_vec  
        desired_push_start [2] = self.target_obj_pos[2] - 0.01

        desired_push_end = copy.deepcopy(desired_push_start)
        desired_push_end[:2] = desired_push_start[:2] + PUSH_DISTANCE*push_vec

        return desired_push_start, desired_push_end
    
    def grasp(self, pose, angle):
        """Execute grasping primitive.

        Args:
            pose: SE(3) grasping pose.
            angle: rotation angle

        Returns:
            success: robot movement success if True.
        """

        # Adjust grasp positions.
        pos = np.array(pose, dtype=np.float32)
        pos[2] = max(pos[2] - 0.04, self.bounds[2][0])
        pos[2] += self.ee_tip_offset

        # Align against grasp direction.
        angle = ((angle) % np.pi) - np.pi / 2
        rot = p.getQuaternionFromEuler([np.pi / 2, np.pi / 2, -angle])

        over = (pos[0], pos[1], pos[2] + 0.2)

        # Execute push.
        self.open_gripper()
        success, _ = self.move_joints(self.ik_rest_joints)
        grasp_sucess = False
        if success:
            success = self.move_ee_pose((over, rot))
        if success:
            success = self.straight_move(over, pos, rot, detect_force=True)
        if success:
            self.close_gripper()
            success = self.straight_move(pos, over, rot)
            grasp_sucess = self.is_gripper_closed
        if success and grasp_sucess:
            success &= self.move_joints(self.drop_joints0, speed=0.005)[0]
            success &= self.move_joints(self.drop_joints1, speed=0.005)[0]
            grasp_sucess = self.is_gripper_closed
            self.open_gripper()
            grasp_sucess &= success
            success &= self.move_joints(self.drop_joints0)[0]
        else:
            grasp_sucess &= success
        self.open_gripper()
        success &= self.go_home()[0]

        print(f"Grasp at {pose}, {success}, the grasp {grasp_sucess}")

        return success, grasp_sucess

    def open_gripper(self):
        self._move_gripper(self.gripper_angle_open, speed=0.01)

    def close_gripper(self):
        self._move_gripper(self.gripper_angle_close, speed=0.01, is_slow=True)

    @property
    def is_gripper_closed(self):
        gripper_angle = self.client_id.getJointState(self.ee, self.gripper_main_joint)[0]
        return gripper_angle < self.gripper_angle_close_threshold

    def _move_gripper(self, target_angle, speed=0.01, timeout=3, max_force=5, is_slow=False):
        t0 = time.time()
        count = 0
        max_count = 3
        current_angle = self.client_id.getJointState(self.ee, self.gripper_main_joint)[0]

        if is_slow:
            self.client_id.setJointMotorControl2(
                self.ee,
                self.gripper_main_joint,
                p.POSITION_CONTROL,
                targetPosition=target_angle,
                maxVelocity=0.5,
                force=1,
            )
            self.client_id.setJointMotorControl2(
                self.ee,
                self.gripper_mimic_joints["right_outer_knuckle_joint"],
                p.POSITION_CONTROL,
                targetPosition=target_angle,
                maxVelocity=0.5,
                force=1,
            )
            for _ in range(500):
                self.client_id.stepSimulation()
        self.client_id.setJointMotorControl2(
            self.ee,
            self.gripper_main_joint,
            p.POSITION_CONTROL,
            targetPosition=target_angle,
            maxVelocity=10,
            force=5,
        )
        self.client_id.setJointMotorControl2(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            p.POSITION_CONTROL,
            targetPosition=target_angle,
            maxVelocity=10,
            force=5,
        )
        for _ in range(10):
            self.client_id.stepSimulation()