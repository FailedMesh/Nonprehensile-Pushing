import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc

import numpy as np
import cameras

from force_model_functions import slider_geometry

class Environment:
    
    def __init__(self, gui=True, time_step=1 / 480. ):

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
        self.table = None
        self.table_height = 0.625
        self.target_obj = None

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

        # ---------------------------------------------- #

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
    
    def create_cuboid(self, obj_size, obj_pose, collision = True):
        
        color = np.array([237, 201, 72, 255])/255
        obj_orientation = self.client_id.getQuaternionFromEuler([0, 0, obj_pose[2]])
        obj_pos = np.array([obj_pose[0], 
                            obj_pose[1], 
                            self.table_height + obj_size[2]/2])

        vuid = self.client_id.createVisualShape(p.GEOM_BOX, 
                                                    halfExtents = obj_size/2, 
                                                    rgbaColor=color)
        
        if collision:
        
            cuid = self.client_id.createCollisionShape(p.GEOM_BOX, halfExtents = obj_size/2)
            
            obj_id = self.client_id.createMultiBody(baseMass=0.2, 
                                                            baseCollisionShapeIndex = cuid, 
                                                            baseVisualShapeIndex = vuid, 
                                                            basePosition = obj_pos, 
                                                            baseOrientation = obj_orientation)
        
        else:

            obj_id = self.client_id.createMultiBody(baseMass=0.2, 
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
            lateralFriction=0.1,
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

    def reset(self):
        
        for i in self.obj_ids["rigid"]:
            self.client_id.removeBody(i)
            self.remove_object_id(i)
        
        self.obj_ids = {"fixed": [], "rigid": []}
        self.client_id.setGravity(0, 0, -9.8)           # Set Gravity

    def update_target_pose(self):

        self.target_obj_pos, self.target_obj_orientation = self.client_id.getBasePositionAndOrientation(self.target_obj)
        self.target_obj_pos, self.target_obj_orientation = np.array(self.target_obj_pos), np.array(self.target_obj_orientation)
        self.target_obj_euler = self.client_id.getEulerFromQuaternion(self.target_obj_orientation)[2]

        self.target_obj_pose = np.append(self.target_obj_pos[:2], self.target_obj_euler)
    
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

    def push_target(self, ext_force, phi, start_pos, params):

        target_obj_size = np.array([params['length'], params['breadth'], params['height']])
        
        self.target_obj = self.create_cuboid(target_obj_size, start_pos)
        ext_force = np.append(ext_force, 0.)

        r = slider_geometry(phi, params['length'], params['breadth'])
        contact_point = np.array([r*np.cos(phi), r*np.sin(phi), 0.])

        self.client_id.applyExternalForce(self.target_obj, -1,
                                          ext_force,
                                          contact_point,
                                          flags = p.LINK_FRAME)
        
        self.update_target_pose()
        vel = 1e+10
        prev_target_pose = self.target_obj_pos.copy()

        velocities = []
        poses = [prev_target_pose]

        thresh = 0.001

        while vel > thresh:

            self.client_id.stepSimulation()
            self.update_target_pose()

            vel = np.linalg.norm(self.target_obj_pos - prev_target_pose)/self.time_step

            prev_target_pose = self.target_obj_pos.copy()

            velocities.append(vel)
            poses.append(prev_target_pose)
        
        return np.array(poses), np.array(velocities)




    