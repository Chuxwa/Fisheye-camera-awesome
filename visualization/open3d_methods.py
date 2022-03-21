import numpy as np
import open3d as o3d


class Open3DVisualizer(object):
    """Point cloud visualizer by using Open3D

    Args:
        name (str, optional): the name of the visualization window. Defaults to "stereo reconstruction".
        width (int, optional): the width of the visualization window. Defaults to 800.
        height (int, optional): the height of the visualization window. Defaults to 600.
        point_size (int, optional): the size of the points. Defaults to 1.
        filename (str, optional): the filename of the viewpoint. Defaults to "./config/view_point.json".
    """

    def __init__(
        self,
        name: str = "stereo reconstruction",
        width: int = 800,
        height: int = 600,
        point_size: int = 5,
        filename: str = "output/config/visualization/view_point.json",
    ):
        """Point cloud visualizer by using Open3D

        Args:
            name (str, optional): the name of the visualization window. Defaults to "stereo reconstruction".
            width (int, optional): the width of the visualization window. Defaults to 800.
            height (int, optional): the height of the visualization window. Defaults to 600.
            point_size (int, optional): the size of the points. Defaults to 1.
            filename (str, optional): the filename of the viewpoint. Defaults to "./config/view_point.json".
        """
        self.name = name
        self.width = width
        self.height = height
        # init the visualization window
        self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window(window_name=name, width=width, height=height)

        # init the visualization parameter
        self.opt = self.visualizer.get_render_option()
        self.opt.background_color = np.asarray([1, 1, 1])
        self.opt.point_size = point_size
        self.opt.show_coordinate_frame = True

        # load viewpoint
        self.LoadViewPoint(filename)
        # self.param = o3d.io.read_pinhole_camera_parameters(filename)
        # self.ctr = self.visualizer.get_view_control()

        # init the point cloud
        self.pointcloud = o3d.geometry.PointCloud()
        self.visualizer.add_geometry(self.pointcloud)

    def __del__(self):
        """Del the window."""
        self.visualizer.destroy_window()

    def UpdatePointCloud(
        self, points: np.ndarray, colors: np.ndarray, use_rgb: bool = True
    ):
        """Update the point cloud

        Args:
            points (np.ndarray): the positions of the points
            colors (np.ndarray): the colors of the points
            use_rgb (bool): use rgb or use bgr
        """
        if use_rgb:
            colors = colors[:, (2, 1, 0)]

        self.pointcloud.points = o3d.utility.Vector3dVector(points)
        self.pointcloud.colors = o3d.utility.Vector3dVector(colors)

        self.DisplayInlier(voxel_size=0.05, nb_points = 16, radius = 1)
        self.DisplayInlier(voxel_size=0.05, nb_points = 16, radius = 1)
        # self.DisplayInlier(nb_points = 8, radius = 6)

        self.visualizer.clear_geometries()  # clear
        self.visualizer.add_geometry(self.pointcloud)  # add
        # self.visualizer.update_geometry(self.pointcloud)  # update
        # self.vis.remove_geometry(self.pointcloud)          # remove

        # set viewpoint
        self.ctr.convert_from_pinhole_camera_parameters(self.param)
        self.visualizer.run()
        self.visualizer.poll_events()
        self.visualizer.update_renderer()

    def CaptureScreen(self, filename: str, depth: bool = False):
        """Capture the visualization screen

        Args:
            filename (str): the path of saving image
            depth (bool, optional): capture depth map or full screen. Defaults to False.
        """
        if depth:
            self.visualizer.capture_depth_image(filename, False)
        else:
            self.visualizer.capture_screen_image(filename, False)

    def SaveViewPoint(self, filename: str):
        """Save view point of the visualization screen

        Args:
            filename (str): the path of saving view point
        """
        self.visualizer.run()  # user changes the view and press "q" to terminate
        self.param = self.ctr.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(filename, self.param)
        self.visualizer.destroy_window()

    def LoadViewPoint(self, filename: str):
        """Load view point of the visualization screen

        Args:
            filename (str): the path of loading view point
        """
        self.param = o3d.io.read_pinhole_camera_parameters(filename)
        self.ctr = self.visualizer.get_view_control()
        self.ctr.convert_from_pinhole_camera_parameters(self.param)

    def custom_draw_geometry_with_key_callback(self):
        def rotate_view_w():
            self.ctr.rotate(10.0, 0.0)
            return False

        def rotate_view_s():
            self.ctr.rotate(10.0, 0.0)
            return False

        def rotate_view_a():
            self.ctr.rotate(10.0, 0.0)
            return False

        def rotate_view_d():
            self.ctr.rotate(0.0, 10.0)
            return False

        key_to_callback = {}
        key_to_callback[ord("w")] = rotate_view_w
        key_to_callback[ord("s")] = rotate_view_s
        key_to_callback[ord("a")] = rotate_view_a
        key_to_callback[ord("d")] = rotate_view_d
        o3d.visualization.draw_geometries_with_key_callbacks(
            [self.pointcloud], key_to_callback
        )

    def DisplayInlier(
        self,
        voxel_size: float = 0.2,
        nb_points: int = 16,
        radius: float = 6.0,
    ):
        """Remove outlier points

        Args:
            voxel_size (float, optional): voxelization. Defaults to 0.2.
            nb_points (int, optional): neighborhood points. Defaults to 16.
            radius (float, optional): radius. Defaults to 6.0.
        """
        self.pointcloud = self.pointcloud.voxel_down_sample(voxel_size=voxel_size)
        _, ind = self.pointcloud.remove_radius_outlier(
            nb_points=nb_points, radius=radius
        )
        self.pointcloud = self.pointcloud.select_by_index(ind)
