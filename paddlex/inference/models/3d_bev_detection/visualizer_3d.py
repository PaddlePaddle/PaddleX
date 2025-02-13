import os
import numpy as np
import argparse
import importlib.util
import sys

class LazyLoader:
    def __init__(self, module_name):
        self.module_name = module_name
        self._module = None

    def __getattr__(self, item):
        if self._module is None:
            self._module = importlib.import_module(self.module_name)
        return getattr(self._module, item)

open3d = LazyLoader('open3d')

class Visualizer3D:
    def __init__(self):
        self.vis = open3d.visualization.Visualizer() # initialize visualizer

    def boxes_to_lines(self, box: np.ndarray):
        """
           4-------- 6
         /|         /|
        5 -------- 3 .
        | |        | |
        . 7 -------- 1
        |/         |/
        2 -------- 0
        """
        center = box[0:3]
        lwh = box[3:6]
        angles = np.array([0, 0, box[6] + 1e-10])
        rot = open3d.geometry.get_rotation_matrix_from_axis_angle(angles)
        box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)
        return open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    def draw_results(self, points: np.ndarray, result: dict, score_threshold: float) -> None:
        scores = result["scores"]
        bbox3d = result["bbox3d"]
        label_preds = result["labels"]

        num_bbox3d, bbox3d_dims = bbox3d.shape
        result_boxes = []
        for box_idx in range(num_bbox3d):
            if scores[box_idx] < score_threshold:
                continue
            if bbox3d_dims == 9:
                print(
                    "Score: {} Label: {} Box(x_c, y_c, z_c, w, l, h, vec_x, vec_y, -rot): {} {} {} {} {} {} {} {} {}"
                    .format(scores[box_idx], label_preds[box_idx],
                            bbox3d[box_idx, 0], bbox3d[box_idx, 1],
                            bbox3d[box_idx, 2], bbox3d[box_idx, 3],
                            bbox3d[box_idx, 4], bbox3d[box_idx, 5],
                            bbox3d[box_idx, 6], bbox3d[box_idx, 7],
                            bbox3d[box_idx, 8]))
            elif bbox3d_dims == 7:
                print(
                    "Score: {} Label: {} Box(x_c, y_c, z_c, w, l, h, -rot): {} {} {} {} {} {} {}"
                    .format(scores[box_idx], label_preds[box_idx],
                            bbox3d[box_idx, 0], bbox3d[box_idx, 1],
                            bbox3d[box_idx, 2], bbox3d[box_idx, 3],
                            bbox3d[box_idx, 4], bbox3d[box_idx, 5],
                            bbox3d[box_idx, 6]))
            # draw result
            result_boxes.append([
                bbox3d[box_idx, 0], bbox3d[box_idx, 1],
                bbox3d[box_idx, 2], bbox3d[box_idx, 3],
                bbox3d[box_idx, 4], bbox3d[box_idx, 5],
                bbox3d[box_idx, -1]
            ])

        # config
        self.vis.create_window()
        self.vis.get_render_option().point_size = 1.0
        self.vis.get_render_option().background_color = [0, 0, 0]
        pc_color = [1, 1, 1]
        num_points = len(points)
        pc_colors = np.tile(pc_color, (num_points, 1))

        # raw point cloud
        pts = open3d.geometry.PointCloud()
        pts.points = open3d.utility.Vector3dVector(points[:, :3])
        pts.colors = open3d.utility.Vector3dVector(pc_colors)
        self.vis.add_geometry(pts)

        # result_boxes
        obs_color = [1, 0, 0]
        result_boxes = np.array(result_boxes)
        for i in range(result_boxes.shape[0]):
            lines = self.boxes_to_lines(result_boxes[i])
            # show different colors for different classes
            if label_preds[i] <= 4:
                obs_color = [0, 1, 0] # 'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            elif (label_preds[i] <= 6):
                obs_color = [0, 0, 1] # 'bicycle', 'motorcycle'
            elif (label_preds[i] <= 7):
                obs_color = [1, 0, 0] # 'pedestrian'
            else:
                obs_color = [1, 0, 1] # 'traffic_cone','barrier'
            lines.paint_uniform_color(obs_color)
            self.vis.add_geometry(lines)

        self.vis.run()
        self.vis.poll_events()
        self.vis.update_renderer()
        # self.vis.capture_screen_image("result.png")
        self.vis.destroy_window()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Visualizer 3d')
    parser.add_argument(
        '--save_path',
        type=str,
        default=None)
    
    args = parser.parse_args()
    save_path = args.save_path
    if save_path is None:
        raise ValueError("Please specify the path to the saved results.")
    
    points = np.load(os.path.join(save_path, "points.npy"), allow_pickle=True)
    result = np.load(os.path.join(save_path, "results.npy"), allow_pickle=True).item()
    
    score_threshold = 0.25
    vis = Visualizer3D()
    vis.draw_results(points, result, score_threshold)
