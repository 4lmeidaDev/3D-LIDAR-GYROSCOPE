#!/usr/bin/env python3
"""
lidar_clustering — Nó ROS2 para clustering de point cloud com DBSCAN (open3d).

Subscreve:  PointCloud2  (input_topic)
Publica:    MarkerArray  (output_topic) — bounding boxes + labels por cluster
"""

import rclpy
from rclpy.node import Node
import numpy as np
import open3d as o3d

from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration

import sensor_msgs_py.point_cloud2 as pc2


# Paleta de 12 cores distintas para os clusters
_PALETTE = [
    (1.00, 0.20, 0.20),  # vermelho
    (0.20, 0.90, 0.20),  # verde
    (0.20, 0.50, 1.00),  # azul
    (1.00, 0.80, 0.00),  # amarelo
    (1.00, 0.40, 0.00),  # laranja
    (0.80, 0.20, 1.00),  # roxo
    (0.00, 0.90, 0.90),  # ciano
    (1.00, 0.00, 0.60),  # rosa
    (0.00, 0.60, 0.00),  # verde escuro
    (0.60, 0.30, 0.00),  # castanho
    (0.50, 0.50, 1.00),  # lavanda
    (1.00, 0.60, 0.80),  # salmão
]


class LidarClusteringNode(Node):

    def __init__(self):
        super().__init__('lidar_clustering')

        # ── Parâmetros ──────────────────────────────────────────────────
        self.declare_parameter('input_topic',       '/lidar/point_cloud')
        self.declare_parameter('output_topic',      '/lidar/clusters')
        self.declare_parameter('frame_id',          'map')
        self.declare_parameter('voxel_size',         0.05)
        self.declare_parameter('z_min',              0.05)
        self.declare_parameter('z_max',              3.00)
        self.declare_parameter('cluster_tolerance',  0.30)
        self.declare_parameter('min_cluster_size',   10)
        self.declare_parameter('max_cluster_size',   25000)
        self.declare_parameter('marker_lifetime',    1.0)

        self._load_params()

        # ── Publisher / Subscriber ───────────────────────────────────────
        self._pub = self.create_publisher(MarkerArray, self._out_topic, 10)
        self._sub = self.create_subscription(
            PointCloud2, self._in_topic, self._callback, 10
        )

        self.add_on_set_parameters_callback(self._on_params_change)

        self._prev_n_clusters = 0

        self.get_logger().info(
            f'LidarClustering iniciado.\n'
            f'  input : {self._in_topic}\n'
            f'  output: {self._out_topic}\n'
            f'  voxel : {self._voxel_size * 100:.0f} cm  |  '
            f'z=[{self._z_min:.2f}, {self._z_max:.2f}] m\n'
            f'  DBSCAN eps={self._eps:.2f} m  |  '
            f'pts=[{self._min_pts}, {self._max_pts}]'
        )

    # ────────────────────────────────────────────────────────────────────
    def _load_params(self):
        self._in_topic   = self.get_parameter('input_topic').value
        self._out_topic  = self.get_parameter('output_topic').value
        self._frame_id   = self.get_parameter('frame_id').value
        self._voxel_size = self.get_parameter('voxel_size').value
        self._z_min      = self.get_parameter('z_min').value
        self._z_max      = self.get_parameter('z_max').value
        self._eps        = self.get_parameter('cluster_tolerance').value
        self._min_pts    = self.get_parameter('min_cluster_size').value
        self._max_pts    = self.get_parameter('max_cluster_size').value
        self._lifetime_s = self.get_parameter('marker_lifetime').value

    def _on_params_change(self, params):
        from rcl_interfaces.msg import SetParametersResult
        self._load_params()
        self.get_logger().info('Parâmetros atualizados em runtime.')
        return SetParametersResult(successful=True)

    # ────────────────────────────────────────────────────────────────────
    def _callback(self, msg: PointCloud2):
        # 1. PointCloud2 → numpy (N, 3)
        pts = self._parse_cloud(msg)
        if pts is None or len(pts) < self._min_pts:
            self._publish_empty()
            return

        # 2. open3d PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # 3. Voxel Grid downsample
        pcd = pcd.voxel_down_sample(self._voxel_size)

        # 4. PassThrough filter (Z) — remove chão e teto
        pts_down = np.asarray(pcd.points, dtype=np.float32)
        mask_z   = (pts_down[:, 2] >= self._z_min) & (pts_down[:, 2] <= self._z_max)
        pts_filt = pts_down[mask_z]

        if len(pts_filt) < self._min_pts:
            self._publish_empty()
            return

        pcd_filt = o3d.geometry.PointCloud()
        pcd_filt.points = o3d.utility.Vector3dVector(pts_filt)

        # 5. DBSCAN clustering
        labels = np.array(
            pcd_filt.cluster_dbscan(
                eps=self._eps,
                min_points=self._min_pts,
                print_progress=False,
            ),
            dtype=np.int32,
        )

        n_clusters = int(labels.max()) + 1 if labels.size > 0 and labels.max() >= 0 else 0

        if n_clusters == 0:
            self._publish_empty()
            return

        # 6. Construir MarkerArray
        markers = MarkerArray()
        frame_id = msg.header.frame_id if msg.header.frame_id else self._frame_id
        stamp    = msg.header.stamp

        # Apagar markers de clusters que já não existem neste frame
        for i in range(n_clusters, self._prev_n_clusters):
            markers.markers.append(self._delete_marker(i * 2))
            markers.markers.append(self._delete_marker(i * 2 + 1))

        for cluster_id in range(n_clusters):
            cluster_pts = pts_filt[labels == cluster_id]

            if len(cluster_pts) > self._max_pts:
                continue

            r, g, b = _PALETTE[cluster_id % len(_PALETTE)]

            bbox   = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(cluster_pts)
            )
            center = bbox.get_center()   # ndarray (3,)
            extent = bbox.get_extent()   # ndarray (3,)

            # Bounding box — CUBE semi-transparente
            markers.markers.append(self._bbox_marker(
                marker_id=cluster_id * 2,
                center=center,
                extent=extent,
                color=ColorRGBA(r=r, g=g, b=b, a=0.25),
                frame_id=frame_id,
                stamp=stamp,
            ))

            # Label de texto flutuante acima do bbox
            label_pos = center.copy()
            label_pos[2] += extent[2] / 2.0 + 0.15
            markers.markers.append(self._text_marker(
                marker_id=cluster_id * 2 + 1,
                pos=label_pos,
                text=f'Entidade {cluster_id + 1}\n{len(cluster_pts)} pts',
                color=ColorRGBA(r=r, g=g, b=b, a=1.0),
                frame_id=frame_id,
                stamp=stamp,
            ))

        self._prev_n_clusters = n_clusters
        self._pub.publish(markers)

        self.get_logger().info(
            f'{n_clusters} entidade(s) | {len(pts_filt)} pts filtrados',
            throttle_duration_sec=2.0,
        )

    # ────────────────────────────────────────────────────────────────────
    def _parse_cloud(self, msg: PointCloud2):
        """Converte PointCloud2 → numpy float32 (N, 3). Suporta arrays estruturados."""
        try:
            raw = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
            if raw.dtype.names:  # structured array (versões antigas do sensor_msgs_py)
                pts = np.column_stack([
                    raw['x'].astype(np.float32),
                    raw['y'].astype(np.float32),
                    raw['z'].astype(np.float32),
                ])
            else:
                pts = raw.astype(np.float32)
            return pts if pts.ndim == 2 and pts.shape[1] == 3 else None
        except Exception as e:
            self.get_logger().warning(f'Erro ao parsear PointCloud2: {e}')
            return None

    # ────────────────────────────────────────────────────────────────────
    def _lifetime(self) -> Duration:
        sec  = int(self._lifetime_s)
        nsec = int((self._lifetime_s - sec) * 1e9)
        return Duration(sec=sec, nanosec=nsec)

    def _delete_marker(self, marker_id: int) -> Marker:
        m = Marker()
        m.ns     = 'clusters'
        m.id     = marker_id
        m.action = Marker.DELETE
        return m

    def _bbox_marker(self, marker_id, center, extent, color, frame_id, stamp) -> Marker:
        m = Marker()
        m.header.frame_id    = frame_id
        m.header.stamp       = stamp
        m.ns                 = 'clusters'
        m.id                 = marker_id
        m.type               = Marker.CUBE
        m.action             = Marker.ADD
        m.pose.position.x    = float(center[0])
        m.pose.position.y    = float(center[1])
        m.pose.position.z    = float(center[2])
        m.pose.orientation.w = 1.0
        m.scale.x            = float(max(extent[0], 0.05))
        m.scale.y            = float(max(extent[1], 0.05))
        m.scale.z            = float(max(extent[2], 0.05))
        m.color              = color
        m.lifetime           = self._lifetime()
        return m

    def _text_marker(self, marker_id, pos, text, color, frame_id, stamp) -> Marker:
        m = Marker()
        m.header.frame_id    = frame_id
        m.header.stamp       = stamp
        m.ns                 = 'clusters'
        m.id                 = marker_id
        m.type               = Marker.TEXT_VIEW_FACING
        m.action             = Marker.ADD
        m.pose.position.x    = float(pos[0])
        m.pose.position.y    = float(pos[1])
        m.pose.position.z    = float(pos[2])
        m.pose.orientation.w = 1.0
        m.scale.z            = 0.15   # altura do texto em metros
        m.color              = color
        m.text               = text
        m.lifetime           = self._lifetime()
        return m

    def _publish_empty(self):
        """Apaga todos os markers do frame anterior."""
        if self._prev_n_clusters == 0:
            return
        markers = MarkerArray()
        for i in range(self._prev_n_clusters):
            markers.markers.append(self._delete_marker(i * 2))
            markers.markers.append(self._delete_marker(i * 2 + 1))
        self._pub.publish(markers)
        self._prev_n_clusters = 0


def main(args=None):
    rclpy.init(args=args)
    node = LidarClusteringNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
