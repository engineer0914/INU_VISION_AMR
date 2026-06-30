# vision_node.py
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from arm_interfaces.srv import GetTargetPose
from vision_pkg.vision_6d_manager import (
    BRICK_IDS,
    COMPONENT_IDS,
    COMP_MODEL_PATH,
    DET_MODEL_PATH,
    ID_TO_CLASS,
    SEG_MODEL_PATH,
    Vision6DPoseManager,
)


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.srv = self.create_service(GetTargetPose, '/get_target_pose', self.get_pose_cb)
        self.get_logger().info('[VISION] loading vision manager')

        self.declare_parameter('det_model_path', DET_MODEL_PATH)
        self.declare_parameter('seg_model_path', SEG_MODEL_PATH)
        self.declare_parameter('comp_model_path', COMP_MODEL_PATH)
        self.declare_parameter('visualize_window', '6D Pose Service Result')
        self.declare_parameter('visualize_scale', 2.0)
        self.declare_parameter('service_visualize', True)
        self.declare_parameter('service_result_display_sec', 5.0)
        self.declare_parameter('use_shape_ratio_filter', True)
        self.declare_parameter('shape_ratio_threshold', 1.5)
        self.declare_parameter('edge_contact_max_px', 10)
        self.declare_parameter('edge_contact_margin_px', 2)
        self.declare_parameter('use_depth_median_filter', True)
        self.declare_parameter('depth_median_margin_m', 0.001)
        self.declare_parameter('depth_median_min_samples', 2)

        self.vision = None
        self.init_error = None
        self.shutdown_requested = False

        try:
            self.vision = Vision6DPoseManager(
                logger=self.get_logger(),
                det_model_path=self.get_parameter('det_model_path').value,
                seg_model_path=self.get_parameter('seg_model_path').value,
                comp_model_path=self.get_parameter('comp_model_path').value,
                # 서비스 함수에서 visualize 인자를 직접 넘기므로 여기 값은 내부 디버그 기본값 정도로만 사용된다.
                visualize=bool(self.get_parameter('service_visualize').value),
                visualize_window=self.get_parameter('visualize_window').value,
                visualize_scale=float(self.get_parameter('visualize_scale').value),
                use_shape_ratio_filter=bool(self.get_parameter('use_shape_ratio_filter').value),
                shape_ratio_threshold=float(self.get_parameter('shape_ratio_threshold').value),
                edge_contact_max_px=int(self.get_parameter('edge_contact_max_px').value),
                edge_contact_margin_px=int(self.get_parameter('edge_contact_margin_px').value),
                use_depth_median_filter=bool(self.get_parameter('use_depth_median_filter').value),
                depth_median_margin_m=float(self.get_parameter('depth_median_margin_m').value),
                depth_median_min_samples=int(self.get_parameter('depth_median_min_samples').value),
            )
            self.get_logger().info('[VISION] vision_node started - service branch mode')
            self.get_logger().info('[VISION] service IDs: brick=1~8, component=13/34/81/241/442/462/711/4482/8518/46262/48132')
        except Exception as e:
            self.init_error = str(e)
            self.get_logger().error(f'[VISION] init failed: {e}')

    def get_pose_cb(self, request, response):
        target_str = request.target_color.strip()
        self.get_logger().info(f'[VISION] service request target ID: {target_str}')

        try:
            if self.vision is None:
                response.success = False
                self.get_logger().error(f'[VISION] unavailable: init failed: {self.init_error}')
                return response

            if not target_str.isdigit():
                self.get_logger().error(f'[VISION] invalid input, expected numeric ID: {target_str}')
                response.success = False
                return response

            target_id = int(target_str)
            wait_ms = int(max(0.0, float(self.get_parameter('service_result_display_sec').value)) * 1000.0)
            service_visualize = bool(self.get_parameter('service_visualize').value)

            if target_id in BRICK_IDS:
                self.get_logger().info(
                    f'[VISION] branch=BRICK single-frame pipeline, ID={target_id}, class={ID_TO_CLASS.get(target_id)}'
                )
                result = self.vision.run_single_frame_brick_by_id(
                    target_id=target_id,
                    visualize=service_visualize,
                    wait_ms=wait_ms,
                )
            elif target_id in COMPONENT_IDS:
                self.get_logger().info(
                    f'[VISION] branch=COMPONENT single-frame 777-style axis pipeline, ID={target_id}, class={ID_TO_CLASS.get(target_id)}'
                )
                result = self.vision.run_single_frame_component_by_id(
                    target_id=target_id,
                    visualize=service_visualize,
                    wait_ms=wait_ms,
                )
            else:
                self.get_logger().error(f'[VISION] unsupported service target ID: {target_id}')
                response.success = False
                return response

            if getattr(self.vision, 'stop_requested', False):
                self.request_shutdown('[VISION] q/ESC pressed during service visualization')

            if result.success:
                response.success = True
                response.x = float(result.x_m)
                response.y = float(result.y_m)
                response.z = float(result.z_m)
                response.yaw = float(result.yaw_deg)
                response.class_name = str(result.class_name)
                self.get_logger().info(
                    f'[VISION] target found! ID={result.target_id}, Class={result.class_name}, '
                    f'X={result.x_m * 1000.0:.1f}mm, Y={result.y_m * 1000.0:.1f}mm, '
                    f'Z={result.z_m * 1000.0:.1f}mm, Yaw={result.yaw_deg:.2f}deg'
                )
            else:
                response.success = False
                self.get_logger().error(
                    f'[VISION] target search failed: ID={result.target_id}, '
                    f'Class={result.class_name}, Reason={result.reason}'
                )

        except Exception as e:
            self.get_logger().error(f'[VISION] fatal processing error: {e}')
            response.success = False

        return response

    def request_shutdown(self, reason):
        if not self.shutdown_requested:
            self.get_logger().info(reason)
        self.shutdown_requested = True
        if self.vision is not None:
            self.vision.stop_requested = True

    def destroy_node(self):
        if self.vision is not None:
            self.vision.shutdown()
            self.vision = None
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        while rclpy.ok() and not getattr(node, 'shutdown_requested', False):
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        node.shutdown_requested = True
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
