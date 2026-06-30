import os
import time
from dataclasses import dataclass

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


WORKSPACE_DIR = os.environ.get("ROS2_WS", "/home/st02/ros2_ws")
DET_MODEL_PATH = os.path.join(WORKSPACE_DIR, "best.pt")
SEG_MODEL_PATH = os.path.join(WORKSPACE_DIR, "best_old.pt")
COMP_MODEL_PATH = os.path.join(WORKSPACE_DIR, "best_comp.pt")

BRICK_IDS = {1, 2, 3, 4, 5, 6, 7, 8}
COMPONENT_IDS = {13, 34, 81, 241, 442, 462, 711, 4482, 8518, 46262, 48132}
COMPONENT_VIEW_ID = 777

ID_TO_CLASS = {
    1: "2x2_red",
    2: "2x2_green",
    3: "2x2_blue",
    4: "2x2_yellow",
    5: "4x2_red",
    6: "4x2_green",
    7: "4x2_blue",
    8: "4x2_yellow",
    777: "components",
    999: "assembly",
    888: "assembly_fine",
    13: "Magnet",
    34: "Battery",
    81: "Estop",
    241: "Trafficlight",
    442: "carrot",
    462: "small tree",
    711: "hammer",
    4482: "bigcarrot",
    8518: "burger",
    46262: "bigtree",
    48132: "icecream",
}

COMPONENT_COLOR_RULES = {
    "magnet": {"axis": "major", "color": "blue"},
    "battery": {"axis": "major", "color": "yellow"},
    "estop": {"axis": "minor", "color": "red"},
    "trafficlight": {"axis": "major", "color": "red"},
    "carrot": {"axis": "major", "color": "green"},
    "smalltree": {"special": "tree"},
    "hammer": {"axis": "major", "color": "blue"},
    "bigcarrot": {"axis": "major", "color": "green"},
    "burger": {"special": "burger"},
    "bigtree": {"special": "tree"},
    "icecream": {"axis": "major", "color": "green"},
}

# OpenCV HSV hue range: 0~179.
# 색 영역 검출은 H값을 중심으로 보고, 무채색/저채도 픽셀만 최소 S guard로 제외한다.
# V는 직접 조건으로 거의 쓰지 않아서 조명 밝기 변화에 덜 흔들리게 한다.
HUE_COLOR_PARAMS = {
    "red": {"center": 0, "tol": 14, "min_s": 30},
    "yellow": {"center": 29, "tol": 18, "min_s": 30},
    "green": {"center": 62, "tol": 28, "min_s": 25},
    "blue": {"center": 112, "tol": 24, "min_s": 25},
}
HSV_COLOR_RANGES = HUE_COLOR_PARAMS  # endpoint score 루프 호환용 alias


@dataclass
class PoseResult:
    success: bool
    target_id: int | None = None
    class_name: str | None = None
    x_m: float | None = None
    y_m: float | None = None
    z_m: float | None = None
    yaw_deg: float | None = None
    layer: int | None = None
    reason: str | None = None


class Vision6DPoseManager:
    def __init__(
        self,
        logger=None,
        det_model_path=DET_MODEL_PATH,
        seg_model_path=SEG_MODEL_PATH,
        comp_model_path=COMP_MODEL_PATH,
        sample_sec=1.2,
        min_samples=5,
        match_distance_px=40.0,
        visualize=False,
        visualize_window="6D Pose (Ensemble Mode)",
        visualize_scale=2.0,
        use_shape_ratio_filter=True,
        shape_ratio_threshold=1.5,
        edge_contact_max_px=10,
        edge_contact_margin_px=2,
        use_depth_median_filter=True,
        depth_median_margin_m=0.030,
        depth_median_min_samples=2,
    ):
        self.logger = logger
        self.det_model_path = det_model_path
        self.seg_model_path = seg_model_path
        self.comp_model_path = comp_model_path
        self.sample_sec = float(sample_sec)
        self.min_samples = int(min_samples)
        self.match_distance_px = float(match_distance_px)
        self.visualize = bool(visualize)
        self.visualize_window = str(visualize_window)
        self.visualize_scale = max(0.1, float(visualize_scale))
        self.use_shape_ratio_filter = bool(use_shape_ratio_filter)
        self.shape_ratio_threshold = float(shape_ratio_threshold)
        self.edge_contact_max_px = int(edge_contact_max_px)
        self.edge_contact_margin_px = int(edge_contact_margin_px)
        self.use_depth_median_filter = bool(use_depth_median_filter)
        self.depth_median_margin_m = float(depth_median_margin_m)
        self.depth_median_min_samples = int(depth_median_min_samples)
        self.stop_requested = False
        self._pipeline_started = False

        self.component_target_ids = set(COMPONENT_IDS)
        self.component_class_keys = {
            self._normalize_class_name(ID_TO_CLASS[item])
            for item in self.component_target_ids
            if item in ID_TO_CLASS
        }
        self.component_class_compact_keys = {
            self._compact_class_name(ID_TO_CLASS[item])
            for item in self.component_target_ids
            if item in ID_TO_CLASS
        }

        self._check_model_file(self.det_model_path)
        self._check_model_file(self.seg_model_path)
        self._check_model_file(self.comp_model_path)

        self.model_det = YOLO(self.det_model_path)
        self.model_seg = YOLO(self.seg_model_path)
        self.model_comp = YOLO(self.comp_model_path)

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = self.pipeline.start(config)
        self._pipeline_started = True
        self.align = rs.align(rs.stream.color)
        self.intrinsics = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )

        self._log_info(
            f"vision loaded: det={self.det_model_path}, seg={self.seg_model_path}, comp={self.comp_model_path}, "
            f"det_task={self.model_det.task}, seg_task={self.model_seg.task}, comp_task={self.model_comp.task}, "
            f"visualize={self.visualize}, visualize_scale={self.visualize_scale}, shape_filter={self.use_shape_ratio_filter}, "
            f"shape_ratio_threshold={self.shape_ratio_threshold}, edge_contact_max_px={self.edge_contact_max_px}, "
            f"edge_contact_margin_px={self.edge_contact_margin_px}, depth_median_filter={self.use_depth_median_filter}, "
            f"depth_median_margin_m={self.depth_median_margin_m}, depth_median_min_samples={self.depth_median_min_samples}"
        )

    def shutdown(self):
        self.stop_requested = True
        if self._pipeline_started:
            try:
                self.pipeline.stop()
            except Exception as exc:
                self._log_warn(f"RealSense pipeline stop failed: {exc}")
            finally:
                self._pipeline_started = False
        try:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)
        except Exception:
            pass

    def run_pipeline_by_id(self, target_id):
        """Backward-compatible entry point.

        이전 버전에서는 sample_sec 동안 여러 프레임을 모아서 median pose를 반환했다.
        현재 서비스 경로에서는 샘플링을 하지 않고 1프레임만 처리한다.
        """
        return self.run_single_frame_by_id(target_id, visualize=self.visualize, wait_ms=1)

    def run_pipeline_by_class(self, target_id, class_name):
        """Backward-compatible wrapper.

        class_name 인자는 기존 호출부 호환용으로만 남긴다.
        실제 분기는 target_id 기준으로 run_single_frame_by_id()에서 수행한다.
        """
        return self.run_single_frame_by_id(target_id, visualize=self.visualize, wait_ms=1)

    def run_single_frame_by_id(self, target_id, visualize=True, wait_ms=5000):
        """서비스 요청 1회에 대해 1프레임만 처리해서 PoseResult를 반환한다.

        분기 규칙:
          - 1~8: brick 전용 단일 프레임 함수
          - 13, 34, ...: component 전용 단일 프레임 함수

        component 전용 함수는 live_view_id=777에서 보던 것과 같은 YOLO component model,
        HSV H 중심 색 분리, tree/burger 특수 축 계산을 그대로 사용한다.
        """
        try:
            target_id = int(target_id)
        except Exception:
            return PoseResult(False, reason=f"invalid target id: {target_id}")

        if target_id in BRICK_IDS:
            return self.run_single_frame_brick_by_id(target_id, visualize=visualize, wait_ms=wait_ms)
        if target_id in COMPONENT_IDS:
            return self.run_single_frame_component_by_id(target_id, visualize=visualize, wait_ms=wait_ms)

        return PoseResult(False, target_id=target_id, reason=f"unsupported service target id: {target_id}")

    def run_single_frame_brick_by_id(self, target_id, visualize=True, wait_ms=5000):
        """1~8 brick 서비스용 단일 프레임 처리.

        이 함수는 live_view target_id=0에서 보던 brick 처리 흐름을 서비스용으로
        분리한 것이다. 샘플링 루프 없이 한 프레임에서만 다음 과정을 수행한다.

        YOLO det/seg → segmentation edge contact filter → 2x2/4x2 shape ratio filter
        → frame median depth filter → mask minAreaRect yaw → x/y/z/yaw 반환.
        """
        class_name = ID_TO_CLASS.get(int(target_id))
        if class_name is None:
            return PoseResult(False, target_id=target_id, reason=f"unknown brick id: {target_id}")

        target_key = self._normalize_class_name(class_name)
        self._log_info(f"single-frame brick live0-style search: id={target_id}, class={class_name}")

        try:
            ok, vis_payload = self._process_one_frame_brick_live0_style(target_id, class_name, target_key)
        except Exception as exc:
            self._log_warn(f"single-frame brick failed: {exc}")
            return PoseResult(False, target_id=target_id, class_name=class_name, reason=str(exc))

        result = self._pose_result_from_payload(target_id, class_name, vis_payload if ok else None)
        if visualize and ok:
            self.show_visualization(
                vis_payload["det_result"],
                vis_payload["detections"],
                vis_payload["target_label"],
                best=vis_payload.get("best"),
                result=result,
                wait_ms=wait_ms,
                close_after=True,
            )
        return result

    def _process_one_frame_brick_live0_style(self, target_id, class_name, target_key):
        """live_view target_id=0의 brick 전처리를 서비스용으로 그대로 사용한다."""
        frames = self.pipeline.wait_for_frames(timeout_ms=500)
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not color_frame or not depth_frame:
            return False, None

        image = np.asanyarray(color_frame.get_data())
        det_result, seg_result = self._infer_for_mode("brick_target", image)
        if det_result.boxes is None:
            return False, None

        detections_for_vis = []
        pre_candidates = []

        # ------------------------------------------------------------
        # 1차 필터: live_view id=0 brick 처리와 동일
        # - YOLO bbox 중심 기준 u,v
        # - segmentation mask가 이미지 테두리에 길게 닿으면 제외
        # - 2x2/4x2 segmentation minAreaRect 비율이 class와 맞지 않으면 제외
        # - 중심 depth가 없으면 제외
        # ------------------------------------------------------------
        for det_idx, box in enumerate(det_result.boxes):
            cls_name = det_result.names[int(box.cls[0])]
            cls_key = self._normalize_class_name(cls_name)

            # service brick branch에서는 brick class만 시각화/후보화한다.
            compact_key = self._compact_class_name(cls_name)
            if not (compact_key.startswith("2x2") or compact_key.startswith("4x2")):
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            u = int((xyxy[0] + xyxy[2]) / 2)
            v = int((xyxy[1] + xyxy[3]) / 2)

            is_target = self._target_matches(target_key, cls_key)
            mask_pts = self.get_matching_mask_points(
                det_result=det_result,
                seg_result=seg_result,
                det_idx=det_idx,
                target_u=u,
                target_v=v,
            )

            if mask_pts is not None:
                edge_info = self.get_mask_edge_contact_info(mask_pts, image.shape, self.edge_contact_margin_px)
                if edge_info["max_px"] > self.edge_contact_max_px:
                    detections_for_vis.append({
                        "u": u,
                        "v": v,
                        "z": 0.0,
                        "yaw": 0.0,
                        "ratio": None,
                        "class_name": f"{cls_name}_edge{edge_info['max_px']}px",
                        "is_target": False,
                        "axis_info": None,
                    })
                    continue

                if self.use_shape_ratio_filter:
                    shape_ok, ratio = self.brick_shape_ratio_pass(cls_name, mask_pts)
                    if not shape_ok:
                        detections_for_vis.append({
                            "u": u,
                            "v": v,
                            "z": 0.0,
                            "yaw": 0.0,
                            "ratio": ratio,
                            "class_name": f"{cls_name}_shape_r{ratio:.2f}",
                            "is_target": False,
                            "axis_info": None,
                        })
                        continue
                else:
                    _, ratio = self.brick_shape_ratio_pass(cls_name, mask_pts)
            else:
                ratio = None

            z = self.get_valid_depth(depth_frame, u, v)
            if z <= 0.0:
                detections_for_vis.append({
                    "u": u,
                    "v": v,
                    "z": z,
                    "yaw": 0.0,
                    "ratio": ratio,
                    "class_name": str(cls_name),
                    "is_target": is_target,
                    "axis_info": None,
                })
                continue

            yaw = self.find_yaw_from_mask_points(mask_pts) if mask_pts is not None else 0.0
            pre_candidates.append({
                "u": u,
                "v": v,
                "z": float(z),
                "yaw": float(yaw),
                "ratio": ratio,
                "detected_class": str(cls_name),
                "class_key": cls_key,
                "is_target": is_target,
                "axis_info": None,
                "mask_pts": mask_pts,
            })

        # ------------------------------------------------------------
        # 2차 필터: live_view id=0의 frame median depth filter
        # 같은 프레임에서 살아남은 brick 후보들의 중심 depth median과
        # depth_median_margin_m 이상 차이나면 제외한다.
        # ------------------------------------------------------------
        depth_median = None
        if self.use_depth_median_filter:
            valid_depths = [item["z"] for item in pre_candidates if item["z"] > 0.0]
            if len(valid_depths) >= self.depth_median_min_samples:
                depth_median = float(np.median(np.array(valid_depths, dtype=float)))

        valid_targets = []
        all_z_values = []
        best = None
        best_z = float("inf")

        for item in pre_candidates:
            cls_name = item["detected_class"]
            depth_diff = 0.0 if depth_median is None else abs(item["z"] - depth_median)
            if (depth_median is not None) and (depth_diff > self.depth_median_margin_m):
                detections_for_vis.append({
                    "u": item["u"],
                    "v": item["v"],
                    "z": item["z"],
                    "yaw": item["yaw"],
                    "ratio": item.get("ratio"),
                    "class_name": f"{cls_name}_depth{depth_diff*1000.0:.0f}mm",
                    "is_target": False,
                    "axis_info": None,
                })
                continue

            detections_for_vis.append({
                "u": item["u"],
                "v": item["v"],
                "z": item["z"],
                "yaw": item["yaw"],
                "ratio": item.get("ratio"),
                "class_name": str(cls_name),
                "is_target": item["is_target"],
                "axis_info": None,
            })
            all_z_values.append(item["z"])
            if item["is_target"]:
                valid_targets.append(item)
                if 0.0 < item["z"] < best_z:
                    best_z = item["z"]
                    best = item

        return True, {
            "detections": detections_for_vis,
            "best": best,
            "valid_targets": valid_targets,
            "all_z_values": all_z_values,
            "det_result": det_result,
            "target_label": f"{class_name} (brick live0-style)",
        }

    def run_single_frame_component_by_id(self, target_id, visualize=True, wait_ms=5000):
        class_name = ID_TO_CLASS.get(int(target_id))
        if class_name is None:
            return PoseResult(False, target_id=target_id, reason=f"unknown component id: {target_id}")

        # live_view_id=777과 동일하게 component YOLO model + component orientation 분석을 사용한다.
        # 단, is_target 판정만 서비스로 들어온 component id에 맞춰 수행한다.
        target_mode = "component_target"
        target_key = self._normalize_class_name(class_name)
        self._log_info(f"single-frame component search: id={target_id}, class={class_name}")

        try:
            ok, vis_payload = self._process_one_frame(target_id, target_mode, target_key)
        except Exception as exc:
            self._log_warn(f"single-frame component failed: {exc}")
            return PoseResult(False, target_id=target_id, class_name=class_name, reason=str(exc))

        result = self._pose_result_from_payload(target_id, class_name, vis_payload if ok else None)
        if visualize and ok:
            self.show_visualization(
                vis_payload["det_result"],
                vis_payload["detections"],
                vis_payload["target_label"],
                best=vis_payload.get("best"),
                result=result,
                wait_ms=wait_ms,
                close_after=True,
            )
        return result

    def _pose_result_from_payload(self, target_id, class_name, vis_payload):
        if vis_payload is None:
            return PoseResult(False, target_id=target_id, class_name=class_name, reason="frame processing failed")

        valid_targets = vis_payload.get("valid_targets", [])
        if not valid_targets:
            return PoseResult(False, target_id=target_id, class_name=class_name, reason="target not detected in this frame")

        # 같은 class가 여러 개 보이면 가까운 객체를 반환한다.
        best_target = min(valid_targets, key=lambda item: item["z"] if item["z"] > 0.0 else float("inf"))
        if best_target["z"] <= 0.0:
            return PoseResult(False, target_id=target_id, class_name=class_name, reason="invalid depth")

        x_m, y_m, z_m = rs.rs2_deproject_pixel_to_point(
            self.intrinsics,
            [int(best_target["u"]), int(best_target["v"])],
            float(best_target["z"]),
        )

        all_z_values = vis_payload.get("all_z_values", [])
        layer = None
        if all_z_values:
            floor_z = max(all_z_values)
            layer = int(round((floor_z - float(best_target["z"])) / 0.016)) + 1

        result = PoseResult(
            True,
            target_id=int(target_id),
            class_name=str(best_target.get("detected_class", class_name)),
            x_m=float(x_m),
            y_m=float(y_m),
            z_m=float(z_m),
            yaw_deg=float(best_target.get("yaw", 0.0)),
            layer=layer,
        )
        self._log_info(
            f"single-frame result: id={target_id}, class={result.class_name}, "
            f"x={result.x_m*1000.0:.1f}mm, y={result.y_m*1000.0:.1f}mm, "
            f"z={result.z_m*1000.0:.1f}mm, yaw={result.yaw_deg:.1f}deg"
        )
        return result

    def show_live_frame(self, target_id=0):
        """수동 live view용 함수.

        서비스 경로는 이 함수를 직접 호출하지 않는다.
        그래도 디버깅을 위해 target_id=777을 넣으면 기존처럼 전체 component 축을 볼 수 있다.
        """
        try:
            target_id = int(target_id)
        except Exception:
            target_id = 0
        target_mode = self._get_target_mode(target_id)
        class_name = ID_TO_CLASS.get(target_id, "all")
        target_key = None if target_mode in {"all_bricks", "all_components"} else self._normalize_class_name(class_name)
        ok, vis_payload = self._process_one_frame(target_id, target_mode, target_key)
        if not ok:
            return False
        if self.visualize:
            self.show_visualization(
                vis_payload["det_result"],
                vis_payload["detections"],
                vis_payload["target_label"],
                best=vis_payload["best"],
                wait_ms=1,
                close_after=False,
            )
        return True

    def _center_from_mask_points(self, mask_pts, shape_hw):
        if mask_pts is None or len(mask_pts) < 3:
            return None
        mask_u8, contour = self.mask_points_to_mask_and_contour(mask_pts, shape_hw)
        if contour is None:
            return None
        return self.contour_centroid(contour)

    def _process_one_frame(self, target_id, target_mode, target_key):
        frames = self.pipeline.wait_for_frames(timeout_ms=500)
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not color_frame or not depth_frame:
            return False, None

        image = np.asanyarray(color_frame.get_data())
        det_result, seg_result = self._infer_for_mode(target_mode, image)
        if det_result.boxes is None:
            return False, None

        pre_candidates = []
        detections_for_vis = []

        for det_idx, box in enumerate(det_result.boxes):
            cls_name = det_result.names[int(box.cls[0])]
            cls_key = self._normalize_class_name(cls_name)

            if target_mode in {"all_components", "component_target"} and not self._is_allowed_component_class(cls_name):
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            bbox_u = int((xyxy[0] + xyxy[2]) / 2)
            bbox_v = int((xyxy[1] + xyxy[3]) / 2)
            u, v = bbox_u, bbox_v

            is_target = self._is_target_candidate(target_mode, target_key, cls_name, cls_key)
            mask_pts = self.get_matching_mask_points(det_result, seg_result, det_idx, bbox_u, bbox_v)

            axis_info = None
            if mask_pts is not None:
                center = self._center_from_mask_points(mask_pts, image.shape[:2])
                if center is not None:
                    # x/y/z 반환 기준점을 YOLO bbox 중심이 아니라 segmentation contour 중심으로 둔다.
                    u, v = int(center[0]), int(center[1])

                edge_info = self.get_mask_edge_contact_info(mask_pts, image.shape, self.edge_contact_margin_px)
                if edge_info["max_px"] > self.edge_contact_max_px:
                    detections_for_vis.append({
                        "u": u, "v": v, "z": 0.0, "yaw": 0.0,
                        "class_name": f"{cls_name}_edge{edge_info['max_px']}px",
                        "is_target": False,
                        "axis_info": None,
                    })
                    continue

                if target_mode == "brick_target" and self.use_shape_ratio_filter:
                    shape_ok, ratio = self.brick_shape_ratio_pass(cls_name, mask_pts)
                    if not shape_ok:
                        detections_for_vis.append({
                            "u": u, "v": v, "z": 0.0, "yaw": 0.0,
                            "class_name": f"{cls_name}_shape_r{ratio:.2f}",
                            "is_target": False,
                            "axis_info": None,
                        })
                        continue

                if target_mode in {"component_target", "all_components"}:
                    axis_info = self.analyze_component_orientation(image, mask_pts, cls_name)

            z = self.get_valid_depth(depth_frame, u, v)
            if z <= 0.0:
                detections_for_vis.append({
                    "u": u, "v": v, "z": z, "yaw": 0.0,
                    "class_name": str(cls_name), "is_target": is_target, "axis_info": axis_info,
                })
                continue

            yaw = 0.0
            if target_mode in {"component_target", "all_components"} and axis_info is not None:
                yaw = float(axis_info.get("angle_deg", 0.0))
            elif mask_pts is not None:
                yaw = self.find_yaw_from_mask_points(mask_pts)

            pre_candidates.append({
                "u": u,
                "v": v,
                "z": z,
                "yaw": yaw,
                "detected_class": str(cls_name),
                "class_key": cls_key,
                "is_target": is_target,
                "axis_info": axis_info,
                "mask_pts": mask_pts,
            })

        depth_median = None
        if self.use_depth_median_filter:
            valid_depths = [item["z"] for item in pre_candidates if item["z"] > 0.0]
            if len(valid_depths) >= self.depth_median_min_samples:
                depth_median = float(np.median(np.array(valid_depths, dtype=float)))

        valid_targets = []
        all_z_values = []
        best = None
        best_z = float("inf")

        for item in pre_candidates:
            cls_name = item["detected_class"]
            depth_diff = 0.0 if depth_median is None else abs(item["z"] - depth_median)
            if (depth_median is not None) and (depth_diff > self.depth_median_margin_m):
                detections_for_vis.append({
                    "u": item["u"], "v": item["v"], "z": item["z"], "yaw": item["yaw"],
                    "class_name": f"{cls_name}_depth{depth_diff*1000.0:.0f}mm",
                    "is_target": False,
                    "axis_info": item.get("axis_info"),
                })
                continue

            detections_for_vis.append({
                "u": item["u"], "v": item["v"], "z": item["z"], "yaw": item["yaw"],
                "class_name": str(cls_name),
                "is_target": item["is_target"],
                "axis_info": item.get("axis_info"),
            })
            all_z_values.append(item["z"])
            if item["is_target"]:
                valid_targets.append(item)
                if 0.0 < item["z"] < best_z:
                    best_z = item["z"]
                    best = item

        target_label = self._target_label_for_mode(target_id, target_mode)
        return True, {
            "detections": detections_for_vis,
            "best": best,
            "valid_targets": valid_targets,
            "all_z_values": all_z_values,
            "det_result": det_result,
            "target_label": target_label,
        }

    def _infer_for_mode(self, target_mode, image):
        if target_mode in {"component_target", "all_components"}:
            result = self.model_comp(image, verbose=False)[0]
            return result, result
        det_result = self.model_det(image, verbose=False)[0]
        if self.model_seg is self.model_det:
            seg_result = det_result
        else:
            seg_result = self.model_seg(image, verbose=False)[0]
        return det_result, seg_result

    def _get_target_mode(self, target_id):
        if target_id == 0:
            return "all_bricks"
        if target_id == COMPONENT_VIEW_ID:
            return "all_components"
        if target_id in BRICK_IDS:
            return "brick_target"
        if target_id in COMPONENT_IDS:
            return "component_target"
        return "default_target"

    def _target_label_for_mode(self, target_id, target_mode):
        if target_mode == "all_bricks":
            return "all bricks"
        if target_mode == "all_components":
            return "all components"
        return ID_TO_CLASS.get(target_id, str(target_id))

    def _is_target_candidate(self, target_mode, target_key, cls_name, cls_key):
        if target_mode in {"all_bricks", "all_components"}:
            return True
        if target_key is None:
            return False
        if target_mode == "component_target":
            # 모델 class 표기가 small tree / small_tree / smalltree처럼 조금 달라도 잡히도록
            # normalize 비교와 compact 비교를 같이 수행한다.
            return (
                self._target_matches(target_key, cls_key) or
                self._target_matches(self._compact_class_name(target_key), self._compact_class_name(cls_name))
            )
        return self._target_matches(target_key, cls_key)

    def get_matching_mask_points(self, det_result, seg_result, det_idx, target_u, target_v):
        if det_result is not None and getattr(det_result, "masks", None) is not None:
            if len(det_result.masks.xy) > det_idx:
                pts = np.asarray(det_result.masks.xy[det_idx], dtype=np.float32)
                if pts.shape[0] >= 3:
                    return pts
        if seg_result is None or seg_result.masks is None or seg_result.boxes is None:
            return None
        min_dist = float("inf")
        best_mask_pts = None
        for idx, seg_box in enumerate(seg_result.boxes):
            xyxy = seg_box.xyxy[0].cpu().numpy()
            seg_u = int((xyxy[0] + xyxy[2]) / 2)
            seg_v = int((xyxy[1] + xyxy[3]) / 2)
            dist = ((target_u - seg_u) ** 2 + (target_v - seg_v) ** 2) ** 0.5
            if dist < self.match_distance_px and dist < min_dist:
                min_dist = dist
                if len(seg_result.masks.xy) > idx:
                    pts = np.asarray(seg_result.masks.xy[idx], dtype=np.float32)
                    if pts.shape[0] >= 3:
                        best_mask_pts = pts
        return best_mask_pts

    def find_yaw_from_mask_points(self, mask_pts):
        if mask_pts is None or len(mask_pts) < 3:
            return 0.0
        rect = cv2.minAreaRect(np.int32(mask_pts))
        return self.calculate_refined_yaw(rect)

    @staticmethod
    def calculate_refined_yaw(rect):
        (_, _), (width, height), angle = rect
        if width < height:
            yaw = angle
        else:
            yaw = angle + 90.0
        if yaw > 90.0:
            yaw -= 180.0
        if yaw < -90.0:
            yaw += 180.0
        return float(yaw)

    def brick_shape_ratio_pass(self, cls_name, mask_pts):
        ratio = 1.0
        if mask_pts is None or len(mask_pts) < 3:
            return True, ratio
        rect = cv2.minAreaRect(np.int32(mask_pts))
        (_, _), (w, h), _ = rect
        short_side = max(min(w, h), 1e-6)
        long_side = max(w, h)
        ratio = float(long_side / short_side)
        cls_key = self._compact_class_name(cls_name)
        if cls_key.startswith("2x2"):
            return ratio <= self.shape_ratio_threshold, ratio
        if cls_key.startswith("4x2"):
            return ratio >= self.shape_ratio_threshold, ratio
        return True, ratio

    def get_mask_edge_contact_info(self, mask_pts, image_shape, margin_px=2):
        h, w = image_shape[:2]
        canvas = np.zeros((h, w), dtype=np.uint8)
        poly = np.int32(mask_pts).reshape(-1, 1, 2)
        cv2.fillPoly(canvas, [poly], 255)
        info = {"left": 0, "right": 0, "top": 0, "bottom": 0, "max_px": 0}
        if margin_px <= 0:
            return info
        left_strip = canvas[:, :margin_px]
        right_strip = canvas[:, max(0, w - margin_px):]
        top_strip = canvas[:margin_px, :]
        bottom_strip = canvas[max(0, h - margin_px):, :]
        info["left"] = self._occupied_span_px(np.where(np.any(left_strip > 0, axis=1))[0])
        info["right"] = self._occupied_span_px(np.where(np.any(right_strip > 0, axis=1))[0])
        info["top"] = self._occupied_span_px(np.where(np.any(top_strip > 0, axis=0))[0])
        info["bottom"] = self._occupied_span_px(np.where(np.any(bottom_strip > 0, axis=0))[0])
        info["max_px"] = max(info["left"], info["right"], info["top"], info["bottom"])
        return info

    @staticmethod
    def _occupied_span_px(indices):
        if indices is None or len(indices) == 0:
            return 0
        return int(indices.max() - indices.min() + 1)

    def analyze_component_orientation(self, image_bgr, mask_pts, cls_name):
        compact_key = self._compact_class_name(cls_name)
        rule = COMPONENT_COLOR_RULES.get(compact_key, {"axis": "major", "color": None})

        mask_u8, contour = self.mask_points_to_mask_and_contour(mask_pts, image_bgr.shape[:2])
        if contour is None or len(contour) < 5:
            return None

        centroid = self.contour_centroid(contour)
        pca_info = self.compute_pca_axes(contour, centroid)
        if pca_info is None:
            return None

        # PCA 주축/단축은 계속 시각화용 기준선으로 남긴다.
        # 단, component yaw 반환축은 아래 special/color-center 로직에서 다시 정한다.
        axis_info = {
            "centroid": centroid,
            "major_pos": pca_info["major_pos"],
            "major_neg": pca_info["major_neg"],
            "minor_pos": pca_info["minor_pos"],
            "minor_neg": pca_info["minor_neg"],
            "chosen_axis": rule.get("axis", "major"),
            "top_color": rule.get("color"),
            "angle_deg": 0.0,
            "selected_pos": pca_info["major_pos"],
            "selected_neg": pca_info["major_neg"],
            "note": compact_key,
        }

        if rule.get("special") == "tree":
            special = self.analyze_tree_orientation(image_bgr, mask_u8, contour, compact_key)
            if special is not None:
                axis_info.update(special)
                return axis_info

        if rule.get("special") == "burger":
            special = self.analyze_burger_orientation(image_bgr, mask_u8, contour, centroid, pca_info)
            if special is not None:
                axis_info.update(special)
                return axis_info

        top_color = rule.get("color", None)
        if top_color is not None:
            color_axis = self.analyze_top_color_center_orientation(
                image_bgr=image_bgr,
                object_mask=mask_u8,
                contour=contour,
                compact_key=compact_key,
                top_color=top_color,
            )
            if color_axis is not None:
                axis_info.update(color_axis)
                return axis_info

        # fallback: top_color가 없거나 HSV 마스크가 실패한 경우에만 예전 PCA endpoint 방식 사용.
        axis_name = rule.get("axis", "major")
        pref_color = rule.get("color", None)
        if axis_name == "minor":
            pos_pt = pca_info["minor_pos"]
            neg_pt = pca_info["minor_neg"]
        else:
            pos_pt = pca_info["major_pos"]
            neg_pt = pca_info["major_neg"]

        pos_scores = self.endpoint_color_scores(image_bgr, mask_u8, pos_pt)
        neg_scores = self.endpoint_color_scores(image_bgr, mask_u8, neg_pt)
        choose_positive = self.choose_endpoint_by_color(pos_scores, neg_scores, pref_color, centroid, pos_pt, neg_pt)
        selected_pt = pos_pt if choose_positive else neg_pt
        direction = np.array(selected_pt, dtype=np.float32) - np.array(centroid, dtype=np.float32)

        axis_info.update({
            "selected_pos": tuple(map(int, selected_pt)),
            "selected_neg": tuple(map(int, centroid)),
            "angle_deg": self.vector_to_clock_angle(direction),
            "pos_scores": pos_scores,
            "neg_scores": neg_scores,
            "chosen_axis": f"{axis_name}_fallback",
            "top_color": pref_color,
            "note": f"{compact_key}_pca_fallback",
        })
        return axis_info

    def analyze_top_color_center_orientation(self, image_bgr, object_mask, contour, compact_key, top_color):
        """Top-color 영역 중심 기반 component 방향 추정.

        YOLO segmentation 내부에서 지정된 HSV top_color 영역을 마스킹하고,
        해당 color contour마다 minAreaRect를 친 뒤 가장 강한 영역을 고른다.
        반환축은 object 무게중심 -> 선택 color minAreaRect 중심이다.
        """
        object_center = np.array(self.contour_centroid(contour), dtype=np.float32)

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        color_mask = self.extract_color_mask(hsv, top_color)
        color_mask = cv2.bitwise_and(color_mask, color_mask, mask=object_mask)

        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        cnts, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        best = None
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 8:
                continue

            rect = cv2.minAreaRect(cnt)
            rect_center = np.array(rect[0], dtype=np.float32)
            (_, _), (w, h), _ = rect
            rect_area = max(float(w * h), 1e-6)
            fill_ratio = float(area / rect_area)

            # 색 영역이 크고 minAreaRect 내부를 잘 채우는 contour를 우선한다.
            # 객체 중심에서 너무 가까운 잡음은 dist_score에서 자연스럽게 약해진다.
            dist_score = float(np.linalg.norm(rect_center - object_center))
            score = float(area) * (0.75 + 0.25 * min(fill_ratio, 1.0)) + dist_score * 0.15

            box = cv2.boxPoints(rect).astype(np.float32)
            candidate = {
                "cnt": cnt,
                "rect": rect,
                "box": box,
                "rect_center": rect_center,
                "area": float(area),
                "fill_ratio": fill_ratio,
                "dist_score": dist_score,
                "score": score,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

        if best is None:
            return None

        top_center = best["rect_center"]
        top_vec = top_center - object_center
        if np.linalg.norm(top_vec) < 1e-6:
            return None

        angle = self.vector_to_clock_angle(top_vec)

        return {
            "selected_pos": tuple(np.round(object_center + top_vec).astype(int)),
            "selected_neg": tuple(np.round(object_center).astype(int)),
            "angle_deg": angle,
            "top_color": top_color,
            "chosen_axis": "top_color_center",
            "note": f"{compact_key}_{top_color}A{best['area']:.0f}_fill{best['fill_ratio']:.2f}",
            "top_color_center": tuple(np.round(top_center).astype(int)),
            "top_color_rect": best["box"].astype(int).tolist(),
        }

    def analyze_tree_orientation(self, image_bgr, object_mask, contour, compact_key):
        """smalltree/bigtree 방향 추정.

        YOLO segmentation 전체 contour의 무게중심에서 노랑 영역의 중심으로 향하는
        축을 먼저 만든다. 노랑은 밑부분이라고 보고, 그 축의 반대 방향을
        top 방향으로 반환한다.
        """
        object_center = np.array(self.contour_centroid(contour), dtype=np.float32)

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        yellow_mask = self.extract_color_mask(hsv, "yellow")
        yellow_mask = cv2.bitwise_and(yellow_mask, yellow_mask, mask=object_mask)

        cnts, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        cnt = max(cnts, key=cv2.contourArea)
        yellow_area = cv2.contourArea(cnt)
        if yellow_area < 10:
            return None

        yellow_center = np.array(self.contour_centroid(cnt), dtype=np.float32)
        bottom_vec = yellow_center - object_center
        if np.linalg.norm(bottom_vec) < 1e-6:
            return None

        top_vec = -bottom_vec
        angle = self.vector_to_clock_angle(top_vec)

        return {
            "selected_pos": tuple(np.round(object_center + top_vec).astype(int)),
            "selected_neg": tuple(np.round(object_center).astype(int)),
            "angle_deg": angle,
            "top_color": "yellow(bottom->invert)",
            "chosen_axis": "tree_yellow_center",
            "note": f"{compact_key}_yellowA{yellow_area:.0f}",
            "yellow_center": tuple(np.round(yellow_center).astype(int)),
        }

    def analyze_burger_orientation(self, image_bgr, object_mask, contour, centroid, pca_info):
        """burger 방향 추정.

        이번 버전에서는 최외곽 edge를 축으로 쓰지 않는다.
        1) YOLO object mask 안에서 red 영역은 제거하고 yellow 영역만 남긴다.
        2) yellow contour마다 minAreaRect를 친다.
        3) fill ratio와 긴 변(long edge)의 직선성을 보고 가장 좋은 yellow box를 고른다.
        4) 객체 무게중심 -> 선택된 yellow minAreaRect 중심 방향을 밑단 방향으로 본다.
        5) 그 반대 방향을 top 방향으로 반환한다.
        """
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        red_mask = self.extract_color_mask(hsv, "red")
        yellow_mask = self.extract_color_mask(hsv, "yellow")

        yellow_mask = cv2.bitwise_and(yellow_mask, yellow_mask, mask=object_mask)
        yellow_mask[red_mask > 0] = 0

        kernel = np.ones((3, 3), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        cnts, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        obj_center = np.array(centroid, dtype=np.float32)
        candidates = []

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 10:
                continue

            rect = cv2.minAreaRect(cnt)
            rect_center = np.array(rect[0], dtype=np.float32)
            (_, _), (w, h), _ = rect
            rect_area = max(float(w * h), 1e-6)
            fill_ratio = float(area / rect_area)

            box = cv2.boxPoints(rect).astype(np.float32)
            long_edges = self.get_long_rect_edges(box)
            if not long_edges:
                continue

            # 버거에서 단축이 잡히는 문제를 피하기 위해,
            # 최외곽 edge가 아니라 minAreaRect의 긴 변 직선성만 점수화한다.
            straight_score = max(
                self.score_contour_edge_straightness(cnt, p1, p2)
                for p1, p2, _mid in long_edges
            )

            # 두 yellow box 중 실제 밑단 쪽은 보통 객체 중심에서 더 멀리 떨어진다.
            # 단, 최종 축은 edge가 아니라 rect_center를 사용한다.
            dist_score = float(np.linalg.norm(rect_center - obj_center))
            score = (fill_ratio * 2.0) + straight_score + (dist_score * 0.002)

            candidates.append({
                "cnt": cnt,
                "rect": rect,
                "box": box,
                "rect_center": rect_center,
                "fill_ratio": fill_ratio,
                "straight_score": straight_score,
                "score": score,
                "dist_score": dist_score,
            })

        if not candidates:
            return None

        # 우선 조건: minAreaBox 내부 yellow 점유율 95% 이상.
        strict_candidates = [c for c in candidates if c["fill_ratio"] >= 0.95]
        if strict_candidates:
            best = max(strict_candidates, key=lambda c: c["score"])
        else:
            # HSV/seg 경계 때문에 0.95가 살짝 안 나오는 경우를 위한 fallback.
            # note에 fill 값을 계속 표시해서 현장에서 threshold를 조절할 수 있게 한다.
            best = max(candidates, key=lambda c: c["score"])

        yellow_box_center = best["rect_center"]
        bottom_vec = yellow_box_center - obj_center
        if np.linalg.norm(bottom_vec) < 1e-6:
            return None

        top_vec = -bottom_vec
        angle = self.vector_to_clock_angle(top_vec)

        return {
            "selected_pos": tuple(np.round(obj_center + top_vec).astype(int)),
            "selected_neg": tuple(np.round(obj_center).astype(int)),
            "angle_deg": angle,
            "top_color": "yellow",
            "chosen_axis": "burger_yellow_box_center",
            "note": f"burger_fill{best['fill_ratio']:.2f}_line{best['straight_score']:.2f}",
            "burger_rect": best["box"].astype(int).tolist(),
            "burger_bottom_mid": tuple(np.round(yellow_box_center).astype(int)),
        }

    @staticmethod
    def get_long_rect_edges(box):
        """Return the two long edges of a minAreaRect box.

        Each item is (p1, p2, midpoint). This is used only for scoring
        straightness; the returned burger axis uses the rect center instead.
        """
        edges = []
        for i in range(4):
            p1 = np.array(box[i], dtype=np.float32)
            p2 = np.array(box[(i + 1) % 4], dtype=np.float32)
            mid = 0.5 * (p1 + p2)
            length = float(np.linalg.norm(p2 - p1))
            edges.append((length, p1, p2, mid))
        if not edges:
            return []
        max_len = max(edge[0] for edge in edges)
        return [(p1, p2, mid) for length, p1, p2, mid in edges if length >= max_len * 0.90]

    @staticmethod
    def select_outer_rect_edge(box, object_center):
        """Return the minAreaRect edge whose midpoint is farthest from object center."""
        edges = []
        c = np.array(object_center, dtype=np.float32)
        for i in range(4):
            p1 = np.array(box[i], dtype=np.float32)
            p2 = np.array(box[(i + 1) % 4], dtype=np.float32)
            mid = 0.5 * (p1 + p2)
            dist = float(np.linalg.norm(mid - c))
            edges.append((dist, p1, p2, mid))
        if not edges:
            return None
        _, p1, p2, mid = max(edges, key=lambda item: item[0])
        return p1, p2, mid

    @staticmethod
    def score_contour_edge_straightness(contour, p1, p2, dist_thresh=3.0):
        """Score how strongly contour points lie on the given edge segment.

        1.0에 가까울수록 해당 edge 주변에 contour 픽셀이 길고 조밀하게 분포한다.
        """
        pts = contour.reshape(-1, 2).astype(np.float32)
        p1 = np.array(p1, dtype=np.float32)
        p2 = np.array(p2, dtype=np.float32)
        edge = p2 - p1
        edge_len = float(np.linalg.norm(edge))
        if edge_len < 1e-6 or pts.shape[0] == 0:
            return 0.0

        unit = edge / edge_len
        rel = pts - p1
        proj = rel @ unit
        valid_proj = np.logical_and(proj >= 0.0, proj <= edge_len)
        cross = np.abs(rel[:, 0] * unit[1] - rel[:, 1] * unit[0])
        near = np.logical_and(valid_proj, cross <= dist_thresh)

        if not np.any(near):
            return 0.0

        near_proj = proj[near]
        coverage = float((near_proj.max() - near_proj.min()) / max(edge_len, 1e-6))
        density = float(np.count_nonzero(near) / max(edge_len, 1.0))
        return float(np.clip(0.65 * coverage + 0.35 * min(density, 1.0), 0.0, 1.0))

    def mask_points_to_mask_and_contour(self, mask_pts, shape_hw):
        h, w = shape_hw[:2]
        canvas = np.zeros((h, w), dtype=np.uint8)
        poly = np.int32(mask_pts).reshape(-1, 1, 2)
        cv2.fillPoly(canvas, [poly], 255)
        cnts, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return canvas, None
        contour = max(cnts, key=cv2.contourArea)
        return canvas, contour

    @staticmethod
    def contour_centroid(contour):
        m = cv2.moments(contour)
        if abs(m["m00"]) < 1e-6:
            pts = contour.reshape(-1, 2)
            mean_pt = np.mean(pts, axis=0)
            return int(round(mean_pt[0])), int(round(mean_pt[1]))
        return int(round(m["m10"] / m["m00"])), int(round(m["m01"] / m["m00"]))

    def compute_pca_axes(self, contour, centroid):
        pts = contour.reshape(-1, 2).astype(np.float32)
        if pts.shape[0] < 5:
            return None
        centered = pts - np.array(centroid, dtype=np.float32)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]
        major_vec = eigvecs[:, 0]
        minor_vec = eigvecs[:, 1]
        major_vec = major_vec / (np.linalg.norm(major_vec) + 1e-6)
        minor_vec = minor_vec / (np.linalg.norm(minor_vec) + 1e-6)
        proj_major = centered @ major_vec
        proj_minor = centered @ minor_vec
        major_pos = np.array(centroid, dtype=np.float32) + major_vec * np.max(proj_major)
        major_neg = np.array(centroid, dtype=np.float32) + major_vec * np.min(proj_major)
        minor_pos = np.array(centroid, dtype=np.float32) + minor_vec * np.max(proj_minor)
        minor_neg = np.array(centroid, dtype=np.float32) + minor_vec * np.min(proj_minor)
        return {
            "major_vec": major_vec,
            "minor_vec": minor_vec,
            "major_pos": tuple(np.round(major_pos).astype(int)),
            "major_neg": tuple(np.round(major_neg).astype(int)),
            "minor_pos": tuple(np.round(minor_pos).astype(int)),
            "minor_neg": tuple(np.round(minor_neg).astype(int)),
        }

    def endpoint_color_scores(self, image_bgr, object_mask, point_xy, radius=8):
        x, y = int(point_xy[0]), int(point_xy[1])
        h, w = object_mask.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        circle = ((xx - x) ** 2 + (yy - y) ** 2) <= (radius ** 2)
        sample_mask = np.logical_and(circle, object_mask > 0)
        if not np.any(sample_mask):
            return {key: 0.0 for key in HSV_COLOR_RANGES.keys()}
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        scores = {}
        total = float(np.count_nonzero(sample_mask))
        for cname in HSV_COLOR_RANGES.keys():
            cmask = self.extract_color_mask(hsv, cname) > 0
            scores[cname] = float(np.count_nonzero(np.logical_and(sample_mask, cmask)) / max(total, 1.0))
        return scores

    def extract_color_mask(self, hsv_image, color_name):
        """H 중심 기반 색 마스크.

        기존처럼 H/S/V 범위를 모두 딱 자르는 방식이 아니라,
        H값의 원형 거리(circular distance)를 기준으로 target color에 가까운 픽셀을 찾는다.
        S는 무채색/흰색/검은색 계열의 H 튐을 막기 위한 최소 guard로만 사용한다.
        V는 조건으로 사용하지 않는다.
        """
        params = HUE_COLOR_PARAMS.get(color_name)
        if params is None:
            return np.zeros(hsv_image.shape[:2], dtype=np.uint8)

        h = hsv_image[:, :, 0].astype(np.int16)
        s = hsv_image[:, :, 1].astype(np.int16)

        center = int(params["center"])
        tol = int(params["tol"])
        min_s = int(params.get("min_s", 0))

        # OpenCV H는 0~179 원형 값이다.
        # red처럼 0/179 경계에 걸친 색도 별도 예외 없이 처리된다.
        diff = np.abs(h - center)
        circular_diff = np.minimum(diff, 180 - diff)

        mask = np.logical_and(circular_diff <= tol, s >= min_s)
        return (mask.astype(np.uint8) * 255)

    def choose_endpoint_by_color(self, pos_scores, neg_scores, preferred_color, centroid, pos_pt, neg_pt):
        if preferred_color is not None:
            pos_val = pos_scores.get(preferred_color, 0.0)
            neg_val = neg_scores.get(preferred_color, 0.0)
            if abs(pos_val - neg_val) > 1e-6:
                return pos_val >= neg_val
        # fallback: 화면 위쪽(y가 작은 쪽) 우선
        return pos_pt[1] <= neg_pt[1]

    @staticmethod
    def vector_to_clock_angle(vec_xy):
        vx, vy = float(vec_xy[0]), float(vec_xy[1])
        if abs(vx) < 1e-6 and abs(vy) < 1e-6:
            return 0.0
        ang = np.degrees(np.arctan2(vx, -vy))
        if ang > 180.0:
            ang -= 360.0
        if ang <= -180.0:
            ang += 360.0
        return float(ang)

    def show_visualization(
        self,
        det_result,
        detections,
        target_class,
        best=None,
        result=None,
        wait_ms=1,
        close_after=False,
    ):
        image = det_result.plot()
        height, width = image.shape[:2]
        cv2.circle(image, (width // 2, height // 2), 5, (0, 0, 255), -1)
        cv2.putText(
            image,
            f"target: {target_class}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        best_u = best["u"] if best is not None else None
        best_v = best["v"] if best is not None else None
        for det in detections:
            u, v = det["u"], det["v"]
            color = (0, 255, 255) if det["is_target"] else (180, 180, 180)
            radius = 7 if det["is_target"] else 4
            if best_u == u and best_v == v:
                color = (0, 0, 255)
                radius = 9
            cv2.circle(image, (u, v), radius, color, -1)

            axis_info = det.get("axis_info")
            if axis_info is not None:
                c = tuple(map(int, axis_info.get("centroid", (u, v))))
                sel_start = tuple(map(int, axis_info.get("selected_neg", c)))
                sel_end = tuple(map(int, axis_info.get("selected_pos", c)))
                cv2.arrowedLine(image, sel_start, sel_end, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.15)

                if "yellow_center" in axis_info:
                    cv2.circle(image, tuple(map(int, axis_info["yellow_center"])), 4, (0, 255, 255), -1)
                if "top_color_center" in axis_info:
                    cv2.circle(image, tuple(map(int, axis_info["top_color_center"])), 5, (0, 255, 255), -1)
                if "top_color_rect" in axis_info:
                    pts = np.array(axis_info["top_color_rect"], dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(image, [pts], True, (0, 255, 255), 2)
                if "burger_rect" in axis_info:
                    pts = np.array(axis_info["burger_rect"], dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(image, [pts], True, (255, 255, 0), 2)
                if "burger_bottom_mid" in axis_info:
                    cv2.circle(image, tuple(map(int, axis_info["burger_bottom_mid"])), 5, (0, 165, 255), -1)

            note = ""
            if axis_info is not None and axis_info.get("note"):
                note = f" {axis_info.get('note')}"

            ratio_note = ""
            if det.get("ratio") is not None:
                ratio_note = f" R:{float(det.get('ratio')):.2f}"

            if det["z"] > 0.0:
                label = f"{det['class_name']} Z:{det['z']*1000.0:.0f} Yaw:{det['yaw']:.1f}{ratio_note}{note}"
            else:
                label = f"{det['class_name']} Z:invalid{ratio_note}{note}"
            cv2.putText(
                image,
                label,
                (max(0, u - 110), min(height - 10, v + 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        if result is not None:
            if result.success:
                result_lines = [
                    f"SERVICE RESULT: id={result.target_id} class={result.class_name}",
                    f"X={result.x_m*1000.0:.1f}mm  Y={result.y_m*1000.0:.1f}mm  Z={result.z_m*1000.0:.1f}mm  Yaw={result.yaw_deg:.1f}deg",
                ]
            else:
                result_lines = [
                    f"SERVICE RESULT: FAILED id={result.target_id} class={result.class_name}",
                    f"Reason: {result.reason}",
                ]
            y0 = max(55, height - 48)
            for idx, line in enumerate(result_lines):
                cv2.putText(
                    image,
                    line,
                    (12, y0 + idx * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 0, 255) if result.success else (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

        if self.visualize_scale != 1.0:
            image = cv2.resize(image, None, fx=self.visualize_scale, fy=self.visualize_scale, interpolation=cv2.INTER_LINEAR)

        cv2.imshow(self.visualize_window, image)

        wait_ms = int(wait_ms) if wait_ms is not None else 1
        if wait_ms <= 1:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self._log_info("OpenCV q/ESC pressed. Stop requested.")
                self.stop_requested = True
        else:
            start = time.time()
            while (time.time() - start) * 1000.0 < wait_ms:
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == 27:
                    self._log_info("OpenCV q/ESC pressed. Stop requested.")
                    self.stop_requested = True
                    break
            if close_after:
                try:
                    cv2.destroyWindow(self.visualize_window)
                    for _ in range(3):
                        cv2.waitKey(1)
                except Exception:
                    pass

    @staticmethod
    def get_valid_depth(depth_frame, u, v, search_radius=10):
        z = depth_frame.get_distance(u, v)
        if z > 0.0:
            return float(z)
        for radius in range(1, search_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nu = u + dx
                    nv = v + dy
                    if 0 <= nu < 640 and 0 <= nv < 480:
                        z = depth_frame.get_distance(nu, nv)
                        if z > 0.0:
                            return float(z)
        return 0.0

    @staticmethod
    def _normalize_class_name(name):
        return str(name).lower().replace(" ", "").replace("-", "_")

    @staticmethod
    def _compact_class_name(name):
        return str(name).lower().replace(" ", "").replace("-", "").replace("_", "")

    @staticmethod
    def _target_matches(target_key, detected_key):
        if target_key == detected_key:
            return True
        return target_key in detected_key or detected_key in target_key

    def _is_allowed_component_class(self, cls_name):
        cls_key = self._normalize_class_name(cls_name)
        if cls_key in self.component_class_keys:
            return True
        return self._compact_class_name(cls_name) in self.component_class_compact_keys

    @staticmethod
    def _majority_class(names):
        counts = {}
        for name in names:
            counts[name] = counts.get(name, 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _check_model_file(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"YOLO model not found: {path}")

    def _log_info(self, message):
        if self.logger is not None:
            self.logger.info(message)

    def _log_warn(self, message):
        if self.logger is not None:
            self.logger.warn(message)
