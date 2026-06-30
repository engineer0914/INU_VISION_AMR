import os
import yaml
import pprint
import glob
import cv2
import math

import torch
import open3d as o3d
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

import pyrealsense2 as rs
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
import copy

from scipy.spatial.transform import Rotation as R
SCIPY_AVAILABLE = True

CAMERA_PROFILES = {
    # 1. 바닥(Floor) 모드: RANSAC 평면 검출용 (넓고 강하게)
    "floor": {
        "preset_id": 4,              # High Density (바닥 구멍 채우기)
        "smooth_alpha": 0.5,
        "smooth_delta": 20,          # 평면을 더 평평하게 다듬기 위해 낮춤
        "min_dist": 0.30,            # 카메라 바로 앞 먼지/노이즈 무시
        "max_dist": 3.00,            # 바닥까지만 보고 불필요한 원거리 컷
        "target_laser_power": 360,   # 바닥 반사율 확보를 위해 최대 파워
        "target_shift": 0,           # 정상 시력 구간
        "roi_percent": 80,           # 화면 전체를 기준으로 노출 계산
        "auto_awb_value": 1,
        "depth_Units": 0.0001
    },
    
    # 2. 근접 30cm 모드: 듀플로 픽킹용 (정밀하고 어둡게)
    "macro_30": {
    "preset_id": 4,              # High Accuracy 계열이면 3 유지, 안 맞으면 4도 테스트
    "smooth_alpha": 0.6,         # 엣지 보존 위해 너무 낮게 하지 않음
    "smooth_delta": 20,          # 50~100은 직각/얇은 부분을 뭉갤 수 있음
    "min_dist": 0.08,
    "max_dist": 0.32,            # 20cm 근처만 보기
    "target_laser_power": 150,    # 너무 가까우면 150도 과할 수 있음
    "target_shift": 20,          # 20cm 근거리용 시작값
    "roi_percent": 15,           # 중앙 객체 기준 AE
    "auto_awb_value": 1,
    "depth_Units": 0.00001       # 근거리 전용. 불안정하면 0.0001로 복귀
    },
    
    # 3. 원거리 50cm 모드: 접근 및 탐색용 (밸런스형)
    "mid_50": {
        "preset_id": 4,              # High Density
        "smooth_alpha": 0.5,
        "smooth_delta": 50,
        "min_dist": 0.10,
        "max_dist": 0.80,            # 작업대(테이블) 영역 정도까지만 컷
        "target_laser_power": 250,   # 너무 세지도 약하지도 않은 중간 파워
        "target_shift": 0,           # 50cm는 기본 시력 구간에 포함됨 (Shift 불필요)
        "roi_percent": 40,           # 작업 영역인 중앙 40% 기준 노출
        "auto_awb_value": 1,
        "depth_Units": 0.0001        
    }
}

# 카메라 설정 함수들

def load_rgb_calibration_from_folder(
    calib_folder,
    yaml_name=None,
    alpha=0,
    make_undistort_map=True
):
    """
    OpenCV RGB 캘리브레이션 YAML을 폴더 경로에서 불러오는 함수.

    Parameters
    ----------
    calib_folder : str
        YAML 파일이 들어있는 폴더 경로.
    yaml_name : str or None
        특정 YAML 파일명을 지정하고 싶을 때 사용.
        None이면 폴더 안의 .yaml 또는 .yml 파일 중 첫 번째를 사용.
    alpha : float
        cv2.getOptimalNewCameraMatrix의 alpha.
        0: 검은 영역 최소화 / crop 느낌
        1: 시야 최대 보존
    make_undistort_map : bool
        True이면 remap용 map1, map2까지 생성.

    Returns
    -------
    calib : dict
        {
            "image_width": int,
            "image_height": int,
            "checkerboard_inner_corners": dict,
            "square_size_mm": float,
            "rms_reprojection_error": float,
            "K": np.ndarray,
            "D": np.ndarray,
            "new_K": np.ndarray or None,
            "roi": tuple or None,
            "map1": np.ndarray or None,
            "map2": np.ndarray or None,
            "yaml_path": str
        }
    """

    calib_folder = os.path.abspath(calib_folder)

    if not os.path.isdir(calib_folder):
        raise FileNotFoundError(f"폴더가 없습니다: {calib_folder}")

    # YAML 파일 찾기
    if yaml_name is not None:
        yaml_path = os.path.join(calib_folder, yaml_name)
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"YAML 파일이 없습니다: {yaml_path}")
    else:
        yaml_files = sorted(
            glob.glob(os.path.join(calib_folder, "*.yaml")) +
            glob.glob(os.path.join(calib_folder, "*.yml"))
        )

        if len(yaml_files) == 0:
            raise FileNotFoundError(f"YAML 파일을 찾지 못했습니다: {calib_folder}")

        yaml_path = yaml_files[0]

    # YAML 로드
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 필수 값 확인
    required_keys = [
        "image_width",
        "image_height",
        "camera_matrix",
        "dist_coeffs"
    ]

    for key in required_keys:
        if key not in data:
            raise KeyError(f"YAML에 '{key}' 항목이 없습니다: {yaml_path}")

    w = int(data["image_width"])
    h = int(data["image_height"])

    K = np.array(data["camera_matrix"], dtype=np.float64)
    D = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)

    new_K = None
    roi = None
    map1 = None
    map2 = None

    if make_undistort_map:
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            K,
            D,
            (w, h),
            alpha,
            (w, h)
        )

        map1, map2 = cv2.initUndistortRectifyMap(
            K,
            D,
            None,
            new_K,
            (w, h),
            cv2.CV_32FC1
        )

    calib = {
        "image_width": w,
        "image_height": h,
        "checkerboard_inner_corners": data.get("checkerboard_inner_corners", None),
        "square_size_mm": data.get("square_size_mm", None),
        "rms_reprojection_error": data.get("rms_reprojection_error", None),
        "K": K,
        "D": D,
        "new_K": new_K,
        "roi": roi,
        "map1": map1,
        "map2": map2,
        "yaml_path": yaml_path,
    }

    print("[CALIB LOADED]")
    print("yaml_path:", yaml_path)
    print("image_size:", (w, h))
    print("rms:", calib["rms_reprojection_error"])
    print("K:\n", K)
    print("D:", D.ravel())

    if new_K is not None:
        print("new_K:\n", new_K)
        print("roi:", roi)

    return calib

def get_realsense_ids():
    """
    연결된 모든 리얼센스 카메라의 이름과 시리얼 번호(ID)를 딕셔너리 형태로 반환합니다.
    """
    connected_devices = {}
    
    # 리얼센스 컨텍스트 생성 (연결된 기기들을 관리)
    ctx = rs.context()
    
    # 연결된 기기가 없는 경우 예외 처리
    if len(ctx.devices) == 0:
        print("연결된 리얼센스 카메라를 찾을 수 없습니다.")
        return connected_devices

    # 연결된 모든 기기를 순회하며 정보 추출
    for dev in ctx.devices:
        name = dev.get_info(rs.camera_info.name)
        serial_number = dev.get_info(rs.camera_info.serial_number)
        
        # 시리얼 번호를 키(Key)로, 모델명을 값(Value)으로 저장
        connected_devices[serial_number] = name
        
    return connected_devices

def configure_realsense(
        serial_number=None,
        preset_id=4, 
        smooth_alpha=0.5, 
        smooth_delta=50, 
        min_dist=0.15, 
        max_dist=2.0, 
        target_laser_power = 150, 
        target_shift = 0, 
        roi_percent=80, 
        auto_awb_value=1,
        depth_Units=0.0001,
        visualize=True
        ):
    
    pipeline = rs.pipeline()
    config = rs.config()

    if serial_number:
        config.enable_device(serial_number)

    # 1. 스트림 해상도 설정
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # 파이프라인 시작
    profile = pipeline.start(config)
    dev = profile.get_device()
    advnc_mode = rs.rs400_advanced_mode(dev)
    
    # 2. 센서 객체 가져오기
    depth_sensor = dev.first_depth_sensor()
    
    # 💡 추가해야 할 부분!
    color_sensor = dev.first_color_sensor()
    
    # 3. 비주얼 프리셋 설정
    if depth_sensor.supports(rs.option.visual_preset):
        depth_sensor.set_option(rs.option.visual_preset, preset_id)
        if visualize:
            print(f"✅ Preset 설정 완료: {depth_sensor.get_option_value_description(rs.option.visual_preset, preset_id)}")

    # 4. 뎁스 유닛(Depth Unit) 설정
    if depth_sensor.supports(rs.option.depth_units):
        try:
            depth_sensor.set_option(rs.option.depth_units, depth_Units)
            if visualize:
                print(f"✅ Depth Unit 설정 완료: {depth_sensor.get_option(rs.option.depth_units)}")
        except Exception as e:
            if visualize:
                print(f"⚠️ Depth Unit 설정 실패: {e}")

    # 5. 자동 노출(Auto Exposure) 활성화 및 ROI 설정
    if depth_sensor.supports(rs.option.enable_auto_exposure):
        depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
        if visualize:
            print("✅ 뎁스 센서 자동 노출(AE) 스위치 ON")

    roi_sensor = rs.roi_sensor(depth_sensor)
    if roi_sensor:
        # 비율을 기반으로 여백(margin) 계산 (예: 80% -> 남은 20%를 반으로 나눠 0.1의 여백 생성)
        # 20: 초근접 중앙 객체
        # 40: 일반적인 중앙 영역
        # 80: 넓은 중앙 영역
        margin = (100 - roi_percent) / 2.0 / 100.0        
        roi = rs.region_of_interest()
        # 해상도 848x480 기준 중앙 영역 동적 계산
        roi.min_x = int(848 * margin)
        roi.max_x = int(848 * (1.0 - margin))
        roi.min_y = int(480 * margin)
        roi.max_y = int(480 * (1.0 - margin))
        
        # 기기에 ROI 세팅 적용
        roi_sensor.set_region_of_interest(roi)
        if visualize:
            print(f"✅ Depth ROI 영역 설정 완료 (중앙 {roi_percent}% / X: {roi.min_x}~{roi.max_x}, Y: {roi.min_y}~{roi.max_y})")

    # 레이저 파워 설정
    # 추천값 -> 60cm 이상 바닥: 360 (최대치) / 30cm 코앞: 약 150
    if depth_sensor.supports(rs.option.laser_power):
        depth_sensor.set_option(rs.option.laser_power, target_laser_power)
        if visualize:
            print(f"✅ 레이저 파워 설정 완료: {target_laser_power}")

    # 🎯 디스패리티 시프트 설정 (고급 모드 사용)
    # 추천값 -> 60cm 이상 바닥: 0 (기본값) / 30cm 코앞: 50 ~ 100 사이 조절
    if advnc_mode.is_enabled():
        depth_table = advnc_mode.get_depth_table()
        depth_table.disparityShift = target_shift
        advnc_mode.set_depth_table(depth_table)
        if visualize:
            print(f"✅ Disparity Shift 설정 완료: {target_shift}")

    # 6. Temporal Filter 설정
    temp_filter = rs.temporal_filter()
    # alpha 값을 낮출수록(예: 0.1~0.2) 이전 프레임의 데이터를 더 오래, 무겁게 기억합니다.
    temp_filter.set_option(rs.option.filter_smooth_alpha, smooth_alpha)
    # delta 값을 높일수록 노이즈를 깎지 않고 찰흙처럼 뭉개버립니다.
    temp_filter.set_option(rs.option.filter_smooth_delta, smooth_delta)
    # # hole filling 옵션: 누적된 데이터로 구멍을 강제로 메웁니다 (0~8)
    # temp_filter.set_option(rs.option.holes_fill, 3)
    if visualize:
        print(f"✅ Temporal Filter 설정 완료")

    # ==========================================
    # 💡 7. Threshold Filter (거리 제한) 설정
    # ==========================================
    thres_filter = rs.threshold_filter()
    thres_filter.set_option(rs.option.min_distance, min_dist)
    thres_filter.set_option(rs.option.max_distance, max_dist)
    if visualize:
        print(f"✅ Threshold Filter 설정 완료 (최소: {min_dist}m, 최대: {max_dist}m)")


    # 🎯 컬러 센서 자동 화이트 밸런스(AWB) 설정
    if color_sensor.supports(rs.option.enable_auto_white_balance):
        color_sensor.set_option(rs.option.enable_auto_white_balance, auto_awb_value)
        if visualize:
            print(f"✅ 컬러 센서 자동 화이트 밸런스(AWB) {'ON' if auto_awb_value == 1 else 'OFF'}")

    # # (선택) 수동으로 색온도를 고정하고 싶을 때의 예시
    # if color_sensor.supports(rs.option.enable_auto_white_balance):
    #     color_sensor.set_option(rs.option.enable_auto_white_balance, 0) # 자동 끄기        
    #     color_sensor.set_option(rs.option.white_balance, 4600)
    #     print("✅ 컬러 센서 화이트 밸런스 수동 고정 (4600K)")

    # 8. 컬러 화면 기준 정렬(Align) 객체 생성
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 리턴 값에 thres_filter 추가!
    return pipeline, align, temp_filter, thres_filter

# 컨트롤 실행부 함수들
def capture_realsense_data(serial_number, mode="mid_50", warmup_frames=10, visualize=False):
    """
    특정 리얼센스 카메라를 지정한 모드로 켜서 예열한 뒤, 핵심 비전 데이터를 추출하는 함수.
    
    Args:
        serial_number (str): get_realsense_ids()로 찾은 기기 시리얼 번호
        mode (str): "floor", "macro_30", "mid_50" 중 택 1
        warmup_frames (int): 센서 안정화를 위해 버릴 초기 프레임 수
        visualize (bool): 캡처된 결과(Color + Depth)를 Matplotlib으로 출력할지 여부
        
    Returns:
        color_img_rgb (ndarray): RGB 포맷의 컬러 이미지 (YOLO, Open3D용)
        depth_img (ndarray): Raw 뎁스 이미지
        intrinsics (rs.intrinsics): 카메라 내부 파라미터 (3D 투영용)
        depth_scale (float): 뎁스 단위를 미터(m)로 변환하기 위한 스케일 값 (매우 중요)
    """

    def make_depth_colormap_meters(
        depth_img,
        depth_scale,
        min_m=0.08,
        max_m=0.35,
        colormap=cv2.COLORMAP_JET
    ):
        """
        raw depth가 아니라 meter 값 기준으로 컬러맵 생성.
        depth_units가 0.00001이든 0.0001이든 시각화가 일관됨.
        """
        depth_m = depth_img.astype(np.float32) * float(depth_scale)

        valid = (depth_m > min_m) & (depth_m < max_m)

        depth_norm = np.zeros_like(depth_img, dtype=np.uint8)

        if np.count_nonzero(valid) > 0:
            clipped = np.clip(depth_m, min_m, max_m)
            depth_norm[valid] = (
                (clipped[valid] - min_m) / (max_m - min_m) * 255.0
            ).astype(np.uint8)

        depth_colormap = cv2.applyColorMap(depth_norm, colormap)
        depth_colormap_rgb = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)

        # invalid는 검정
        depth_colormap_rgb[~valid] = 0

        return depth_colormap_rgb, depth_m

    print(f"[{mode}] 모드로 카메라(ID: {serial_number}) 구동을 시작합니다...")

    # 1. 프로필 파라미터 로드
    profile_params = CAMERA_PROFILES.get(mode)
    if profile_params is None:
        raise ValueError(f"지원하지 않는 모드입니다: {mode}")
        
    profile_depth_units = profile_params.get("depth_Units", None)
    
    # 2. 카메라 파이프라인 설정 및 구동
    pipeline, align, temp_filter, thres_filter = configure_realsense(
        serial_number=serial_number,
        **profile_params,
        visualize=visualize
    )
    
    intrinsics = None
    color_img = None
    depth_img = None
    depth_scale = None
    
    try:
        # 3. 센서 예열 (안정화)
        if warmup_frames > 0:
            print(f"🔥 센서 안정화 중... ({warmup_frames} 프레임 대기)")
            for _ in range(warmup_frames):
                pipeline.wait_for_frames()

        depth_img, color_img, depth_scale, debug_info = get_aligned_frames_with_units(
            pipeline=pipeline,
            align=align,
            temp_filter=temp_filter,
            thres_filter=thres_filter,
            profile_depth_units=profile_depth_units,
            apply_filter=True
        )
            
        intrinsics = get_aligned_intrinsics(pipeline)
        
        # 5. 시각화 (옵션)
        if visualize and depth_img is not None and color_img is not None:
            # 모드별 가시화 거리 설정
            vis_ranges = {
                "macro_30": (0.08, 0.35),
                "mid_50": (0.15, 0.80),
                "floor": (0.20, 3.00)
            }
            vis_min_m, vis_max_m = vis_ranges.get(mode, (0.20, 1.00))
            
            depth_colormap_rgb, _ = make_depth_colormap_meters(
                depth_img=depth_img, depth_scale=depth_scale, 
                min_m=vis_min_m, max_m=vis_max_m
            )
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            images = np.hstack((color_img, depth_colormap_rgb))
            ax.imshow(images)
            ax.axis("off")
            ax.set_title(f"Capture Result | Mode: {mode} | Scale: {depth_scale:.6f}")
            plt.show()

    except Exception as e:
        print(f"❌ 프레임 캡처 중 에러 발생: {e}")
        
    finally:
        pipeline.stop()
        print("✅ 카메라 스트리밍 안전 종료 완료.")
        
    return color_img, depth_img, intrinsics, depth_scale

def get_aligned_intrinsics(pipeline):
    """
    현재 활성화된 파이프라인에서 정렬된(Aligned) 영상의 인트린직 정보를 반환합니다.
    (뎁스가 컬러에 정렬되므로, 컬러 센서의 인트린직을 반환합니다.)
    """
    # 1. 파이프라인에서 현재 실행 중인 프로필(Profile) 가져오기
    active_profile = pipeline.get_active_profile()
    
    # 2. 컬러 스트림(rs.stream.color) 정보 가져오기
    color_stream = active_profile.get_stream(rs.stream.color)
    
    # 3. 비디오 스트림 프로필로 캐스팅한 뒤 인트린직(Intrinsics) 추출
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    
    return intrinsics

def intrinsics_checker(pipeline):

    intrinsics = get_aligned_intrinsics(pipeline)
    # 인트린직 내부 값 확인하기

    print("\n[카메라 인트린직 정보]")
    print(f"해상도 (Width x Height): {intrinsics.width} x {intrinsics.height}")
    print(f"초점 거리 (fx, fy): {intrinsics.fx:.2f}, {intrinsics.fy:.2f}")
    print(f"주점/중심점 (ppx, ppy): {intrinsics.ppx:.2f}, {intrinsics.ppy:.2f}")
    print(f"왜곡 모델 (Distortion Model): {intrinsics.model}")
    print(f"왜곡 계수 (Coeffs): {intrinsics.coeffs}\n")

    return 0

def get_aligned_frames_with_units(
    pipeline,
    align,
    temp_filter,
    thres_filter,
    profile_depth_units=None,
    apply_filter=True
):
    """
    aligned depth/color 프레임을 받고,
    실제 3D 변환에 사용할 depth_scale까지 같이 반환.

    중요:
    - sensor.get_depth_scale()이 0.0001로 남아 있어도,
      macro_30처럼 profile에서 depth_Units=0.00001을 쓴 경우
      profile_depth_units를 우선 사용한다.
    """
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not aligned_depth_frame or not color_frame:
        return None, None, None, None

    if apply_filter:
        aligned_depth_frame = thres_filter.process(aligned_depth_frame)
        aligned_depth_frame = temp_filter.process(aligned_depth_frame)

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # frame 자체의 unit 확인
    frame_units = None
    center_distance_m = None
    center_raw = None

    try:
        depth_frame_obj = aligned_depth_frame.as_depth_frame()
        frame_units = depth_frame_obj.get_units()

        H, W = depth_image.shape[:2]
        cx = W // 2
        cy = H // 2
        center_raw = int(depth_image[cy, cx])
        center_distance_m = float(depth_frame_obj.get_distance(cx, cy))
    except Exception as e:
        frame_units = None
        center_distance_m = None
        center_raw = None

    # sensor scale 확인
    try:
        depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
        sensor_scale = float(depth_sensor.get_depth_scale())
        sensor_depth_units_option = float(depth_sensor.get_option(rs.option.depth_units))
    except Exception:
        sensor_scale = None
        sensor_depth_units_option = None

    # 핵심:
    # profile_depth_units가 주어지면 그걸 우선 사용.
    # macro_30에서 depth_Units=0.00001을 쓰는 경우 이게 제일 안전.
    if profile_depth_units is not None:
        depth_scale_used = float(profile_depth_units)
    elif frame_units is not None:
        depth_scale_used = float(frame_units)
    elif sensor_scale is not None:
        depth_scale_used = float(sensor_scale)
    else:
        depth_scale_used = 0.001

    debug_info = {
        "frame_units": frame_units,
        "sensor_scale": sensor_scale,
        "sensor_depth_units_option": sensor_depth_units_option,
        "depth_scale_used": depth_scale_used,
        "center_raw": center_raw,
        "center_distance_m": center_distance_m,
        "center_raw_times_used_scale": None if center_raw is None else center_raw * depth_scale_used,
    }

    return depth_image, color_image, depth_scale_used, debug_info


# 조립체 분석용 함수들

def depth_to_xyz_map(depth_img, depth_scale, intrinsics):
    """
    depth_img: uint16 raw depth image, H x W
    depth_scale: raw depth unit -> meter
    intrinsics: pyrealsense2 intrinsics
    return:
        xyz_map: H x W x 3, meter
        valid_mask: H x W, bool
    """
    H, W = depth_img.shape[:2]

    z = depth_img.astype(np.float32) * float(depth_scale)
    valid_mask = z > 0

    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))

    x = (u_grid.astype(np.float32) - intrinsics.ppx) / intrinsics.fx * z
    y = (v_grid.astype(np.float32) - intrinsics.ppy) / intrinsics.fy * z

    xyz_map = np.stack([x, y, z], axis=-1)
    xyz_map[~valid_mask] = 0

    return xyz_map, valid_mask

def fit_plane_ransac_numpy(
    points,
    num_iter=500,
    distance_threshold=0.006,
    sample_size=3,
    max_points=15000,
    random_seed=0
):
    """
    points: N x 3, meter
    plane: ax + by + cz + d = 0
    return:
        best_plane: np.array([a, b, c, d])
        best_inlier_mask: N bool
    """
    rng = np.random.default_rng(random_seed)

    if points.shape[0] > max_points:
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        sample_points = points[idx]
    else:
        sample_points = points

    N = sample_points.shape[0]

    if N < 3:
        raise ValueError("RANSAC에 사용할 포인트가 너무 적습니다.")

    best_plane = None
    best_inlier_mask = None
    best_inlier_count = -1

    for _ in range(num_iter):
        ids = rng.choice(N, size=sample_size, replace=False)
        p1, p2, p3 = sample_points[ids]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)

        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue

        normal = normal / norm
        d = -np.dot(normal, p1)

        distances = np.abs(sample_points @ normal + d)
        inlier_mask = distances < distance_threshold
        inlier_count = np.count_nonzero(inlier_mask)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)
            best_inlier_mask = inlier_mask

    if best_plane is None:
        raise RuntimeError("RANSAC plane fitting 실패")

    # inlier들로 plane 한 번 더 정밀 보정
    inlier_points = sample_points[best_inlier_mask]
    centroid = np.mean(inlier_points, axis=0)
    centered = inlier_points - centroid

    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    d = -np.dot(normal, centroid)

    refined_plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)

    # 원본 points 기준 inlier mask 재계산
    distances_full = np.abs(points @ refined_plane[:3] + refined_plane[3])
    inlier_mask_full = distances_full < distance_threshold

    return refined_plane, inlier_mask_full

def compute_plane_distance_map(xyz_map, valid_mask, plane):
    """
    xyz_map: H x W x 3
    plane: [a, b, c, d]
    return:
        distance_map: H x W, meter
    """
    normal = plane[:3]
    d = plane[3]

    dist = np.abs(
        xyz_map[..., 0] * normal[0] +
        xyz_map[..., 1] * normal[1] +
        xyz_map[..., 2] * normal[2] +
        d
    )

    dist[~valid_mask] = 0
    return dist

def pca_2d_from_mask(mask):
    """
    mask 안의 픽셀 좌표 기준 PCA.
    return:
        center_uv: [u, v]
        major_axis_uv: [du, dv]
        minor_axis_uv: [du, dv]
        eigenvalues
        angle_deg
        major_length_px: 장축 방향 실제 투영 길이 [px]
        minor_length_px: 단축 방향 실제 투영 길이 [px]
    """
    ys, xs = np.where(mask > 0)

    if len(xs) < 5:
        return None

    pts = np.stack([xs, ys], axis=1).astype(np.float32)

    center = np.mean(pts, axis=0)
    centered = pts - center

    cov = centered.T @ centered / max(len(pts) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    major = eigvecs[:, 0]
    minor = eigvecs[:, 1]

    # 방향 안정화
    if major[0] < 0:
        major = -major

    angle_rad = np.arctan2(major[1], major[0])
    angle_deg = np.degrees(angle_rad)

    # ------------------------------------------------
    # 실제 마스크 픽셀을 PCA 축에 투영해서 길이 계산
    # ------------------------------------------------
    proj_major = centered @ major
    proj_minor = centered @ minor

    major_min = np.min(proj_major)
    major_max = np.max(proj_major)
    minor_min = np.min(proj_minor)
    minor_max = np.max(proj_minor)

    major_length_px = major_max - major_min
    minor_length_px = minor_max - minor_min

    return {
        "center_uv": center,
        "major_axis_uv": major,
        "minor_axis_uv": minor,
        "eigenvalues": eigvals,
        "angle_deg": angle_deg,

        "major_length_px": float(major_length_px),
        "minor_length_px": float(minor_length_px),

        "major_range_px": (float(major_min), float(major_max)),
        "minor_range_px": (float(minor_min), float(minor_max)),
    }

def pca_3d_from_mask(mask, xyz_map, valid_mask):
    """
    mask 안의 3D point 기준 PCA.
    return:
        center_xyz
        major_axis_xyz
        middle_axis_xyz
        minor_axis_xyz
        eigenvalues
        major_length_m
        middle_length_m
        minor_length_m
    """
    use_mask = (mask > 0) & valid_mask
    pts = xyz_map[use_mask]

    if pts.shape[0] < 10:
        return None

    center = np.mean(pts, axis=0)
    centered = pts - center

    cov = centered.T @ centered / max(pts.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    major = eigvecs[:, 0]
    middle = eigvecs[:, 1]
    minor = eigvecs[:, 2]

    proj_major = centered @ major
    proj_middle = centered @ middle
    proj_minor = centered @ minor

    major_length_m = np.max(proj_major) - np.min(proj_major)
    middle_length_m = np.max(proj_middle) - np.min(proj_middle)
    minor_length_m = np.max(proj_minor) - np.min(proj_minor)

    return {
        "center_xyz": center,
        "major_axis_xyz": major,
        "middle_axis_xyz": middle,
        "minor_axis_xyz": minor,
        "eigenvalues": eigvals,

        "major_length_m": float(major_length_m),
        "middle_length_m": float(middle_length_m),
        "minor_length_m": float(minor_length_m),
    }

def extract_object_components_with_pca(
    depth_img,
    depth_scale,
    intrinsics,
    color_img_rgb=None,
    and_mask=None,
    median_ksize=3,
    ransac_distance_threshold=0.006,
    object_min_plane_dist=0.010,
    min_area_px=80,
    morph_open_ksize=3,
    morph_close_ksize=5,
    show=True,
    visualize=False
):
    """
    depth_img: raw depth image
    depth_scale: meter scale
    intrinsics: aligned color intrinsics
    color_img_rgb: 시각화용 RGB 이미지
    and_mask: YOLO mask 같은 추가 마스크. None이면 depth 기반만 사용.
              shape은 depth_img와 같아야 함.
    """

    # -----------------------------
    # A. 약한 median 처리
    # -----------------------------
    if median_ksize is not None and median_ksize >= 3:
        depth_med = cv2.medianBlur(depth_img, median_ksize)
        depth_med[depth_img == 0] = 0
    else:
        depth_med = depth_img.copy()

    # -----------------------------
    # B. depth -> xyz
    # -----------------------------
    xyz_map, valid_mask = depth_to_xyz_map(
        depth_img=depth_med,
        depth_scale=depth_scale,
        intrinsics=intrinsics
    )

    points = xyz_map[valid_mask]

    if points.shape[0] < 100:
        raise ValueError("유효 depth point가 너무 적습니다.")

    # -----------------------------
    # C. RANSAC plane fitting
    # -----------------------------
    plane, plane_inlier_mask_1d = fit_plane_ransac_numpy(
        points=points,
        num_iter=700,
        distance_threshold=ransac_distance_threshold,
        max_points=20000
    )

    plane_dist_map = compute_plane_distance_map(
        xyz_map=xyz_map,
        valid_mask=valid_mask,
        plane=plane
    )

    floor_mask = valid_mask & (plane_dist_map < ransac_distance_threshold)

    # 바닥에서 일정 거리 이상 튀어나온 부분만 객체 후보
    object_depth_mask = valid_mask & (plane_dist_map > object_min_plane_dist)

    # -----------------------------
    # D. YOLO mask 등과 AND
    # -----------------------------
    if and_mask is not None:
        and_mask_bool = and_mask.astype(bool)

        if and_mask_bool.shape != object_depth_mask.shape:
            raise ValueError(
                f"and_mask shape이 depth_img와 다릅니다. "
                f"and_mask={and_mask_bool.shape}, depth={object_depth_mask.shape}"
            )

        object_mask = object_depth_mask & and_mask_bool
    else:
        object_mask = object_depth_mask

    object_mask_u8 = object_mask.astype(np.uint8) * 255

    # -----------------------------
    # E. Morphology 정리
    # -----------------------------
    if morph_open_ksize is not None and morph_open_ksize > 1:
        k_open = np.ones((morph_open_ksize, morph_open_ksize), np.uint8)
        object_mask_u8 = cv2.morphologyEx(object_mask_u8, cv2.MORPH_OPEN, k_open)

    if morph_close_ksize is not None and morph_close_ksize > 1:
        k_close = np.ones((morph_close_ksize, morph_close_ksize), np.uint8)
        object_mask_u8 = cv2.morphologyEx(object_mask_u8, cv2.MORPH_CLOSE, k_close)

    # -----------------------------
    # F. Connected Components
    # -----------------------------
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        object_mask_u8,
        connectivity=8
    )

    components = []

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < min_area_px:
            continue

        comp_mask = (labels == label_id).astype(np.uint8)

        pca2d = pca_2d_from_mask(comp_mask)
        pca3d = pca_3d_from_mask(comp_mask, xyz_map, valid_mask)

        if pca2d is None or pca3d is None:
            continue

        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]

        components.append({
            "label_id": label_id,
            "area_px": int(area),
            "bbox_xywh": (int(x), int(y), int(w), int(h)),
            "mask": comp_mask,
            "pca2d": pca2d,
            "pca3d": pca3d,
        })

    # 큰 덩어리 순서로 정렬
    components = sorted(components, key=lambda c: c["area_px"], reverse=True)

    result = {
        "depth_med": depth_med,
        "xyz_map": xyz_map,
        "valid_mask": valid_mask,
        "plane": plane,
        "plane_dist_map": plane_dist_map,
        "floor_mask": floor_mask,
        "object_depth_mask": object_depth_mask,
        "object_mask": object_mask_u8,
        "labels": labels,
        "components": components
    }

    if show:
        visualize_components_pca(
            color_img_rgb=color_img_rgb,
            object_mask=object_mask_u8,
            floor_mask=floor_mask,
            components=components
        )

    return result

def visualize_components_pca(
    color_img_rgb,
    object_mask,
    floor_mask,
    components,
    axis_len=70
):
    if color_img_rgb is None:
        H, W = object_mask.shape[:2]
        vis = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        vis = color_img_rgb.copy()

    # 혹시 BGR이 들어오면 색이 이상해질 수 있으니, 여기서는 RGB 기준으로 처리
    overlay = vis.copy()

    # 객체 마스크 영역을 초록색으로 표시
    overlay[object_mask > 0] = (
        0.5 * overlay[object_mask > 0] + 0.5 * np.array([0, 255, 0])
    ).astype(np.uint8)

    # 바닥 plane 영역을 어둡게 표시
    overlay[floor_mask] = (
        0.7 * overlay[floor_mask] + 0.3 * np.array([80, 80, 80])
    ).astype(np.uint8)

    draw = overlay.copy()

    for idx, comp in enumerate(components):
        pca2d = comp["pca2d"]
        center = pca2d["center_uv"]
        major = pca2d["major_axis_uv"]
        minor = pca2d["minor_axis_uv"]

        cx, cy = center.astype(int)

        # major axis
        p1 = (
            int(cx - major[0] * axis_len),
            int(cy - major[1] * axis_len)
        )
        p2 = (
            int(cx + major[0] * axis_len),
            int(cy + major[1] * axis_len)
        )

        # minor axis
        q1 = (
            int(cx - minor[0] * axis_len * 0.5),
            int(cy - minor[1] * axis_len * 0.5)
        )
        q2 = (
            int(cx + minor[0] * axis_len * 0.5),
            int(cy + minor[1] * axis_len * 0.5)
        )

        # RGB 이미지지만 cv2 line은 그냥 배열에 색값만 넣는 거라 RGB 색으로 지정
        cv2.line(draw, p1, p2, (255, 0, 0), 3)      # major axis: red
        cv2.line(draw, q1, q2, (0, 0, 255), 2)      # minor axis: blue
        cv2.circle(draw, (cx, cy), 5, (255, 255, 0), -1)

        x, y, w, h = comp["bbox_xywh"]
        cv2.rectangle(draw, (x, y), (x + w, y + h), (255, 255, 0), 2)

        angle = pca2d["angle_deg"]
        text = f"id={idx}, area={comp['area_px']}, yaw2d={angle:.1f}"
        cv2.putText(
            draw,
            text,
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )

    plt.figure(figsize=(14, 7))
    plt.imshow(draw)
    plt.axis("off")
    plt.title("Object Components + PCA Axes")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(object_mask, cmap="gray")
    plt.axis("off")
    plt.title("Final Object Mask")
    plt.show()

def extract_contour_pca_from_mask(
    object_mask,
    xyz_map=None,
    valid_mask=None,
    min_contour_area=80,
    axis_scale=1.0
):
    """
    object_mask:
        RANSAC 바닥 제거 후 남은 객체 마스크.
        0/255 또는 0/1 모두 가능.

    xyz_map:
        H x W x 3, meter.
        3D PCA도 같이 계산하고 싶으면 입력.

    valid_mask:
        H x W bool.
        xyz_map 사용 시 필요.

    return:
        contour_objects: list of dict
    """

    mask_u8 = (object_mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contour_objects = []

    for contour_id, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area < min_contour_area:
            continue

        # -----------------------------------------
        # 1. contour 중심점 계산
        # -----------------------------------------
        M = cv2.moments(contour)

        if abs(M["m00"]) < 1e-6:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        center_uv = np.array([cx, cy], dtype=np.float32)

        # -----------------------------------------
        # 2. contour 내부 mask 생성
        # -----------------------------------------
        contour_mask = np.zeros_like(mask_u8)
        cv2.drawContours(
            contour_mask,
            [contour],
            contourIdx=-1,
            color=255,
            thickness=-1
        )

        ys, xs = np.where(contour_mask > 0)

        if len(xs) < 5:
            continue

        pts_uv = np.stack([xs, ys], axis=1).astype(np.float32)

        # 중심은 contour moment 중심을 기준으로 사용
        centered_uv = pts_uv - center_uv

        # -----------------------------------------
        # 3. 2D PCA
        # -----------------------------------------
        cov_2d = centered_uv.T @ centered_uv / max(len(pts_uv) - 1, 1)

        eigvals_2d, eigvecs_2d = np.linalg.eigh(cov_2d)

        order = np.argsort(eigvals_2d)[::-1]
        eigvals_2d = eigvals_2d[order]
        eigvecs_2d = eigvecs_2d[:, order]

        major_axis_uv = eigvecs_2d[:, 0]
        minor_axis_uv = eigvecs_2d[:, 1]

        # 방향 안정화
        if major_axis_uv[0] < 0:
            major_axis_uv = -major_axis_uv

        # minor도 major에 맞춰 오른손 방향 느낌으로 정리
        minor_axis_uv = np.array(
            [-major_axis_uv[1], major_axis_uv[0]],
            dtype=np.float32
        )

        # -----------------------------------------
        # 4. contour 픽셀들을 PCA 축에 투영해서 실제 길이 계산
        # -----------------------------------------
        proj_major = centered_uv @ major_axis_uv
        proj_minor = centered_uv @ minor_axis_uv

        major_min = np.min(proj_major)
        major_max = np.max(proj_major)
        minor_min = np.min(proj_minor)
        minor_max = np.max(proj_minor)

        major_length_px = major_max - major_min
        minor_length_px = minor_max - minor_min

        angle_rad = np.arctan2(major_axis_uv[1], major_axis_uv[0])
        angle_deg = np.degrees(angle_rad)

        # -----------------------------------------
        # 5. minAreaRect도 같이 저장
        # -----------------------------------------
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)

        x, y, w, h = cv2.boundingRect(contour)

        obj = {
            "contour_id": contour_id,
            "contour": contour,
            "contour_mask": contour_mask,
            "area_px": float(area),
            "bbox_xywh": (int(x), int(y), int(w), int(h)),

            "center_uv": center_uv,

            "major_axis_uv": major_axis_uv.astype(np.float32),
            "minor_axis_uv": minor_axis_uv.astype(np.float32),

            "major_length_px": float(major_length_px),
            "minor_length_px": float(minor_length_px),
            "major_range_px": (float(major_min), float(major_max)),
            "minor_range_px": (float(minor_min), float(minor_max)),

            "angle_deg": float(angle_deg),

            "eigvals_2d": eigvals_2d,

            "min_area_rect": rect,
            "min_area_box": box,
        }

        # -----------------------------------------
        # 6. 선택: 3D PCA도 계산
        # -----------------------------------------
        if xyz_map is not None and valid_mask is not None:
            use_3d_mask = (contour_mask > 0) & valid_mask
            pts_xyz = xyz_map[use_3d_mask]

            if pts_xyz.shape[0] >= 10:
                center_xyz = np.mean(pts_xyz, axis=0)
                centered_xyz = pts_xyz - center_xyz

                cov_3d = centered_xyz.T @ centered_xyz / max(pts_xyz.shape[0] - 1, 1)

                eigvals_3d, eigvecs_3d = np.linalg.eigh(cov_3d)

                order3 = np.argsort(eigvals_3d)[::-1]
                eigvals_3d = eigvals_3d[order3]
                eigvecs_3d = eigvecs_3d[:, order3]

                major_axis_xyz = eigvecs_3d[:, 0]
                middle_axis_xyz = eigvecs_3d[:, 1]
                minor_axis_xyz = eigvecs_3d[:, 2]

                proj_x = centered_xyz @ major_axis_xyz
                proj_y = centered_xyz @ middle_axis_xyz
                proj_z = centered_xyz @ minor_axis_xyz

                obj["center_xyz"] = center_xyz
                obj["major_axis_xyz"] = major_axis_xyz
                obj["middle_axis_xyz"] = middle_axis_xyz
                obj["minor_axis_xyz"] = minor_axis_xyz

                obj["major_length_m"] = float(np.max(proj_x) - np.min(proj_x))
                obj["middle_length_m"] = float(np.max(proj_y) - np.min(proj_y))
                obj["minor_length_m"] = float(np.max(proj_z) - np.min(proj_z))

                obj["eigvals_3d"] = eigvals_3d

        contour_objects.append(obj)

    contour_objects = sorted(
        contour_objects,
        key=lambda o: o["area_px"],
        reverse=True
    )

    return contour_objects

def visualize_contour_pca_axes(
    color_img_rgb,
    object_mask,
    contour_objects,
    draw_mask=True,
    draw_contour=True,
    draw_min_rect=True,
    axis_len_mode="pca_length",
    fixed_axis_len=80
):
    """
    axis_len_mode:
        "pca_length" -> contour 투영 길이 기반으로 축 길이 그림
        "fixed"      -> fixed_axis_len으로 고정 길이 그림
    """

    if color_img_rgb is None:
        H, W = object_mask.shape[:2]
        vis = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        vis = color_img_rgb.copy()

    mask_u8 = (object_mask > 0).astype(np.uint8) * 255

    overlay = vis.copy()

    if draw_mask:
        overlay[mask_u8 > 0] = (
            0.55 * overlay[mask_u8 > 0] +
            0.45 * np.array([0, 255, 0])
        ).astype(np.uint8)

    draw = overlay.copy()

    for idx, obj in enumerate(contour_objects):
        contour = obj["contour"]
        center = obj["center_uv"]
        major = obj["major_axis_uv"]
        minor = obj["minor_axis_uv"]

        cx, cy = center
        cxi, cyi = int(round(cx)), int(round(cy))

        if axis_len_mode == "pca_length":
            major_half_len = obj["major_length_px"] * 0.5
            minor_half_len = obj["minor_length_px"] * 0.5
        else:
            major_half_len = fixed_axis_len
            minor_half_len = fixed_axis_len * 0.5

        # 장축 endpoints
        p1 = (
            int(round(cx - major[0] * major_half_len)),
            int(round(cy - major[1] * major_half_len))
        )
        p2 = (
            int(round(cx + major[0] * major_half_len)),
            int(round(cy + major[1] * major_half_len))
        )

        # 단축 endpoints
        q1 = (
            int(round(cx - minor[0] * minor_half_len)),
            int(round(cy - minor[1] * minor_half_len))
        )
        q2 = (
            int(round(cx + minor[0] * minor_half_len)),
            int(round(cy + minor[1] * minor_half_len))
        )

        if draw_contour:
            cv2.drawContours(draw, [contour], -1, (255, 255, 0), 2)

        if draw_min_rect:
            box = obj["min_area_box"]
            cv2.drawContours(draw, [box], 0, (255, 0, 255), 2)

        # PCA 장축: 빨강
        cv2.line(draw, p1, p2, (255, 0, 0), 3)

        # PCA 단축: 파랑
        cv2.line(draw, q1, q2, (0, 0, 255), 2)

        # 중심점: 노랑
        cv2.circle(draw, (cxi, cyi), 5, (255, 255, 0), -1)

        x, y, w, h = obj["bbox_xywh"]

        text1 = f"id={idx}, area={obj['area_px']:.0f}"
        text2 = f"yaw={obj['angle_deg']:.1f}, L={obj['major_length_px']:.1f}, W={obj['minor_length_px']:.1f}"

        cv2.putText(
            draw,
            text1,
            (x, max(15, y - 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            draw,
            text2,
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )

    plt.figure(figsize=(14, 7))
    plt.imshow(draw)
    plt.axis("off")
    plt.title("RANSAC Object Mask Contours + PCA Axes")
    plt.show()

    return draw



# 포인트 클라우드 함수

def refine_2d_mask_with_hull(projected_mask_01, color_bgr):
    """
    파먹히고 조각난 2D 투영 마스크(0 or 1)를 Convex Hull과 모폴로지 연산을 
    이용해 꽉 찬(Solid) 객체 마스크로 복원하고 컬러 이미지를 커팅합니다.
    """
    # 1. OpenCV 연산을 위해 마스크를 0~255 스케일(uint8)로 변환
    mask_255 = (projected_mask_01 * 255).astype(np.uint8)
    
    # 2. 미세하게 끊어진 조각들을 하나로 뭉치기 위해 팽창(Dilation) 및 닫기(Close) 적용
    # 커널 사이즈(3x3)는 객체가 서로 너무 가까워 붙지 않는 선에서 조절 (필요시 5x5 로 변경)
    kernel = np.ones((7, 7), np.uint8)
    closed_mask = cv2.morphologyEx(mask_255, cv2.MORPH_CLOSE, kernel)
    
    # 3. 마스크 내의 모든 윤곽선(Contours) 찾기
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4. 복원된 마스크를 그릴 빈 도화지 생성
    refined_mask = np.zeros_like(mask_255)
    
    # 5. 각 윤곽선에 대해 Convex Hull(볼록 껍질)을 구하고 꽉 채워서 그리기
    for cnt in contours:
        # 노이즈(너무 작은 덩어리)는 무시 (면적 임계값: 100픽셀)
        if cv2.contourArea(cnt) > 200:
            # 윤곽선을 고무줄로 묶듯이 볼록한 형태로 감쌈
            hull = cv2.convexHull(cnt)
            # 도화지에 하얀색(255)으로 꽉 채워서(thickness=-1) 그리기
            cv2.drawContours(refined_mask, [hull], 0, 255, -1)
            
    # 6. 최종 복원된 마스크를 0과 1로 다시 변환
    refined_mask_01 = (refined_mask / 255).astype(np.uint8)
    
    # 7. 완벽해진 마스크로 원본 컬러 이미지 다시 커팅
    final_cut_color = cv2.bitwise_and(color_bgr, color_bgr, mask=refined_mask_01)
    
    return refined_mask_01, final_cut_color

def create_floor_anchored_3d_box_with_axes(box_2d, intrinsics, plane_normal, d, max_h, color, axis_size=0.03):
# [함수] 2D OBB를 3D 바닥 평면으로 역투영하여 3D 박스 및 좌표계 생성
    base_corners = []
    # 1. 2D 픽셀 4개의 꼭짓점을 3D 바닥 평면에 대입하여 교점 획득
    for (u, v) in box_2d:
        dir_vec = np.array([(u - intrinsics.ppx) / intrinsics.fx,
                            (v - intrinsics.ppy) / intrinsics.fy,
                            1.0])
        t = -d / np.dot(plane_normal, dir_vec)
        base_pt = t * dir_vec
        base_corners.append(base_pt)
        
    base_corners = np.array(base_corners)
    top_corners = base_corners + (plane_normal * max_h)
    
    # 2. 3D 박스 외곽선(LineSet) 생성
    vertices = np.vstack((base_corners, top_corners))
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0], # 바닥면
        [4, 5], [5, 6], [6, 7], [7, 4], # 천장면
        [0, 4], [1, 5], [2, 6], [3, 7]  # 수직 기둥
    ]
    
    colors = [color for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # 🎯 3. 로컬 좌표계(Coordinate Frame) 생성 로직
    # 박스 중심점 계산
    center = np.mean(vertices, axis=0)

    # Z축: 평면의 법선 벡터 (바닥에서 위를 향함)
    Z = plane_normal / np.linalg.norm(plane_normal)

    # X축: 바닥의 변 중에서 '더 긴 변(Major Axis)'을 X축으로 설정 (그리퍼 파지 방향)
    vec1 = base_corners[1] - base_corners[0]
    vec2 = base_corners[2] - base_corners[1]
    
    if np.linalg.norm(vec1) > np.linalg.norm(vec2):
        X_dir = vec1
    else:
        X_dir = vec2
        
    X = X_dir / np.linalg.norm(X_dir)

    # Y축: Z와 X의 외적 (직교 보장)
    Y = np.cross(Z, X)
    Y = Y / np.linalg.norm(Y)

    # 회전 행렬 조립 (3x3 Matrix)
    R = np.column_stack((X, Y, Z))

    # Open3D 좌표계 메쉬 생성 및 이동
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
    axes.rotate(R, center=(0, 0, 0)) # 회전 적용
    axes.translate(center)           # 박스 중심으로 이동

    # 리턴값에 좌표계(axes) 추가
    return line_set, axes

def create_floor_anchored_3d_box(box_2d, intrinsics, plane_normal, d, max_h, color):
# [함수 2] 2D OBB를 3D 바닥 평면으로 역투영하여 바닥 밀착형 박스 생성
    base_corners = []
    # 2D 픽셀 4개의 꼭짓점을 3D 바닥 평면에 대입하여 교점 획득
    for (u, v) in box_2d:
        dir_vec = np.array([(u - intrinsics.ppx) / intrinsics.fx,
                            (v - intrinsics.ppy) / intrinsics.fy,
                            1.0])
        t = -d / np.dot(plane_normal, dir_vec)
        base_pt = t * dir_vec
        base_corners.append(base_pt)
        
    base_corners = np.array(base_corners)
    top_corners = base_corners + (plane_normal * max_h)
    
    vertices = np.vstack((base_corners, top_corners))
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0], # 바닥면
        [4, 5], [5, 6], [6, 7], [7, 4], # 천장면
        [0, 4], [1, 5], [2, 6], [3, 7]  # 수직 기둥
    ]
    
    colors = [color for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

# 컨트롤 함수

def detect_objects_yolo(model, color_img_bgr, target_classes=None, visualize=False):
    """
    YOLOv8 모델을 사용하여 특정 클래스에 대한 객체를 검출하고, 이진 마스크로 반환
    
    Args:
        model (YOLO): 로드된 YOLO 모델 객체 (예: YOLO("best.pt"))
        color_img_bgr (ndarray): 모델 입력용 원본 BGR 이미지
        target_classes (list, optional): 검출할 클래스 ID 리스트. (예: [0, 1, 3, 4, 5, 6, 8, 9])
                                         None일 경우 모든 클래스를 검출합니다.
        visualize (bool): 추론 결과(YOLO plot)를 Matplotlib으로 시각화할지 여부
        
    Returns:
        results (list): YOLO 모델의 원본 추론 결과 객체 리스트
        mask_binary (ndarray): 검출된 모든 객체의 마스크를 하나로 합친 이진 마스크 (0 or 1, 형태: H x W)
        vis_yolo (ndarray): 바운딩 박스와 라벨이 그려진 시각화용 이미지 (BGR)
    """

    # 1. 원본 이미지 크기 파악 (마스크 리사이즈용)
    img_height, img_width = color_img_bgr.shape[:2]
    
    # 2. 모델 추론 (클래스 필터링 적용)
    if target_classes is not None:
        results = model(color_img_bgr, classes=target_classes, verbose=False)
    else:
        results = model(color_img_bgr, verbose=False)

    # 3. 마스크 병합용 빈 도화지 생성
    mask_binary = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # YOLO의 내장 시각화 결과 이미지 생성
    vis_yolo = results[0].plot()

    # 4. 검출된 마스크 합치기
    if len(results) > 0 and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        
        for mask in masks:
            # YOLO는 내부적으로 마스크 크기를 조절할 수 있으므로, 원본 이미지 크기에 맞춰 리사이즈
            mask_resized = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            # 기존 마스크에 겹쳐서 누적 (OR 연산)
            mask_binary = np.logical_or(mask_binary, mask_resized > 0.5).astype(np.uint8)
            
        print(f"🎯 [SUCCESS] {len(masks)}개의 타겟 객체 마스크 병합 완료")
    else:
        print("⚠️ [WARN] 지정된 클래스의 객체가 검출되지 않았거나, 마스크가 없습니다.")

    # 5. 시각화 (옵션)
    if visualize:

        mask_vis = mask_binary * 255
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 6))

        axes[0].imshow(cv2.cvtColor(vis_yolo, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"YOLO Segmentations (Targets: {target_classes})")
        axes[0].axis("off")
        
        axes[1].imshow(mask_vis, cmap='gray')
        axes[1].set_title("Merged Binary Mask (ROI)")
        axes[1].axis("off")

        vis_orig = cv2.cvtColor(color_img_bgr, cv2.COLOR_BGR2RGB)
        axes[2].imshow(vis_orig)
        axes[2].set_title("original")
        axes[2].axis("off")
        
        plt.tight_layout()
        plt.show()

    return results, mask_binary, vis_yolo

    # YOLOv8 세그멘테이션 결과에서 겹치는 마스크를 병합하고 오검출을 정리하는 함수.
    # 작은 객체의 마스크가 큰 객체의 마스크에 설정된 비율 이상 포함되면 오검출로 간주하고 억제(Suppression)합니다.

def filter_overlapping_masks(results, overlap_threshold=0.70, img_shape=(640, 480), visualize=False):
    """
    YOLOv8 세그멘테이션 결과에서 겹치는 마스크를 병합하고 오검출을 정리하는 함수.
    작은 객체의 마스크가 큰 객체의 마스크에 설정된 비율 이상 포함되면 오검출로 간주하고 억제(Suppression)합니다.
    
    Args:
        results (list): YOLO 모델의 추론 결과 객체 리스트 (model(img)의 반환값)
        overlap_threshold (float): 포함 판단 기준 (0.0 ~ 1.0). 기본값 0.70 (70% 이상 겹치면 무시)
        img_shape (tuple): 원본 이미지의 (Width, Height). 마스크 리사이즈용
        visualize (bool): 원본 추론 결과와 정리된 마스크를 비교하는 시각화 플롯 출력 여부
        
    Returns:
        final_detected_objects (list): 억제 후 살아남은 최종 객체들의 리스트. 
                                       각 요소는 dict 형태 (class_id, class_name, confidence, mask)
        final_combined_mask (ndarray): 병합된 최종 전체 ROI 마스크 (0 or 1, uint8)
    """
    final_detected_objects = []
    target_w, target_h = img_shape
    final_combined_mask = np.zeros((target_h, target_w), dtype=np.uint8)

    if len(results) > 0 and results[0].masks is not None:
        boxes = results[0].boxes
        masks = results[0].masks.data.cpu().numpy()  # 형상: (N, H, W)
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        
        # YOLO 결과 객체 내부에 저장된 클래스 이름 딕셔너리 활용 (model 객체 불필요)
        class_names = results[0].names 

        # 1. 각 마스크의 픽셀 면적 계산
        areas = np.array([np.sum(mask > 0.5) for mask in masks])
        
        # 2. 면적이 큰 순서대로 인덱스 정렬
        sorted_indices = np.argsort(-areas)
        suppressed_indices = set()  # 먹혀서 사라질(무시할) 작은 객체의 인덱스 모음

        for i in range(len(sorted_indices)):
            idx_large = sorted_indices[i]
            
            # 이미 다른 큰 객체에 먹힌 객체라면 패스
            if idx_large in suppressed_indices:
                continue
                
            mask_large = masks[idx_large]
            area_large = areas[idx_large]
            
            # 현재 (가장 큰) 객체를 최종 리스트에 추가 (이때 마스크를 원본 해상도로 리사이즈)
            resized_mask_large = cv2.resize(mask_large, (target_w, target_h), interpolation=cv2.INTER_NEAREST) > 0.5
            
            current_obj = {
                "class_id": class_ids[idx_large],
                "class_name": class_names[class_ids[idx_large]],
                "confidence": confidences[idx_large],
                "mask": resized_mask_large
            }
            final_detected_objects.append(current_obj)
            
            # 3. 나보다 작은 나머지 객체들과 비교
            for j in range(i + 1, len(sorted_indices)):
                idx_small = sorted_indices[j]
                
                if idx_small in suppressed_indices:
                    continue
                    
                mask_small = masks[idx_small]
                area_small = areas[idx_small]
                
                # 두 마스크의 교집합(AND) 계산
                intersection = np.sum(np.logical_and(mask_large > 0.5, mask_small > 0.5))
                
                # 작은 마스크가 큰 마스크에 얼마나 포함되어 있는지 비율 계산
                overlap_ratio = intersection / area_small if area_small > 0 else 0
                
                # 작은 마스크의 대부분(지정된 임계값 이상)이 겹친다면 오검출 판단
                if overlap_ratio > overlap_threshold:
                    print(f"✂️ [INFO] 억제됨: '{class_names[class_ids[idx_small]]}' (면적:{area_small})가 "
                          f"'{class_names[class_ids[idx_large]]}' (면적:{area_large})에 {overlap_ratio*100:.1f}% 포함됨.")
                    suppressed_indices.add(idx_small)
                    
                    # 옵션: 작은 마스크의 삐져나온 영역까지 큰 객체로 흡수하고 싶다면
                    # current_obj["mask"] = np.logical_or(current_obj["mask"], cv2.resize(mask_small, (target_w, target_h), interpolation=cv2.INTER_NEAREST) > 0.5)

    else:
        print("⚠️ [WARN] 검출된 객체가 없습니다.")

    # 5. 시각화 (옵션)
    if visualize:
        # 4. 최종 결과 출력 및 전체 마스크 병합
        print(f"\n✅ 최종 검출된 유효 객체/군집 수: {len(final_detected_objects)}개")
        for obj in final_detected_objects:
            print(f" - 🏷️ {obj['class_name']} (신뢰도: {obj['confidence']:.2f})")
            final_combined_mask = np.logical_or(final_combined_mask, obj["mask"]).astype(np.uint8)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        if len(results) > 0:
            axes[0].imshow(results[0].plot())
        else:
            axes[0].text(0.5, 0.5, 'No Detections', ha='center', va='center', fontsize=15)
            
        axes[0].set_title("Original YOLO Output")
        axes[0].axis("off")

        axes[1].imshow(final_combined_mask, cmap='gray')
        axes[1].set_title(f"nms Cleaned Masks ({len(final_detected_objects)} Objects)")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    return final_detected_objects, final_combined_mask

def estimate_floor_plane(depth_img, yolo_combined_mask, intrinsics, depth_scale, depth_trunc=1.5, visualize=False):
    """
    [STEP 1] YOLO 마스크를 제외한 바닥(Background) 영역만 추출하여 RANSAC 평면 방정식을 도출합니다.
    """
    print("\n[INFO] RANSAC 바닥 평면 추정 시작...")
    
    # 1. 객체 영역 제외 및 순수 바닥 Depth 추출
    filtered_depth_img = cv2.medianBlur(depth_img, 5)
    kernel = np.ones((7, 7), np.uint8)
    expanded_yolo_mask = cv2.dilate(yolo_combined_mask, kernel, iterations=3)
    
    bg_depth_img = filtered_depth_img.copy()
    bg_depth_img[expanded_yolo_mask > 0] = 0
    
    # 2. Open3D 카메라 파라미터 세팅
    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        int(intrinsics.width), int(intrinsics.height),
        float(intrinsics.fx), float(intrinsics.fy),
        float(intrinsics.ppx), float(intrinsics.ppy)
    )
    o3d_depth_scale = 1.0 / float(depth_scale)
    
    # 3. 바닥 포인트 클라우드 생성
    bg_depth_o3d = o3d.geometry.Image(bg_depth_img)
    bg_pcd = o3d.geometry.PointCloud.create_from_depth_image(
        bg_depth_o3d, o3d_intr, depth_scale=o3d_depth_scale, depth_trunc=depth_trunc
    )
    bg_pcd = bg_pcd.voxel_down_sample(voxel_size=0.003)
    
    # 🚨 방어 코드 1: 다운샘플링 후 포인트가 너무 적으면 중단
    if len(bg_pcd.points) < 10:
        print(f"❌ [ERROR] 바닥 포인트 클라우드 생성 실패 (현재 포인트 수: {len(bg_pcd.points)}개)")
        print(f"   -> 카메라와 바닥의 거리가 {depth_trunc}m 보다 멀 수 있습니다. depth_trunc 값을 늘려보세요.")
        return None, None, filtered_depth_img

    bg_pcd, _ = bg_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # 🚨 방어 코드 2: 아웃라이어 제거 후 RANSAC 최소 조건(3개) 검사
    if len(bg_pcd.points) < 3:
        print("❌ [ERROR] 아웃라이어 제거 후 RANSAC을 수행할 유효한 포인트가 부족합니다.")
        return None, None, filtered_depth_img
    
    # 4. RANSAC 평면 검출
    plane_model, inliers = bg_pcd.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=1000)
    a, b, c, d = plane_model
    plane_normal = np.array([a, b, c])
    
    # 카메라 시점 방향 보정
    if c > 0: 
        plane_normal = -plane_normal
        d = -d
        plane_model = (a, b, c, d)
        
    print(f"✅ 바닥 평면 방정식 도출: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
    
    if visualize:
        inlier_cloud = bg_pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        outlier_cloud = bg_pcd.select_by_index(inliers, invert=True)
        outlier_cloud.paint_uniform_color([1, 0, 0])
        print("💡 [Visualizer] 3D RANSAC 결과 창이 열립니다. (창을 닫아야 다음 코드가 진행됩니다)")
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name="Floor RANSAC Result")

    return plane_model, plane_normal, filtered_depth_img

def extract_high_objects_mask(filtered_depth_img, plane_normal, d, intrinsics, depth_scale, color_img_shape, height_threshold=0.040, visualize=False):
    """
    [STEP 2] 전체 씬에서 바닥 기준 지정된 높이(height_threshold) 이상 돌출된 포인트를 2D 마스크로 추출합니다.
    """
    print(f"\n[INFO] 바닥 기준 {height_threshold*1000:.1f}mm 이상 돌출된 포인트 추출 중...")
    
    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        int(intrinsics.width), int(intrinsics.height),
        float(intrinsics.fx), float(intrinsics.fy),
        float(intrinsics.ppx), float(intrinsics.ppy)
    )
    o3d_depth_scale = 1.0 / float(depth_scale)
    
    # 1. 전체 포인트 클라우드 생성
    depth_o3d = o3d.geometry.Image(filtered_depth_img)
    full_pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d, o3d_intr, depth_scale=o3d_depth_scale, depth_trunc=1.5
    )
    full_pcd = full_pcd.voxel_down_sample(voxel_size=0.001)
    
    # 2. 바닥 평면으로부터의 높이 계산
    points = np.asarray(full_pcd.points)
    signed_height = np.dot(points, plane_normal) + d
    
    above_threshold_indices = np.where(signed_height > height_threshold)[0]
    pcd_above = full_pcd.select_by_index(above_threshold_indices)
    
    print(f"✅ {height_threshold*1000:.1f}mm 이상 돌출된 3D 포인트 개수: {len(pcd_above.points)}개")
    
    # 3. 3D -> 2D 투영 마스크 생성
    h, w = color_img_shape[:2]
    projected_mask = np.zeros((h, w), dtype=np.uint8)
    obj_points = np.asarray(pcd_above.points)
    
    if len(obj_points) > 0:
        x_3d, y_3d, z_3d = obj_points[:, 0], obj_points[:, 1], obj_points[:, 2]
        z_3d = np.where(z_3d == 0, 0.00001, z_3d)
        
        u_coords = np.round((x_3d * intrinsics.fx / z_3d) + intrinsics.ppx).astype(int)
        v_coords = np.round((y_3d * intrinsics.fy / z_3d) + intrinsics.ppy).astype(int)
        
        valid = (u_coords >= 0) & (u_coords < w) & (v_coords >= 0) & (v_coords < h)
        projected_mask[v_coords[valid], u_coords[valid]] = 1
        
        # 미세한 구멍 메우기
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        projected_mask = cv2.morphologyEx(projected_mask, cv2.MORPH_CLOSE, kernel)
        
    if visualize:
        plt.figure(figsize=(8, 6))
        plt.imshow(projected_mask, cmap='hot')
        plt.title(f"Projected 2D Mask (> {height_threshold*1000:.1f}mm)")
        plt.axis("off")
        plt.show()
        
    return projected_mask

def correct_object_ids(detected_objects, mask_high_2d, color_img_bgr, ratio_threshold=1.5, overlap_threshold=0.20, visualize=False):
    """
    [STEP 3] OBB 비율(가로/세로) 및 3D 높이 마스크와의 교집합을 통해 객체의 오분류를 교정합니다.
    """
    if visualize:
        print("\n[INFO] 객체 마스크 기반 OBB 추출 및 물리적 조건 기반 ID 교정 중...")
    
    vis_image = color_img_bgr.copy()
    h, w = color_img_bgr.shape[:2]
    
    mask_high_vis = np.zeros((h, w), dtype=np.uint8)  # 높은 객체 누적용
    mask_low_vis = np.zeros((h, w), dtype=np.uint8)   # 낮은 객체 누적용
    
    # 딕셔너리 리스트 순회 (참조로 값 직접 변경)
    for obj in detected_objects:
        yolo_mask = obj["mask"].astype(np.uint8)
        contours, _ = cv2.findContours(yolo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
            
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        
        # 1. OBB 비율 계산
        rect_w, rect_h = rect[1]
        ratio = max(rect_w, rect_h) / min(rect_w, rect_h) if min(rect_w, rect_h) > 0 else 0
        
        # 2. 높이 맵 교집합 계산
        overlap = np.logical_and(yolo_mask, mask_high_2d)
        overlap_ratio = np.count_nonzero(overlap) / np.count_nonzero(yolo_mask) if np.count_nonzero(yolo_mask) > 0 else 0
        
        old_name = obj["class_name"]

        # 단순 이름 교정
        if overlap_ratio > overlap_threshold:
            # [A] 쌓인 객체 (높이 조건 충족)
            mask_high_vis = np.logical_or(mask_high_vis, yolo_mask).astype(np.uint8)
            if "2x2" in old_name:
                new_name = old_name.replace("2x2", "4x2")
                obj["class_name"] = f"{new_name}"
                if visualize:                
                    print(f" ⚠️ [높이 교정] 쌓인 블록 감지! '{old_name}' -> '{obj['class_name']}'")
        else:
            # [B] 바닥에 깔린 객체
            mask_low_vis = np.logical_or(mask_low_vis, yolo_mask).astype(np.uint8)
            if ("4x2" in old_name or "2x4" in old_name) and ratio <= ratio_threshold:
                new_name = old_name.replace("4x2", "2x2").replace("2x4", "2x2")
                obj["class_name"] = f"{new_name}"
                if visualize:
                    print(f" 🔍 [비율 교정] 짧은 블록 감지 (비율:{ratio:.2f}). '{old_name}' -> '{obj['class_name']}'")

        # 4. 시각화 데이터 렌더링
        box = np.intp(cv2.boxPoints(rect))
        cv2.drawContours(vis_image, [box], 0, (0, 0, 255), 2)
        
        top_point = box[np.argmin(box[:, 1])]
        cv2.putText(vis_image, obj['class_name'], (top_point[0] - 20, top_point[1] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(vis_image, f"Ratio: {ratio:.2f}", (top_point[0] - 20, top_point[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if visualize:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(mask_high_vis, cmap="gray")
        axes[0].set_title("YOLO Objects: HIGH (> 40mm)")
        axes[0].axis("off")
        axes[1].imshow(mask_low_vis, cmap="gray")
        axes[1].set_title("YOLO Objects: LOW (<= 40mm)")
        axes[1].axis("off")
        axes[2].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Final IDs & Oriented Bounding Boxes")
        axes[2].axis("off")
        plt.tight_layout()
        plt.show()

    return detected_objects, vis_image

def extract_3d_protruding_objects(depth_img, color_img_bgr, intrinsics, depth_scale, yolo_combined_mask=None, depth_trunc=1.5, height_threshold=0.005, visualize=False):
    """
    Depth 맵을 3D Point Cloud로 변환 후, 바닥(Plane)을 찾아 지정된 높이 이상 
    돌출된 객체만 추출하고 이를 2D 이미지로 마스킹하여 반환하는 통합 함수.
    """
    if visualize:
        print("\n[INFO] 3D 기반 돌출 객체 추출 및 2D 마스킹 파이프라인 시작...")
    
    filtered_depth_img = cv2.medianBlur(depth_img, 5)
    
    # 🎯 [추가됨] YOLO 마스크가 주어졌다면, 객체 영역을 지워 '순수한 바닥용 뎁스' 생성
    if yolo_combined_mask is not None:
        kernel = np.ones((7, 7), np.uint8)
        expanded_yolo_mask = cv2.dilate(yolo_combined_mask, kernel, iterations=3)
        bg_depth_img = filtered_depth_img.copy()
        bg_depth_img[expanded_yolo_mask > 0] = 0
    else:
        bg_depth_img = filtered_depth_img.copy()
        
    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        int(intrinsics.width), int(intrinsics.height),
        float(intrinsics.fx), float(intrinsics.fy),
        float(intrinsics.ppx), float(intrinsics.ppy)
    )
    o3d_depth_scale = 1.0 / float(depth_scale)
    
    # =================================================================
    # 파트 A: 바닥 방정식 찾기 (bg_depth_img 활용)
    # =================================================================
    bg_depth_o3d = o3d.geometry.Image(bg_depth_img)
    bg_pcd = o3d.geometry.PointCloud.create_from_depth_image(
        bg_depth_o3d, o3d_intr, depth_scale=o3d_depth_scale, depth_trunc=depth_trunc
    )
    bg_pcd = bg_pcd.voxel_down_sample(voxel_size=0.003) 
    
    if len(bg_pcd.points) < 10:
        print(f"❌ [ERROR] 바닥 검출을 위한 유효한 3D 포인트가 부족합니다.")
        return None, None, None, None

    bg_pcd, _ = bg_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    labels = np.array(bg_pcd.cluster_dbscan(eps=0.02, min_points=20, print_progress=False))
    a, b, c, d = 0, 0, 1, 0  
    
    if len(labels) > 0 and labels.max() >= 0:
        largest_cluster_idx = np.argmax(np.bincount(labels[labels >= 0]))
        main_cluster_indices = np.where(labels == largest_cluster_idx)[0]
        floor_candidate_pcd = bg_pcd.select_by_index(main_cluster_indices)
        
        if len(floor_candidate_pcd.points) >= 3:
            plane_model, inliers = floor_candidate_pcd.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=1000)
            a, b, c, d = plane_model
            plane_normal = np.array([a, b, c])
            if c > 0: 
                plane_normal = -plane_normal
                d = -d
                plane_model = (a, b, c, d)
            if visualize:
                print(f"✅ 바닥 평면 도출 성공: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
        else:
            print("⚠️ [WARN] 바닥 후보군 포인트 부족.")
            return None, None, None, None
    else:
        print("⚠️ [WARN] DBSCAN 클러스터링으로 바닥을 찾지 못했습니다.")
        return None, None, None, None

    # =================================================================
    # 파트 B: 전체 씬에서 돌출 객체 추출 (원본 filtered_depth_img 활용)
    # =================================================================
    full_depth_o3d = o3d.geometry.Image(filtered_depth_img)
    full_pcd = o3d.geometry.PointCloud.create_from_depth_image(
        full_depth_o3d, o3d_intr, depth_scale=o3d_depth_scale, depth_trunc=depth_trunc
    )
    full_pcd = full_pcd.voxel_down_sample(voxel_size=0.003)
    
    points = np.asarray(full_pcd.points)
    signed_height = np.dot(points, plane_normal) + d
    
    object_indices = np.where(signed_height > height_threshold)[0]
    object_pcd = full_pcd.select_by_index(object_indices)
    object_points = np.asarray(object_pcd.points)
    
    print(f"✅ {height_threshold*1000:.1f}mm 이상 돌출된 객체 포인트: {len(object_points)}개")

    # =================================================================
    # 파트 C: 2D 사영 및 마스크 정제
    # =================================================================
    h, w = color_img_bgr.shape[:2]
    object_mask_2d = np.zeros((h, w), dtype=np.uint8)

    if len(object_points) > 0:
        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.ppx, intrinsics.ppy

        x_3d, y_3d = object_points[:, 0], object_points[:, 1]
        z_3d = np.where(object_points[:, 2] == 0, 0.00001, object_points[:, 2])

        u_coords = np.round((x_3d * fx / z_3d) + cx).astype(int)
        v_coords = np.round((y_3d * fy / z_3d) + cy).astype(int)

        valid = (u_coords >= 0) & (u_coords < w) & (v_coords >= 0) & (v_coords < h)
        object_mask_2d[v_coords[valid], u_coords[valid]] = 1

    kernel_fill = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    object_mask_filled = cv2.dilate(object_mask_2d, kernel_fill, iterations=1)
    object_mask_filled = cv2.morphologyEx(object_mask_filled, cv2.MORPH_CLOSE, kernel_fill)

    mask_255 = (object_mask_filled * 255).astype(np.uint8)
    kernel_close = np.ones((7, 7), np.uint8)
    closed_mask = cv2.morphologyEx(mask_255, cv2.MORPH_CLOSE, kernel_close)
    
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined_color_img = cv2.bitwise_and(color_img_bgr, color_img_bgr, mask=closed_mask)

    if visualize:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(object_mask_filled, cmap="gray")
        axes[0].set_title("1. Original Protruding Mask")
        axes[0].axis("off")
        axes[1].imshow(closed_mask, cmap="gray")
        axes[1].set_title("2. Morphological Closed Mask")
        axes[1].axis("off")
        vis_refined_color = refined_color_img.copy()
        cv2.drawContours(vis_refined_color, contours, -1, (0, 255, 0), 2)
        axes[2].imshow(cv2.cvtColor(vis_refined_color, cv2.COLOR_BGR2RGB))
        axes[2].set_title("3. Refined Color Objects")
        axes[2].axis("off")
        plt.tight_layout()
        plt.show()

    return closed_mask, refined_color_img, contours, plane_model

def process_scene_and_get_height_masks(depth_img, intrinsics, depth_scale, color_img_shape):
    """
    [STEP 1~3] 3D Point Cloud 생성, DBSCAN+RANSAC 바닥 평탄화 및 높이별 2D 마스크 사영
    """
    print("\n[INFO] 3D Scene 분석 및 높이별 2D 마스크 추출 시작...")
    
    # 1. Depth 전처리 및 3D 점군 생성
    filtered_depth_img = cv2.medianBlur(depth_img, 5)
    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        int(intrinsics.width), int(intrinsics.height),
        float(intrinsics.fx), float(intrinsics.fy),
        float(intrinsics.ppx), float(intrinsics.ppy)
    )
    o3d_depth_scale = 1.0 / float(depth_scale)

    depth_o3d = o3d.geometry.Image(filtered_depth_img)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d, o3d_intr, depth_scale=o3d_depth_scale, depth_trunc=1.5
    )
    pcd = pcd.voxel_down_sample(voxel_size=0.003)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd, _ = pcd.remove_radius_outlier(nb_points=10, radius=0.01)

    # 2. 바닥 검출 및 평탄화
    labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=20, print_progress=False))
    a, b, c, d = 0, 0, 1, 0
    floor_pcd = None
    
    if len(labels) > 0 and labels.max() >= 0:
        main_cluster_indices = np.where(labels == np.argmax(np.bincount(labels[labels >= 0])))[0]
        floor_candidate_pcd = pcd.select_by_index(main_cluster_indices)
        plane_model, _ = floor_candidate_pcd.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=1000)
        a, b, c, d = plane_model
        plane_normal = np.array([a, b, c])
        if c > 0:
            plane_normal = -plane_normal
            d = -d
            
        # 바닥 평탄화 (다림질)
        points = np.asarray(pcd.points)
        signed_height = np.dot(points, plane_normal) + d
        floor_indices = np.where(signed_height <= 0.005)[0]
        floor_pcd = pcd.select_by_index(floor_indices)
        floor_points = np.asarray(floor_pcd.points)
        distances = np.dot(floor_points, plane_normal) + d
        flattened_points = floor_points - np.outer(distances, plane_normal)
        floor_pcd.points = o3d.utility.Vector3dVector(flattened_points)
        floor_pcd.paint_uniform_color([0.8, 0.8, 0.8])
    else:
        print("⚠️ 바닥을 검출하지 못했습니다.")
        points = np.asarray(pcd.points)
        signed_height = np.zeros(len(points))
        plane_normal = np.array([0, 0, 1])

    # 3. 2D 마스크 사영 내부 함수
    h, w = color_img_shape[:2]
    def get_projected_mask(height_threshold):
        indices = np.where(signed_height > height_threshold)[0]
        obj_pts = points[indices]
        mask_2d = np.zeros((h, w), dtype=np.uint8)
        if len(obj_pts) > 0:
            x_3d, y_3d = obj_pts[:, 0], obj_pts[:, 1]
            z_3d = np.where(obj_pts[:, 2] == 0, 0.00001, obj_pts[:, 2])
            u_coords = np.round((x_3d * intrinsics.fx / z_3d) + intrinsics.ppx).astype(int)
            v_coords = np.round((y_3d * intrinsics.fy / z_3d) + intrinsics.ppy).astype(int)
            valid = (u_coords >= 0) & (u_coords < w) & (v_coords >= 0) & (v_coords < h)
            mask_2d[v_coords[valid], u_coords[valid]] = 1
        return mask_2d

    mask_5mm_2d = get_projected_mask(0.005)
    mask_40mm_2d = get_projected_mask(0.040)
    
    # 3D 융합 시 사용할 데이터 패키지
    pcd_data = {"points": points, "signed_height": signed_height}
    plane_data = {"normal": plane_normal, "d": d}

    return mask_5mm_2d, mask_40mm_2d, pcd_data, plane_data, floor_pcd

def fuse_yolo_and_generate_3d_obbs(detected_objects, refined_mask_01, mask_40mm_2d, pcd_data, plane_data, intrinsics, color_img_rgb, floor_pcd=None):
    """
    [STEP 4~5] YOLO와 3D 마스크 융합, ID 교정, 최저 높이 객체 판별 및 바닥 밀착형 3D OBB 생성
    """
    print("\n[INFO] YOLO 융합, ID 교정 및 3D 바운딩 박스 생성 중...")
    
    h, w = color_img_rgb.shape[:2]
    vis_image = color_img_rgb.copy()
    mask_high_vis = np.zeros((h, w), dtype=np.uint8)
    mask_low_vis = np.zeros((h, w), dtype=np.uint8)

    points = pcd_data["points"]
    signed_height = pcd_data["signed_height"]
    plane_normal = plane_data["normal"]
    d = plane_data["d"]

    # 3D -> 2D 맵핑 인덱스 준비
    z_3d_safe = np.where(points[:, 2] == 0, 0.00001, points[:, 2])
    u_all = np.round((points[:, 0] * intrinsics.fx / z_3d_safe) + intrinsics.ppx).astype(int)
    v_all = np.round((points[:, 1] * intrinsics.fy / z_3d_safe) + intrinsics.ppy).astype(int)
    valid_idx = (u_all >= 0) & (u_all < w) & (v_all >= 0) & (v_all < h)
    u_valid, v_valid = u_all[valid_idx], v_all[valid_idx]
    points_valid = points[valid_idx]
    heights_valid = signed_height[valid_idx]

    vis_elements_3d = [floor_pcd] if floor_pcd is not None else []
    overlay_geometries_3d = []
    object_data_list = []
    color_dict = {}
    cmap = cm.get_cmap("tab20")
    N = 1.5

    # 1. 융합 및 데이터 수집
    for obj in detected_objects:
        yolo_mask = obj["mask"].astype(np.uint8)
        fused_mask = np.logical_and(yolo_mask > 0, refined_mask_01 > 0).astype(np.uint8)
        
        contours, _ = cv2.findContours(fused_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
            
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        rect_w, rect_h = rect[1]
        ratio = max(rect_w, rect_h) / min(rect_w, rect_h) if min(rect_w, rect_h) > 0 else 0
        
        overlap_ratio = np.count_nonzero(np.logical_and(fused_mask, mask_40mm_2d)) / np.count_nonzero(fused_mask)
        if "original_name" not in obj: obj["original_name"] = obj["class_name"]
        old_name = obj["original_name"]
        
        # ID 교정 로직
        if overlap_ratio > 0.20:
            mask_high_vis = np.logical_or(mask_high_vis, fused_mask).astype(np.uint8)
            obj["class_name"] = f"[C]{old_name.replace('2x2', '4x2')}" if "2x2" in old_name else f"[C]{old_name}"
        else:
            mask_low_vis = np.logical_or(mask_low_vis, fused_mask).astype(np.uint8)
            obj["class_name"] = f"[C]{old_name.replace('4x2', '2x2').replace('2x4', '2x2')}" if ("4x2" in old_name or "2x4" in old_name) and ratio <= N else f"[C]{old_name}"

        # 최대 높이 계산
        in_mask_pixels = fused_mask[v_valid, u_valid] > 0
        obj_heights = heights_valid[in_mask_pixels]
        if len(obj_heights) > 0:
            max_h = max(np.percentile(obj_heights, 95), 0.005)
            object_data_list.append({"final_id": obj["class_name"], "rect": rect, "in_mask_pixels": in_mask_pixels, "max_h": max_h})

    # 2. 최저 높이 강제 고정 및 3D 객체 생성
    if object_data_list:
        min_idx = min(range(len(object_data_list)), key=lambda i: object_data_list[i]["max_h"])
        old_h = object_data_list[min_idx]["max_h"]
        object_data_list[min_idx]["max_h"] = 0.024
        print(f" 🎯 [높이 강제 고정] 가장 낮은 객체('{object_data_list[min_idx]['final_id']}') 높이: {old_h*1000:.1f}mm -> 24.0mm")

    for data in object_data_list:
        final_id, rect, in_mask_pixels, max_h = data["final_id"], data["rect"], data["in_mask_pixels"], data["max_h"]
        
        if final_id not in color_dict: color_dict[final_id] = cmap(len(color_dict) % 20)[:3]
        obj_color = color_dict[final_id]

        # 3D 클러스터 색상 칠하기
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(points_valid[in_mask_pixels])
        obj_pcd.paint_uniform_color(obj_color)
        vis_elements_3d.append(obj_pcd)

        # 3D OBB 및 좌표계
        box_2d = np.intp(cv2.boxPoints(rect))
        box_3d, axes_3d = create_floor_anchored_3d_box_with_axes(box_2d, intrinsics, plane_normal, d, max_h, obj_color, axis_size=0.03)
        
        vis_elements_3d.extend([box_3d, axes_3d])
        overlay_geometries_3d.extend([box_3d, axes_3d])

        # 2D 시각화 (color_img_rgb 기준이므로 빨간색은 (255,0,0)으로 그립니다)
        cv2.drawContours(vis_image, [box_2d], 0, (255, 0, 0), 2)
        top_point = box_2d[np.argmin(box_2d[:, 1])]
        cv2.putText(vis_image, final_id, (top_point[0] - 20, top_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    return detected_objects, vis_elements_3d, overlay_geometries_3d, vis_image, mask_high_vis, mask_low_vis

def visualize_final_rgbd_pointcloud(color_img_rgb, depth_img, intrinsics, depth_scale, refined_mask_01, mask_high_vis, vis_image_2d, vis_elements_3d, overlay_geometries_3d):
    """
    [STEP 6] 2D Matplotlib 시각화 및 두 개의 Open3D 뷰어(분석용, RGBD 오버레이용)를 순차적으로 띄웁니다.
    """
    # 1. 2D 결과 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(refined_mask_01, cmap="gray")
    axes[0].set_title("1. Refined Depth Mask (Convex Hull)")
    axes[0].axis("off")
    axes[1].imshow(mask_high_vis, cmap="gray")
    axes[1].set_title("2. Stacked Objects (> 40mm)")
    axes[1].axis("off")
    axes[2].imshow(cv2.cvtColor(vis_image_2d,cv2.COLOR_BGR2RGB))
    axes[2].set_title("3. Fused YOLO+Depth & OBB")
    axes[2].axis("off")
    plt.tight_layout()
    plt.show(block=False)

    # 2. 첫 번째 3D 뷰어 (분석용 색상 클러스터)
    print("\n[INFO] 1번 창(분석용 색상 클러스터)을 엽니다. 창을 닫으면 원본 맵이 열립니다.")
    o3d.visualization.draw_geometries(vis_elements_3d, window_name="1. Analytical 3D Clustered Objects & OBBs")

    # 3. 두 번째 3D 뷰어 (RGB-D 오버레이)
    print("[INFO] 2번 창(원본 RGB-D 오버레이)을 엽니다.")
    color_o3d = o3d.geometry.Image(cv2.cvtColor(color_img_rgb,cv2.COLOR_BGR2RGB))
    depth_o3d_raw = o3d.geometry.Image(cv2.medianBlur(depth_img, 5))
    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        int(intrinsics.width), int(intrinsics.height), float(intrinsics.fx), float(intrinsics.fy), float(intrinsics.ppx), float(intrinsics.ppy)
    )
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d_raw, depth_scale=1.0/float(depth_scale), depth_trunc=1.5, convert_rgb_to_intensity=False
    )
    rgb_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d_intr)
    rgb_pcd = rgb_pcd.voxel_down_sample(voxel_size=0.0015)
    
    final_overlay_elements = [rgb_pcd] + overlay_geometries_3d
    o3d.visualization.draw_geometries(final_overlay_elements, window_name="2. Real RGB-D Point Cloud with OBBs & Axes")

def fill_object_mask_holes(mask):
    """
    객체 segmentation mask 내부를 외곽 contour 기준으로 채움.
    mask: 0/1 또는 0/255 uint8
    return: 0/1 uint8
    """
    mask_01 = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_01,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filled = np.zeros_like(mask_01, dtype=np.uint8)

    if len(contours) > 0:
        cv2.drawContours(
            filled,
            contours,
            contourIdx=-1,
            color=1,
            thickness=-1
        )

    return filled

def visualize_id_correction_and_final_segments(
    color_img_bgr,
    final_objects_before,
    final_objects_after,
    mask_40mm_2d,
    mask_before,
    mask_hull_after,
    show=True,
    rng_seed=42,
    figsize_compare=(20, 12),
    figsize_overlay=(12, 8)
):
    """
    ID 교정 전/후 OBB 비교 + 40mm 이상 높이 마스크 + 변경 객체만 보기 +
    최종 instance segmentation overlay 시각화 함수.

    Args:
        color_img_bgr (np.ndarray): BGR 원본 이미지
        final_objects_before (list[dict]): ID 교정 전 객체 리스트
        final_objects_after (list[dict]): ID 교정 후 객체 리스트
        mask_40mm_2d (np.ndarray): 40mm 이상 돌출 마스크
        show (bool): True면 plt.show() 실행
        rng_seed (int): instance mask 색상 고정용 seed
        figsize_compare (tuple): 전후 비교 figure 크기
        figsize_overlay (tuple): 최종 segmentation overlay figure 크기

    Returns:
        dict:
            changed_indices
            vis_before
            vis_after
            changed_mask
            changed_only_bgr
            final_combined_mask
            instance_color_layer
            final_overlay
    """

    # ============================================================
    # 1. 내부 함수: 객체 리스트 OBB + class_name 시각화
    # ============================================================
    def draw_objects_for_id_compare(
        image_bgr,
        objects,
        changed_indices=None,
        before_objects=None,
        after_objects=None
    ):
        vis = image_bgr.copy()

        if changed_indices is None:
            changed_indices = set()

        for idx, obj in enumerate(objects):
            yolo_mask = obj["mask"].astype(np.uint8)

            contours, _ = cv2.findContours(
                yolo_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            box = np.intp(cv2.boxPoints(rect))

            rect_w, rect_h = rect[1]
            ratio = max(rect_w, rect_h) / min(rect_w, rect_h) if min(rect_w, rect_h) > 0 else 0

            is_changed = idx in changed_indices

            box_color = (0, 0, 255) if is_changed else (0, 255, 0)
            text_color = (0, 255, 255) if is_changed else (255, 255, 255)
            thickness = 3 if is_changed else 1

            cv2.drawContours(vis, [box], 0, box_color, thickness)

            top_point = box[np.argmin(box[:, 1])]
            x_text = int(top_point[0] - 25)
            y_text = int(top_point[1] - 35)

            if is_changed and before_objects is not None and after_objects is not None:
                before_name = before_objects[idx]["class_name"]
                after_name = after_objects[idx]["class_name"]
                label = f"{idx}: {before_name} -> {after_name}"
            else:
                label = f"{idx}: {obj['class_name']}"

            cv2.putText(
                vis,
                label,
                (x_text, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                text_color,
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                vis,
                f"R:{ratio:.2f}",
                (x_text, y_text + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return vis

    # ============================================================
    # 2. 변경된 객체 index 찾기
    # ============================================================
    changed_indices = []

    for i, (before_obj, after_obj) in enumerate(zip(final_objects_before, final_objects_after)):
        before_name = before_obj["class_name"]
        after_name = after_obj["class_name"]

        if before_name != after_name:
            changed_indices.append(i)

    changed_indices = set(changed_indices)

    print("\n[ID 변경 목록]")
    if len(changed_indices) == 0:
        print("변경된 객체 없음")
    else:
        for idx in sorted(changed_indices):
            print(
                f" - index {idx}: "
                f"{final_objects_before[idx]['class_name']} -> {final_objects_after[idx]['class_name']}"
            )

    # ============================================================
    # 3. 전/후 이미지 생성
    # ============================================================
    vis_before = draw_objects_for_id_compare(
        image_bgr=color_img_bgr,
        objects=final_objects_before,
        changed_indices=changed_indices,
        before_objects=final_objects_before,
        after_objects=final_objects_after
    )

    vis_after = draw_objects_for_id_compare(
        image_bgr=color_img_bgr,
        objects=final_objects_after,
        changed_indices=changed_indices,
        before_objects=final_objects_before,
        after_objects=final_objects_after
    )

    # ============================================================
    # 4. 변경된 객체 마스크만 따로 생성
    # ============================================================
    changed_mask = np.zeros(color_img_bgr.shape[:2], dtype=np.uint8)

    for idx in changed_indices:
        changed_mask = np.logical_or(
            changed_mask,
            final_objects_after[idx]["mask"].astype(bool)
        ).astype(np.uint8)

    changed_mask_255 = (changed_mask * 255).astype(np.uint8)

    changed_only_bgr = cv2.bitwise_and(
        color_img_bgr,
        color_img_bgr,
        mask=changed_mask_255
    )

    # ============================================================
    # 5. 최종 segmentation overlay 생성
    # ============================================================
    vis = color_img_bgr.copy()
    instance_color_layer = np.zeros_like(color_img_bgr)
    final_combined_mask = np.zeros(color_img_bgr.shape[:2], dtype=np.uint8)

    rng = np.random.default_rng(rng_seed)

    for idx, obj in enumerate(final_objects_after):
        mask = obj["mask"].astype(np.uint8)
        mask_bool = mask.astype(bool)

        final_combined_mask = np.logical_or(
            final_combined_mask,
            mask_bool
        ).astype(np.uint8)

        color = rng.integers(50, 255, size=3).tolist()  # BGR
        instance_color_layer[mask_bool] = color

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(vis, [largest_contour], -1, (0, 0, 255), 2)

        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cx, cy = x, y

        label = f"{idx}: {obj['class_name']}"

        cv2.putText(
            vis,
            label,
            (cx - 30, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    final_overlay = cv2.addWeighted(vis, 0.65, instance_color_layer, 0.35, 0)

    color_before_bgr = cv2.bitwise_and(
        color_img_bgr,
        color_img_bgr,
        mask=(mask_before * 255).astype(np.uint8)
    )

    color_hull_bgr = cv2.bitwise_and(
        color_img_bgr,
        color_img_bgr,
        mask=(mask_hull_after * 255).astype(np.uint8)
    )

    # ============================================================
    # 6. 시각화
    # ============================================================
    if show:
        plt.figure(figsize=figsize_compare)

        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(vis_before, cv2.COLOR_BGR2RGB))
        plt.title("Before ID Correction")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(vis_after, cv2.COLOR_BGR2RGB))
        plt.title("After ID Correction")
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.imshow(mask_40mm_2d, cmap="hot")
        plt.title("High Object Mask > 40mm")
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(changed_only_bgr, cv2.COLOR_BGR2RGB))
        plt.title("Changed Objects Only")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=figsize_overlay)
        plt.imshow(cv2.cvtColor(final_overlay, cv2.COLOR_BGR2RGB))
        plt.title("Final Segmented Objects with Labels")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


        plt.figure(figsize=(18, 8))

        plt.subplot(2, 2, 1)
        plt.imshow(mask_before, cmap="gray")
        plt.title("Before Hull Mask")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.imshow(mask_hull_after, cmap="gray")
        plt.title("After Convex Hull Mask")
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(color_before_bgr, cv2.COLOR_BGR2RGB))
        plt.title("Color AND Before Hull")
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(color_hull_bgr, cv2.COLOR_BGR2RGB))
        plt.title("Color AND After Convex Hull")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    # ============================================================
    # 7. 출력 반환
    # ============================================================
    return {
        "changed_indices": changed_indices,
        "vis_before": vis_before,
        "vis_after": vis_after,
        "changed_mask": changed_mask,
        "changed_only_bgr": changed_only_bgr,
        "final_combined_mask": final_combined_mask,
        "instance_color_layer": instance_color_layer,
        "final_overlay": final_overlay,
    }

def add_side2_suffix_for_high_corrected_objects(
    objects_before,
    objects_after,
    mask_40mm_2d,
    overlap_threshold=0.20,
    only_changed=True,
    remove_c_prefix=True
):
    """
    40mm 이상 돌출 마스크와 overlap이 큰 객체에 '_side2' suffix를 붙임.

    only_changed=True:
        ID가 실제로 변경된 객체 중에서만 side2 부여
        예: 2x2_red -> 4x2_red_side2

    only_changed=False:
        40mm 이상 overlap이 큰 모든 객체에 side2 부여
        예: 기존 4x2_red도 4x2_red_side2 가능
    """

    mask_high_bool = mask_40mm_2d > 0

    for idx, (before_obj, after_obj) in enumerate(zip(objects_before, objects_after)):
        before_name = before_obj["class_name"]
        after_name = after_obj["class_name"]

        # [C] prefix 제거하고 싶으면 제거
        if remove_c_prefix:
            before_name_clean = before_name.replace("[C]", "")
            after_name_clean = after_name.replace("[C]", "")
        else:
            before_name_clean = before_name
            after_name_clean = after_name

        obj_mask = after_obj["mask"] > 0
        obj_area = np.count_nonzero(obj_mask)

        if obj_area == 0:
            after_obj["class_name"] = after_name_clean
            after_obj["is_side2"] = False
            after_obj["height_overlap_ratio"] = 0.0
            continue

        overlap = np.logical_and(obj_mask, mask_high_bool)
        overlap_ratio = np.count_nonzero(overlap) / obj_area

        is_high = overlap_ratio > overlap_threshold
        is_changed = before_name_clean != after_name_clean

        add_side2 = is_high and (is_changed if only_changed else True)

        new_name = after_name_clean

        if add_side2:
            if not new_name.endswith("_side2"):
                new_name = f"{new_name}_side2"

            print(
                f" 🧱 [SIDE2 부여] index {idx}: "
                f"{before_name_clean} -> {new_name} "
                f"(overlap={overlap_ratio:.3f})"
            )

        after_obj["class_name"] = new_name
        after_obj["is_side2"] = add_side2
        after_obj["height_overlap_ratio"] = overlap_ratio

    return objects_after

def fill_object_mask_by_convex_hull(mask, min_area=20):
    """
    객체 mask의 외곽 contour를 잡고 convex hull로 내부를 채움.

    Args:
        mask: bool, 0/1, 0/255 형태 모두 가능
        min_area: 너무 작은 contour 제거 기준

    Returns:
        hull_mask: 0/1 uint8
    """
    mask_01 = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_01,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    hull_mask = np.zeros_like(mask_01, dtype=np.uint8)

    if len(contours) == 0:
        return hull_mask

    # 여러 조각이 있으면 너무 작은 조각은 제거하고 hull 생성
    valid_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            valid_contours.append(cnt)

    if len(valid_contours) == 0:
        return hull_mask

    # 가장 큰 contour만 쓰는 버전
    largest_contour = max(valid_contours, key=cv2.contourArea)

    hull = cv2.convexHull(largest_contour)

    cv2.drawContours(
        hull_mask,
        [hull],
        contourIdx=-1,
        color=1,
        thickness=-1
    )

    return hull_mask

def build_floor_scene_data_from_depth(
    depth_img,
    intrinsics,
    depth_scale,
    object_mask_01,
    depth_trunc=5.0,
    voxel_size=0.003,
    plane_dist_thresh=0.015,
    floor_height_eps=0.005,
    visualize=False
):
    """
    Convex Hull 등으로 만든 객체 마스크를 제외하고 바닥 RANSAC을 다시 수행.
    이후 전체 point cloud에 대해 signed_height를 계산하고,
    바닥 포인트는 plane 위로 투영해서 매끈한 floor_pcd를 생성.

    Returns:
        pcd_data:
            {
                "points": 전체 points,
                "signed_height": 바닥 기준 signed height,
                "valid_uv": (u_all, v_all, valid_idx)
            }

        plane_data:
            {
                "normal": 객체 방향으로 향하는 plane normal,
                "d": plane equation d,
                "plane_model": (a,b,c,d)
            }

        floor_pcd:
            바닥으로 판정된 점들을 plane 위에 투영한 Open3D point cloud
    """

    if visualize:
        print("\n[INFO] Convex Hull 객체 마스크 제외 후 바닥 평면 재추정 중...")

    h, w = depth_img.shape[:2]

    object_mask_01 = (object_mask_01 > 0).astype(np.uint8)

    # 객체 영역 확장해서 바닥 추정에서 제외
    kernel = np.ones((7, 7), np.uint8)
    expanded_obj_mask = cv2.dilate(object_mask_01, kernel, iterations=3)

    filtered_depth = cv2.medianBlur(depth_img, 5)

    bg_depth = filtered_depth.copy()
    bg_depth[expanded_obj_mask > 0] = 0

    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        int(intrinsics.width),
        int(intrinsics.height),
        float(intrinsics.fx),
        float(intrinsics.fy),
        float(intrinsics.ppx),
        float(intrinsics.ppy)
    )

    o3d_depth_scale = 1.0 / float(depth_scale)

    # ------------------------------------------------------------
    # 1. 바닥 후보 point cloud
    # ------------------------------------------------------------
    bg_depth_o3d = o3d.geometry.Image(bg_depth)

    bg_pcd = o3d.geometry.PointCloud.create_from_depth_image(
        bg_depth_o3d,
        o3d_intr,
        depth_scale=o3d_depth_scale,
        depth_trunc=depth_trunc
    )

    bg_pcd = bg_pcd.voxel_down_sample(voxel_size=voxel_size)

    if len(bg_pcd.points) < 30:
        raise RuntimeError(f"바닥 후보 포인트가 너무 적습니다: {len(bg_pcd.points)}")

    bg_pcd, _ = bg_pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )

    if len(bg_pcd.points) < 3:
        raise RuntimeError("RANSAC 수행 가능한 바닥 포인트가 부족합니다.")

    # ------------------------------------------------------------
    # 2. RANSAC plane
    # ------------------------------------------------------------
    raw_plane_model, inliers = bg_pcd.segment_plane(
        distance_threshold=plane_dist_thresh,
        ransac_n=3,
        num_iterations=1000
    )

    a, b, c, d = raw_plane_model
    n_raw = np.array([a, b, c], dtype=np.float64)
    norm = np.linalg.norm(n_raw)

    if norm < 1e-9:
        raise RuntimeError("RANSAC plane normal이 비정상입니다.")

    n_raw = n_raw / norm
    d_raw = d / norm

    # ------------------------------------------------------------
    # 3. 전체 point cloud
    # ------------------------------------------------------------
    full_depth_o3d = o3d.geometry.Image(filtered_depth)

    full_pcd = o3d.geometry.PointCloud.create_from_depth_image(
        full_depth_o3d,
        o3d_intr,
        depth_scale=o3d_depth_scale,
        depth_trunc=depth_trunc
    )

    full_pcd = full_pcd.voxel_down_sample(voxel_size=voxel_size)

    points = np.asarray(full_pcd.points)

    if len(points) == 0:
        raise RuntimeError("전체 point cloud가 비어 있습니다.")

    # ------------------------------------------------------------
    # 4. 3D point -> 2D pixel projection
    # ------------------------------------------------------------
    z_safe = np.where(points[:, 2] == 0, 1e-6, points[:, 2])

    u_all = np.round((points[:, 0] * intrinsics.fx / z_safe) + intrinsics.ppx).astype(int)
    v_all = np.round((points[:, 1] * intrinsics.fy / z_safe) + intrinsics.ppy).astype(int)

    valid_idx = (
        (u_all >= 0) & (u_all < w) &
        (v_all >= 0) & (v_all < h)
    )

    # ------------------------------------------------------------
    # 5. normal 방향 선택
    #    객체 마스크 내부 포인트들의 height가 양수가 되도록 normal 방향 결정
    # ------------------------------------------------------------
    u_valid = u_all[valid_idx]
    v_valid = v_all[valid_idx]
    points_valid = points[valid_idx]

    in_obj = object_mask_01[v_valid, u_valid] > 0

    h1 = np.dot(points_valid, n_raw) + d_raw
    h2 = np.dot(points_valid, -n_raw) - d_raw

    if np.count_nonzero(in_obj) > 10:
        score1 = np.percentile(h1[in_obj], 90)
        score2 = np.percentile(h2[in_obj], 90)

        if score2 > score1:
            plane_normal = -n_raw
            plane_d = -d_raw
        else:
            plane_normal = n_raw
            plane_d = d_raw
    else:
        # 객체 포인트가 부족하면 카메라 방향 기준으로 normal z가 음수가 되게 설정
        # top-down RealSense 기준: 객체 돌출 방향은 대체로 -Z
        if n_raw[2] > 0:
            plane_normal = -n_raw
            plane_d = -d_raw
        else:
            plane_normal = n_raw
            plane_d = d_raw

    signed_height = np.dot(points, plane_normal) + plane_d

    print(
        f"✅ Plane: "
        f"{plane_normal[0]:.4f}x + {plane_normal[1]:.4f}y + "
        f"{plane_normal[2]:.4f}z + {plane_d:.4f} = 0"
    )

    # ------------------------------------------------------------
    # 6. 매끈한 바닥 point cloud 생성
    # ------------------------------------------------------------
    floor_indices = np.where(np.abs(signed_height) <= floor_height_eps)[0]

    floor_points = points[floor_indices]

    if len(floor_points) > 0:
        floor_dist = np.dot(floor_points, plane_normal) + plane_d
        floor_points_flat = floor_points - np.outer(floor_dist, plane_normal)

        floor_pcd = o3d.geometry.PointCloud()
        floor_pcd.points = o3d.utility.Vector3dVector(floor_points_flat)
        floor_pcd.paint_uniform_color([0.75, 0.75, 0.75])
    else:
        floor_pcd = None
        print("⚠️ floor_pcd 생성용 바닥 포인트가 부족합니다.")

    pcd_data = {
        "points": points,
        "signed_height": signed_height,
        "u_all": u_all,
        "v_all": v_all,
        "valid_idx": valid_idx
    }

    plane_data = {
        "normal": plane_normal,
        "d": plane_d,
        "plane_model": (
            float(plane_normal[0]),
            float(plane_normal[1]),
            float(plane_normal[2]),
            float(plane_d)
        )
    }

    if visualize:
        geoms = []
        if floor_pcd is not None:
            geoms.append(floor_pcd)

        inlier_cloud = bg_pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([0.2, 0.8, 0.2])
        geoms.append(inlier_cloud)

        o3d.visualization.draw_geometries(
            geoms,
            window_name="Re-estimated Smooth Floor"
        )

    return pcd_data, plane_data, floor_pcd

def project_pixel_to_plane(u, v, intrinsics, plane_normal, plane_d):
    """
    픽셀 좌표 u,v에서 나가는 camera ray와 plane의 교점을 계산.
    Open3D camera coordinate 기준.
    """
    x = (u - intrinsics.ppx) / intrinsics.fx
    y = (v - intrinsics.ppy) / intrinsics.fy
    ray = np.array([x, y, 1.0], dtype=np.float64)

    denom = np.dot(plane_normal, ray)

    if abs(denom) < 1e-9:
        return None

    t = -plane_d / denom

    if t <= 0:
        return None

    return ray * t

def create_floor_anchored_box_lineset(
    box_2d,
    intrinsics,
    plane_normal,
    plane_d,
    height,
    color=(1.0, 0.0, 0.0)
):
    """
    2D minAreaRect box 4점을 바닥 plane에 투영한 뒤,
    plane_normal 방향으로 height만큼 올려 3D OBB line set 생성.

    Returns:
        line_set
        center_3d
        floor_center_3d
        corners_3d
    """

    floor_corners = []

    for p in box_2d:
        u, v = int(p[0]), int(p[1])
        p3d = project_pixel_to_plane(
            u,
            v,
            intrinsics,
            plane_normal,
            plane_d
        )

        if p3d is None:
            return None, None, None, None

        floor_corners.append(p3d)

    floor_corners = np.asarray(floor_corners, dtype=np.float64)

    top_corners = floor_corners + plane_normal.reshape(1, 3) * float(height)

    corners = np.vstack([floor_corners, top_corners])

    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    colors = [color for _ in lines]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    floor_center = np.mean(floor_corners, axis=0)
    center_3d = floor_center + plane_normal * (float(height) * 0.5)

    return line_set, center_3d, floor_center, corners

def generate_3d_obbs_from_hull_objects(
    objects,
    refined_mask_01,
    pcd_data,
    plane_data,
    intrinsics,
    color_img_rgb,
    floor_pcd=None,
    min_height=0.024,
    max_height_limit=0.12,
    height_percentile=95,
    visualize_2d=True
):
    """
    이미 ID 교정 + side2 처리 + Convex Hull 적용이 끝난 objects를 기준으로
    floor anchored 3D OBB 생성.

    objects:
        objects_hull 또는 objects_no_side2

    refined_mask_01:
        mask_hull_after 또는 mask_no_side2

    Returns:
        objects_out:
            각 obj에 obj["obb_3d"] 정보 추가

        vis_elements_3d:
            Open3D 분석용 geometry list

        overlay_geometries_3d:
            RGB-D overlay에 얹을 3D box list

        vis_2d_rgb:
            2D OBB 표시 이미지
    """

    if visualize_2d:
        print("\n[INFO] Convex Hull mask 기준 3D OBB 생성 중...")

    objects_out = copy.deepcopy(objects)

    h, w = color_img_rgb.shape[:2]

    points = pcd_data["points"]
    signed_height = pcd_data["signed_height"]
    u_all = pcd_data["u_all"]
    v_all = pcd_data["v_all"]
    valid_idx = pcd_data["valid_idx"]

    plane_normal = plane_data["normal"]
    plane_d = plane_data["d"]

    u_valid = u_all[valid_idx]
    v_valid = v_all[valid_idx]
    points_valid = points[valid_idx]
    heights_valid = signed_height[valid_idx]

    refined_mask_01 = (refined_mask_01 > 0).astype(np.uint8)

    vis_elements_3d = []
    if floor_pcd is not None:
        vis_elements_3d.append(floor_pcd)

    overlay_geometries_3d = []

    vis_2d_rgb = color_img_rgb.copy()

    cmap = cm.get_cmap("tab20")
    color_dict = {}

    obb_results = []

    for idx, obj in enumerate(objects_out):
        class_name = obj["class_name"]

        obj_mask = (obj["mask"] > 0).astype(np.uint8)

        fused_mask = np.logical_and(
            obj_mask > 0,
            refined_mask_01 > 0
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            fused_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            print(f"[SKIP] idx {idx}: contour 없음")
            obj["obb_3d"] = None
            continue

        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) < 20:
            print(f"[SKIP] idx {idx}: contour area 너무 작음")
            obj["obb_3d"] = None
            continue

        rect = cv2.minAreaRect(largest_contour)
        box_2d = np.intp(cv2.boxPoints(rect))

        # --------------------------------------------------------
        # 객체 mask 내부 3D point 추출
        # --------------------------------------------------------
        in_mask_pixels = fused_mask[v_valid, u_valid] > 0

        obj_points = points_valid[in_mask_pixels]
        obj_heights = heights_valid[in_mask_pixels]

        if len(obj_points) < 5:
            print(f"[SKIP] idx {idx}: 3D point 부족")
            obj["obb_3d"] = None
            continue

        # height 계산
        raw_h = np.percentile(obj_heights, height_percentile)

        # 음수/이상값 방어
        max_h = float(np.clip(raw_h, min_height, max_height_limit))

        # --------------------------------------------------------
        # 색상
        # --------------------------------------------------------
        if class_name not in color_dict:
            color_dict[class_name] = cmap(len(color_dict) % 20)[:3]

        obj_color = color_dict[class_name]

        # --------------------------------------------------------
        # 객체 point cloud
        # --------------------------------------------------------
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
        obj_pcd.paint_uniform_color(obj_color)
        vis_elements_3d.append(obj_pcd)

        # --------------------------------------------------------
        # floor anchored 3D box
        # --------------------------------------------------------
        box_3d, center_3d, floor_center_3d, corners_3d = create_floor_anchored_box_lineset(
            box_2d=box_2d,
            intrinsics=intrinsics,
            plane_normal=plane_normal,
            plane_d=plane_d,
            height=max_h,
            color=obj_color
        )

        if box_3d is None:
            print(f"[SKIP] idx {idx}: 2D box → plane projection 실패")
            obj["obb_3d"] = None
            continue

        vis_elements_3d.append(box_3d)
        overlay_geometries_3d.append(box_3d)

        # 중심점 sphere
        center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
        center_sphere.translate(center_3d)
        center_sphere.paint_uniform_color([1.0, 0.0, 0.0])
        vis_elements_3d.append(center_sphere)
        overlay_geometries_3d.append(center_sphere)

        obj["obb_3d"] = {
            "center_3d_m": center_3d,
            "center_3d_mm": center_3d * 1000.0,
            "floor_center_3d_m": floor_center_3d,
            "height_m": max_h,
            "height_mm": max_h * 1000.0,
            "box_2d": box_2d,
            "corners_3d_m": corners_3d,
            "num_points": len(obj_points)
        }

        obb_results.append({
            "idx": idx,
            "class_name": class_name,
            "center_3d_m": center_3d,
            "center_3d_mm": center_3d * 1000.0,
            "floor_center_3d_m": floor_center_3d,
            "height_m": max_h,
            "height_mm": max_h * 1000.0,
            "num_points": len(obj_points)
        })

        # --------------------------------------------------------
        # 2D 시각화
        # color_img_rgb 기준이라 빨강은 (255,0,0)
        # --------------------------------------------------------
        cv2.drawContours(vis_2d_rgb, [box_2d], 0, (255, 0, 0), 2)

        cx2d, cy2d = rect[0]
        cx2d, cy2d = int(cx2d), int(cy2d)

        label = f"{idx}: {class_name}"
        cv2.putText(
            vis_2d_rgb,
            label,
            (cx2d - 40, cy2d - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

        coord_text = (
            f"({center_3d[0]*1000:.1f}, "
            f"{center_3d[1]*1000:.1f}, "
            f"{center_3d[2]*1000:.1f})mm"
        )

        cv2.putText(
            vis_2d_rgb,
            coord_text,
            (cx2d - 55, cy2d + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    if visualize_2d:
        print("\n[3D OBB 중심 좌표]")
        for item in obb_results:
            c = item["center_3d_mm"]
            print(
                f" - idx {item['idx']:02d} | {item['class_name']} | "
                f"center(mm)=({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}) | "
                f"h={item['height_mm']:.1f}mm | points={item['num_points']}"
            )

        plt.figure(figsize=(12, 8))
        plt.imshow(vis_2d_rgb)
        plt.title("2D OBB + 3D Center Coordinates")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return objects_out, vis_elements_3d, overlay_geometries_3d, vis_2d_rgb, obb_results

def rotation_matrix_to_rpy_xyz_deg(R_mat):
    """
    Camera frame 기준 roll, pitch, yaw 계산.
    Convention:
        R_mat columns = [object_x, object_y, object_z] in camera coordinates
        Euler order = xyz
    """
    if SCIPY_AVAILABLE:
        rpy = R.from_matrix(R_mat).as_euler("xyz", degrees=True)
        return rpy  # roll, pitch, yaw

    # scipy 없을 때 fallback
    sy = np.sqrt(R_mat[0, 0] ** 2 + R_mat[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R_mat[2, 1], R_mat[2, 2])
        pitch = np.arctan2(-R_mat[2, 0], sy)
        yaw = np.arctan2(R_mat[1, 0], R_mat[0, 0])
    else:
        roll = np.arctan2(-R_mat[1, 2], R_mat[1, 1])
        pitch = np.arctan2(-R_mat[2, 0], sy)
        yaw = 0.0

    return np.rad2deg([roll, pitch, yaw])

def make_axes_lineset(center, R_obj_cam, axis_size=0.04):
    """
    Open3D 좌표축 LineSet 생성.
    X: red
    Y: green
    Z: blue
    """
    center = np.asarray(center, dtype=np.float64)

    x_axis = R_obj_cam[:, 0]
    y_axis = R_obj_cam[:, 1]
    z_axis = R_obj_cam[:, 2]

    points = np.array([
        center,
        center + x_axis * axis_size,
        center,
        center + y_axis * axis_size,
        center,
        center + z_axis * axis_size,
    ], dtype=np.float64)

    lines = [
        [0, 1],
        [2, 3],
        [4, 5],
    ]

    colors = [
        [1.0, 0.0, 0.0],  # X red
        [0.0, 1.0, 0.0],  # Y green
        [0.0, 0.2, 1.0],  # Z blue
    ]

    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector(points)
    axes.lines = o3d.utility.Vector2iVector(lines)
    axes.colors = o3d.utility.Vector3dVector(colors)

    return axes

def estimate_pose_axes_from_obb3d(
    obb_3d,
    plane_normal,
    class_name="unknown",
    axis_size=0.04
):
    """
    3D OBB corners를 기준으로 object coordinate frame 계산.

    Args:
        obb_3d:
            obj["obb_3d"] 딕셔너리
        plane_normal:
            plane_data["normal"]
        class_name:
            객체 이름
        axis_size:
            Open3D 좌표축 길이

    Returns:
        pose_data dict
    """

    def normalize_vec(v, eps=1e-9):
        v = np.asarray(v, dtype=np.float64)
        n = np.linalg.norm(v)
        if n < eps:
            return None
        return v / n

    if obb_3d is None:
        return None

    corners = np.asarray(obb_3d["corners_3d_m"], dtype=np.float64)
    center = np.asarray(obb_3d["center_3d_m"], dtype=np.float64)

    if corners.shape[0] < 8:
        return None

    # 바닥 4점
    floor_corners = corners[:4]

    # Z축: 바닥 normal
    z_axis = normalize_vec(plane_normal)
    if z_axis is None:
        return None

    # 바닥 사각형의 4개 edge 계산
    edge_candidates = []

    for i in range(4):
        p0 = floor_corners[i]
        p1 = floor_corners[(i + 1) % 4]
        e = p1 - p0

        # normal 성분 제거해서 바닥 평면 위 방향으로 보정
        e = e - np.dot(e, z_axis) * z_axis

        length = np.linalg.norm(e)
        if length > 1e-6:
            edge_candidates.append((length, e, i))

    if len(edge_candidates) == 0:
        return None

    # X축: 가장 긴 edge 방향
    edge_candidates.sort(key=lambda x: x[0], reverse=True)
    x_axis = normalize_vec(edge_candidates[0][1])

    if x_axis is None:
        return None

    # X축 방향 부호 안정화
    # 카메라 좌표계에서 x 성분이 양수가 되게 함.
    # 필요 없으면 이 블록 삭제 가능.
    if x_axis[0] < 0:
        x_axis = -x_axis

    # Y축: 오른손 좌표계
    y_axis = np.cross(z_axis, x_axis)
    y_axis = normalize_vec(y_axis)

    if y_axis is None:
        return None

    # X축 재직교화
    x_axis = np.cross(y_axis, z_axis)
    x_axis = normalize_vec(x_axis)

    # Rotation matrix
    # columns = object axes in camera frame
    R_obj_cam = np.column_stack([x_axis, y_axis, z_axis])

    # det 보정
    if np.linalg.det(R_obj_cam) < 0:
        y_axis = -y_axis
        R_obj_cam = np.column_stack([x_axis, y_axis, z_axis])

    rpy_deg = rotation_matrix_to_rpy_xyz_deg(R_obj_cam)

    axes_3d = make_axes_lineset(
        center=center,
        R_obj_cam=R_obj_cam,
        axis_size=axis_size
    )

    pose_data = {
        "class_name": class_name,
        "center_m": center,
        "center_mm": center * 1000.0,
        "R_obj_cam": R_obj_cam,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "roll_deg": float(rpy_deg[0]),
        "pitch_deg": float(rpy_deg[1]),
        "yaw_deg": float(rpy_deg[2]),
        "axes_3d": axes_3d,
    }

    return pose_data

def normalize_class_name(name, remove_c_prefix=True, remove_side2=False):
    """
    '[C]2x2_red_side2' 같은 이름을 정리.
    """
    name = str(name)

    if remove_c_prefix:
        name = name.replace("[C]", "")

    if remove_side2:
        name = name.replace("_side2", "")

    return name

def build_class_sorted_pose_index(
    objects_obb,
    use_pose_cam=True,
    remove_c_prefix=True,
    remove_side2=False,
    verbose=True
):
    """
    객체들을 클래스별로 묶고, 카메라 optical axis에서 가까운 순서대로 local_id 부여.

    기준:
        axis_dist_m = sqrt(x^2 + y^2)

    Args:
        objects_obb:
            obj["obb_3d"] 또는 obj["pose_cam"]이 들어있는 객체 리스트

        use_pose_cam:
            True면 obj["pose_cam"]["center_m"] 우선 사용
            False면 obj["obb_3d"]["center_3d_m"] 사용

    Returns:
        pose_table:
            list[dict], 모든 객체 pose 정보

        class_index:
            dict[class_name] = 해당 클래스 객체 리스트, axis_dist_m 오름차순
    """

    pose_table = []

    for global_idx, obj in enumerate(objects_obb):
        raw_name = obj.get("class_name", "unknown")
        class_name = normalize_class_name(
            raw_name,
            remove_c_prefix=remove_c_prefix,
            remove_side2=remove_side2
        )

        center_m = None
        roll_deg = None
        pitch_deg = None
        yaw_deg = None
        R_obj_cam = None

        # 1순위: pose_cam
        if use_pose_cam and obj.get("pose_cam", None) is not None:
            pose = obj["pose_cam"]
            center_m = np.asarray(pose["center_m"], dtype=np.float64)
            roll_deg = float(pose["roll_deg"])
            pitch_deg = float(pose["pitch_deg"])
            yaw_deg = float(pose["yaw_deg"])
            R_obj_cam = pose["R_obj_cam"]

        # 2순위: obb_3d
        elif obj.get("obb_3d", None) is not None:
            obb = obj["obb_3d"]
            center_m = np.asarray(obb["center_3d_m"], dtype=np.float64)

            # 아직 RPY가 없으면 None
            roll_deg = None
            pitch_deg = None
            yaw_deg = None
            R_obj_cam = None

        else:
            if verbose:
                print(f"[SKIP] global_idx {global_idx}: pose/obb 없음")
            continue

        x, y, z = center_m
        axis_dist_m = float(np.sqrt(x**2 + y**2))
        depth_m = float(z)

        item = {
            "global_idx": global_idx,
            "class_name": class_name,
            "raw_class_name": raw_name,
            "axis_dist_m": axis_dist_m,
            "axis_dist_mm": axis_dist_m * 1000.0,
            "depth_m": depth_m,
            "x_m": float(x),
            "y_m": float(y),
            "z_m": float(z),
            "x_mm": float(x * 1000.0),
            "y_mm": float(y * 1000.0),
            "z_mm": float(z * 1000.0),
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg,
            "R_obj_cam": R_obj_cam,
            "object_ref": obj,
        }

        pose_table.append(item)

    # 전체를 optical axis 거리 기준으로 정렬
    pose_table = sorted(pose_table, key=lambda x: x["axis_dist_m"])

    # 클래스별 묶기
    class_index = {}

    for item in pose_table:
        cls = item["class_name"]

        if cls not in class_index:
            class_index[cls] = []

        class_index[cls].append(item)

    # 클래스 내부 local_id 부여
    for cls, items in class_index.items():
        items.sort(key=lambda x: x["axis_dist_m"])

        for local_id, item in enumerate(items):
            item["local_id"] = local_id

    if verbose:
        print("\n[클래스별 optical axis 가까운 순서]")
        for cls, items in class_index.items():
            print(f"\nClass: {cls}")

            for item in items:
                print(
                    f"  local_id {item['local_id']:02d} | "
                    f"global_idx {item['global_idx']:02d} | "
                    f"axis_dist={item['axis_dist_mm']:.1f} mm | "
                    f"center=({item['x_mm']:.1f}, {item['y_mm']:.1f}, {item['z_mm']:.1f}) mm | "
                    f"RPY=({item['roll_deg']}, {item['pitch_deg']}, {item['yaw_deg']})"
                )

    return pose_table, class_index

def get_nearest_6d_pose_by_class(
    class_index,
    target_class_name,
    local_id=0,
    remove_c_prefix=True,
    remove_side2=False
):
    """
    클래스 이름으로 요청하면 optical axis 기준 가까운 순서 중 local_id번째 객체의 6D 반환.

    예:
        get_nearest_6d_pose_by_class(class_index, "2x2_red", local_id=0)
    """

    target = normalize_class_name(
        target_class_name,
        remove_c_prefix=remove_c_prefix,
        remove_side2=remove_side2
    )

    if target not in class_index:
        print(f"❌ 요청 클래스 없음: {target}")
        print("가능 클래스:", list(class_index.keys()))
        return None

    items = class_index[target]

    if local_id >= len(items):
        print(f"❌ {target} 클래스에 local_id {local_id} 없음. 개수: {len(items)}")
        return None

    item = items[local_id]

    result_6d = {
        "class_name": item["class_name"],
        "local_id": item["local_id"],
        "global_idx": item["global_idx"],

        # position
        "x_m": item["x_m"],
        "y_m": item["y_m"],
        "z_m": item["z_m"],

        "x_mm": item["x_mm"],
        "y_mm": item["y_mm"],
        "z_mm": item["z_mm"],

        # orientation
        "roll_deg": item["roll_deg"],
        "pitch_deg": item["pitch_deg"],
        "yaw_deg": item["yaw_deg"],

        # sorting metric
        "axis_dist_m": item["axis_dist_m"],
        "axis_dist_mm": item["axis_dist_mm"],

        # matrix
        "R_obj_cam": item["R_obj_cam"],
    }

    print("\n[요청 객체 6D Pose]")
    print(f"Class      : {result_6d['class_name']}")
    print(f"local_id   : {result_6d['local_id']}")
    print(f"global_idx : {result_6d['global_idx']}")
    print(
        f"Position mm: "
        f"x={result_6d['x_mm']:.1f}, "
        f"y={result_6d['y_mm']:.1f}, "
        f"z={result_6d['z_mm']:.1f}"
    )
    print(
        f"RPY deg    : "
        f"roll={result_6d['roll_deg']:.2f}, "
        f"pitch={result_6d['pitch_deg']:.2f}, "
        f"yaw={result_6d['yaw_deg']:.2f}"
    )
    print(f"Axis dist  : {result_6d['axis_dist_mm']:.1f} mm")

    return result_6d

def visualize_3d_obb_results(
    vis_3d,
    overlay_3d,
    color_rgb,
    depth,
    intrinsics,
    scale,
    show_analysis=True,
    show_rgbd_overlay=True,
    depth_trunc=5.0,
    rgbd_voxel_size=0.0015,
    median_ksize=5,
    analysis_window_name="Smooth Floor + Convex Hull Object 3D OBBs",
    overlay_window_name="RGB-D Point Cloud + Convex Hull 3D OBBs"
):
    """
    3D OBB 결과를 Open3D로 시각화.

    1) 분석용 창:
        - 매끈한 floor_pcd
        - 객체별 point cloud
        - 3D OBB
        - 중심점 sphere

    2) RGB-D overlay 창:
        - 원본 RGB-D point cloud
        - 3D OBB
        - 중심점 / 좌표축 등 overlay geometry

    Args:
        vis_3d:
            generate_3d_obbs_from_hull_objects()에서 반환된 분석용 geometry list.

        overlay_3d:
            generate_3d_obbs_from_hull_objects()에서 반환된 overlay geometry list.

        color_rgb:
            RealSense에서 받은 RGB 이미지. shape: H x W x 3.

        depth:
            RealSense raw depth 이미지. shape: H x W.

        intrinsics:
            RealSense aligned color intrinsics.

        scale:
            depth raw value를 meter로 바꾸는 scale.

        show_analysis:
            True면 분석용 Open3D 창 표시.

        show_rgbd_overlay:
            True면 RGB-D 원본 point cloud 위에 overlay 표시.

        depth_trunc:
            Open3D RGBD 생성 시 depth 최대 거리.

        rgbd_voxel_size:
            RGB-D point cloud downsample voxel 크기.

        median_ksize:
            depth medianBlur 커널 크기. None 또는 1이면 미적용.

    Returns:
        result:
            {
                "rgb_pcd": rgb_pcd,
                "final_overlay_elements": final_overlay_elements
            }
    """

    rgb_pcd = None
    final_overlay_elements = None

    # =================================================================
    # [1] Open3D 분석용 시각화
    # =================================================================
    if show_analysis:
        print("\n[INFO] 매끈한 바닥 + 객체 point cloud + 3D OBB 표시")

        o3d.visualization.draw_geometries(
            vis_3d,
            window_name=analysis_window_name
        )

    # =================================================================
    # [2] RGB-D 원본 point cloud 위에 OBB overlay
    # =================================================================
    if show_rgbd_overlay:
        print("\n[INFO] RGB-D 원본 point cloud 생성 중...")

        # color_rgb는 이미 RGB이므로 cvtColor 하지 않음
        color_o3d = o3d.geometry.Image(color_rgb)

        if median_ksize is not None and median_ksize >= 3:
            if median_ksize % 2 == 0:
                median_ksize += 1
            depth_for_o3d = cv2.medianBlur(depth, median_ksize)
        else:
            depth_for_o3d = depth.copy()

        depth_o3d = o3d.geometry.Image(depth_for_o3d)

        o3d_intr = o3d.camera.PinholeCameraIntrinsic(
            int(intrinsics.width),
            int(intrinsics.height),
            float(intrinsics.fx),
            float(intrinsics.fy),
            float(intrinsics.ppx),
            float(intrinsics.ppy)
        )

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1.0 / float(scale),
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )

        rgb_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d_intr
        )

        if rgbd_voxel_size is not None and rgbd_voxel_size > 0:
            rgb_pcd = rgb_pcd.voxel_down_sample(
                voxel_size=rgbd_voxel_size
            )

        final_overlay_elements = [rgb_pcd] + overlay_3d

        print("\n[INFO] RGB-D 원본 point cloud 위에 3D OBB overlay 표시")

        o3d.visualization.draw_geometries(
            final_overlay_elements,
            window_name=overlay_window_name
        )

    return {
        "rgb_pcd": rgb_pcd,
        "final_overlay_elements": final_overlay_elements
    }

def project_point_to_image(pt_3d, intrinsics):
    """
    3D point (camera frame, meter) -> 2D pixel
    """
    x, y, z = pt_3d
    if z <= 1e-9:
        return None

    u = int(round((x * intrinsics.fx / z) + intrinsics.ppx))
    v = int(round((y * intrinsics.fy / z) + intrinsics.ppy))
    return (u, v)

def visualize_class_pose_on_rgb(
    class_index,
    target_class_name,
    color_rgb,
    intrinsics,
    local_id=None,
    axis_size_m=0.03,
    show=True,
    show_roll_pitch=False,
    remove_c_prefix=True,
    remove_side2=False,
    line_thickness=2,
    font_scale=0.45,
    text_thickness=2
):
    """
    class_index에서 특정 클래스 객체를 골라
    RGB 이미지 위에 local_id / global_idx / XYZ / Yaw / 2D OBB / 2D axes를 그려서 보여줌.

    Args:
        class_index:
            build_class_sorted_pose_index()의 출력 dict

        target_class_name:
            예: "2x2_red"

        color_rgb:
            원본 RGB 이미지

        intrinsics:
            RealSense intrinsics

        local_id:
            None이면 해당 클래스의 모든 객체를 그림
            정수면 해당 local_id 하나만 그림

        axis_size_m:
            3D axes 길이(m)

        show:
            True면 plt.show()

        show_roll_pitch:
            True면 Yaw만 아니라 Roll/Pitch도 같이 표기

        remove_c_prefix:
            class_index 조회 시 [C] 제거 여부

        remove_side2:
            class_index 조회 시 _side2 제거 여부

    Returns:
        vis_rgb:
            시각화된 RGB 이미지

        selected_items:
            실제로 그린 객체 item 리스트
    """

    def normalize_class_name_for_query(name, remove_c_prefix=True, remove_side2=False):
        name = str(name)

        if remove_c_prefix:
            name = name.replace("[C]", "")

        if remove_side2:
            name = name.replace("_side2", "")

        return name

    query_name = normalize_class_name_for_query(
        target_class_name,
        remove_c_prefix=remove_c_prefix,
        remove_side2=remove_side2
    )

    if query_name not in class_index:
        raise ValueError(
            f"요청 클래스 '{query_name}' 없음. 가능한 클래스: {list(class_index.keys())}"
        )

    class_items = class_index[query_name]

    if local_id is None:
        selected_items = class_items
    else:
        matched = [item for item in class_items if item.get("local_id", None) == local_id]
        if len(matched) == 0:
            raise ValueError(
                f"클래스 '{query_name}'에 local_id={local_id} 없음. "
                f"가능한 local_id: {[item.get('local_id', None) for item in class_items]}"
            )
        selected_items = matched

    # cv2 그리기는 BGR가 편해서 내부적으로 변환
    vis_bgr = cv2.cvtColor(color_rgb.copy(), cv2.COLOR_RGB2BGR)

    # 클래스 내부에서 보기 좋게 색 다르게
    rng = np.random.default_rng(42)

    for k, item in enumerate(selected_items):
        obj = item["object_ref"]

        # ---------- 색상 ----------
        color = rng.integers(80, 255, size=3).tolist()
        color = tuple(int(c) for c in color)   # BGR로 사용

        # ---------- 기본 정보 ----------
        local_id_val = item.get("local_id", None)
        global_idx = item.get("global_idx", None)

        x_mm = item["x_mm"]
        y_mm = item["y_mm"]
        z_mm = item["z_mm"]

        yaw_deg = item["yaw_deg"]
        roll_deg = item["roll_deg"]
        pitch_deg = item["pitch_deg"]

        # ---------- 2D OBB ----------
        box_2d = None
        if obj.get("obb_3d", None) is not None:
            box_2d = np.asarray(obj["obb_3d"]["box_2d"], dtype=np.int32)

        if box_2d is not None and len(box_2d) == 4:
            cv2.drawContours(vis_bgr, [box_2d], 0, color, line_thickness)

            # 중심 텍스트 기준점
            cx = int(np.mean(box_2d[:, 0]))
            cy = int(np.mean(box_2d[:, 1]))
        else:
            # fallback: 중심 투영
            center_m = np.array([item["x_m"], item["y_m"], item["z_m"]], dtype=np.float64)
            center_uv = project_point_to_image(center_m, intrinsics)
            if center_uv is None:
                continue
            cx, cy = center_uv

        # ---------- 2D 중심점 ----------
        cv2.circle(vis_bgr, (cx, cy), 4, (0, 0, 255), -1)

        # ---------- 3D 좌표축을 2D로 투영 ----------
        if obj.get("pose_cam", None) is not None:
            pose = obj["pose_cam"]

            center_m = np.asarray(pose["center_m"], dtype=np.float64)
            R_obj_cam = np.asarray(pose["R_obj_cam"], dtype=np.float64)

            x_axis = R_obj_cam[:, 0]
            y_axis = R_obj_cam[:, 1]
            z_axis = R_obj_cam[:, 2]

            p_center = project_point_to_image(center_m, intrinsics)
            p_x = project_point_to_image(center_m + x_axis * axis_size_m, intrinsics)
            p_y = project_point_to_image(center_m + y_axis * axis_size_m, intrinsics)
            p_z = project_point_to_image(center_m + z_axis * axis_size_m, intrinsics)

            if p_center is not None:
                if p_x is not None:
                    cv2.line(vis_bgr, p_center, p_x, (0, 0, 255), 2)   # X = red
                    cv2.putText(vis_bgr, "X", (p_x[0] + 2, p_x[1] - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)

                if p_y is not None:
                    cv2.line(vis_bgr, p_center, p_y, (0, 255, 0), 2)   # Y = green
                    cv2.putText(vis_bgr, "Y", (p_y[0] + 2, p_y[1] - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2, cv2.LINE_AA)

                if p_z is not None:
                    cv2.line(vis_bgr, p_center, p_z, (255, 0, 0), 2)   # Z = blue
                    cv2.putText(vis_bgr, "Z", (p_z[0] + 2, p_z[1] - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2, cv2.LINE_AA)

        # ---------- 텍스트 ----------
        text_lines = [
            f"{query_name} | LID:{local_id_val} | GID:{global_idx}",
            f"XYZ(mm): ({x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f})",
        ]

        if show_roll_pitch:
            text_lines.append(
                f"RPY(deg): ({roll_deg:.1f}, {pitch_deg:.1f}, {yaw_deg:.1f})"
            )
        else:
            text_lines.append(
                f"YAW(deg): {yaw_deg:.1f}"
            )

        # 텍스트 위치
        tx = cx + 10
        ty = cy - 25

        for i, line in enumerate(text_lines):
            yy = ty + i * 18

            # 검정 외곽선
            cv2.putText(
                vis_bgr,
                line,
                (tx, yy),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                text_thickness + 1,
                cv2.LINE_AA
            )

            # 흰 글씨
            cv2.putText(
                vis_bgr,
                line,
                (tx, yy),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA
            )

    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

    if show:
        plt.figure(figsize=(14, 10))
        plt.imshow(vis_rgb)
        title = f"Class Pose Visualization: {query_name}"
        if local_id is not None:
            title += f" (local_id={local_id})"
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return vis_rgb, selected_items

def dilate_final_objects(
    final_obj_fine,
    image_shape,
    kernel_size=7,
    iterations=1,
    kernel_type="ellipse",
    visualize=False,
    color_img_bgr=None
):
    """
    final_obj_fine 안의 객체별 mask에 dilation 팽창 연산을 적용하는 함수.

    Args:
        final_obj_fine (list): 객체 리스트. 각 obj는 obj["mask"]를 가져야 함.
        image_shape (tuple): 이미지 크기. 예: color_img_bgr.shape[:2]
        kernel_size (int): 팽창 커널 크기. 3, 5, 7, 9 등 홀수 권장.
        iterations (int): dilation 반복 횟수.
        kernel_type (str): "ellipse" 또는 "rect".
        visualize (bool): 결과 시각화 여부.
        color_img_bgr (ndarray): 원본 BGR 이미지. overlay 시각화용.

    Returns:
        dilated_objects (list): dilation 적용된 객체 리스트
        mask_before_all (ndarray): dilation 전 전체 통합 마스크, 0/1
        mask_after_all (ndarray): dilation 후 전체 통합 마스크, 0/1
        vis_img (ndarray): dilation contour가 그려진 BGR 이미지 또는 None
    """

    h, w = image_shape[:2]

    # 원본 final_obj_fine을 보존하려고 deepcopy
    dilated_objects = copy.deepcopy(final_obj_fine)

    mask_before_all = np.zeros((h, w), dtype=np.uint8)
    mask_after_all = np.zeros((h, w), dtype=np.uint8)

    vis_img = color_img_bgr.copy() if color_img_bgr is not None else None

    # kernel size는 홀수 권장
    if kernel_size % 2 == 0:
        kernel_size += 1

    if kernel_type == "ellipse":
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )
    elif kernel_type == "rect":
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_size, kernel_size)
        )
    else:
        raise ValueError("kernel_type은 'ellipse' 또는 'rect'만 가능합니다.")

    for idx, obj in enumerate(dilated_objects):
        if "mask" not in obj or obj["mask"] is None:
            if visualize:
                print(f"[WARNING] obj {idx}: mask 없음. skip")
            continue

        original_mask = (np.asarray(obj["mask"]) > 0).astype(np.uint8)

        # 혹시 크기 안 맞으면 보정
        if original_mask.shape[:2] != (h, w):
            original_mask = cv2.resize(
                original_mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )
            original_mask = (original_mask > 0).astype(np.uint8)

        # dilation 적용
        dilated_mask = cv2.dilate(
            original_mask,
            kernel,
            iterations=iterations
        )
        dilated_mask = (dilated_mask > 0).astype(np.uint8)

        # 객체별 mask 교체
        obj["mask_before_dilation"] = original_mask.astype(bool)
        obj["mask"] = dilated_mask.astype(bool)

        # 디버그 정보 저장
        obj["dilation_kernel_size"] = kernel_size
        obj["dilation_iterations"] = iterations
        obj["mask_area_before_dilation"] = int(np.count_nonzero(original_mask))
        obj["mask_area_after_dilation"] = int(np.count_nonzero(dilated_mask))

        # 전체 마스크 누적
        mask_before_all = np.logical_or(
            mask_before_all,
            original_mask > 0
        ).astype(np.uint8)

        mask_after_all = np.logical_or(
            mask_after_all,
            dilated_mask > 0
        ).astype(np.uint8)

        if visualize:
            print(
                f"[DILATE] obj {idx} | {obj.get('class_name', 'N/A')} | "
                f"area {np.count_nonzero(original_mask)} -> {np.count_nonzero(dilated_mask)} px | "
                f"kernel={kernel_size}, iter={iterations}"
            )

        # overlay용 contour
        if vis_img is not None:
            dilated_255 = (dilated_mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                dilated_255,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)

            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(largest_contour)

                cv2.putText(
                    vis_img,
                    f"{idx}:{obj.get('class_name', 'N/A')}",
                    (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1
                )

    if visualize:
        before_vis = mask_before_all * 255
        after_vis = mask_after_all * 255

        if color_img_bgr is not None:
            color_img_rgb = cv2.cvtColor(color_img_bgr, cv2.COLOR_BGR2RGB)
            vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

            fig, axes = plt.subplots(1, 4, figsize=(20, 6))

            axes[0].imshow(color_img_rgb)
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(before_vis, cmap="gray")
            axes[1].set_title("Before Dilation")
            axes[1].axis("off")

            axes[2].imshow(after_vis, cmap="gray")
            axes[2].set_title(f"After Dilation k={kernel_size}, iter={iterations}")
            axes[2].axis("off")

            axes[3].imshow(vis_img_rgb)
            axes[3].set_title("Dilated Contours Overlay")
            axes[3].axis("off")

        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].imshow(before_vis, cmap="gray")
            axes[0].set_title("Before Dilation")
            axes[0].axis("off")

            axes[1].imshow(after_vis, cmap="gray")
            axes[1].set_title(f"After Dilation k={kernel_size}, iter={iterations}")
            axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    return dilated_objects, mask_before_all, mask_after_all, vis_img


def get_axis_endpoints_inside_contour(center_xy, axis_v, contour, image_shape, margin_px=5, step_px=1):
    """
    중심점에서 axis_v 방향 양끝으로 가면서
    contour 내부에 남아있는 마지막 지점을 찾음.
    """
    h, w = image_shape[:2]
    center = np.array(center_xy, dtype=np.float64)
    v = np.array(axis_v, dtype=np.float64)

    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return None, None

    v = v / norm
    max_len = int(np.hypot(w, h))

    endpoints = []

    for sign in [+1, -1]:
        last_t = 0

        for t in range(0, max_len, step_px):
            p = center + sign * v * t
            x, y = int(round(p[0])), int(round(p[1]))

            if x < 0 or x >= w or y < 0 or y >= h:
                break

            inside = cv2.pointPolygonTest(contour, (float(x), float(y)), False)

            if inside >= 0:
                last_t = t
            else:
                break

        # 경계 바로 끝은 노이즈가 많으니 살짝 안쪽으로 당김
        safe_t = max(0, last_t - margin_px)
        p_end = center + sign * v * safe_t
        endpoints.append((int(round(p_end[0])), int(round(p_end[1]))))

    return endpoints[0], endpoints[1]


def make_endpoint_region_mask(endpoint_xy, contour, image_shape, radius_px=8):
    """
    endpoint 주변 원형 영역 중 contour 내부인 부분만 mask로 생성.
    """
    h, w = image_shape[:2]
    ex, ey = endpoint_xy

    contour_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)

    disk_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(disk_mask, (ex, ey), radius_px, 255, thickness=-1)

    region_mask = cv2.bitwise_and(contour_mask, disk_mask)
    return region_mask


def get_lab_color_ratio_rgb4(color_rgb, region_mask):
    """
    region_mask 영역의 LAB 색을 보고
    red / green / yellow / blue 중 nearest prototype 비율 계산.
    """
    valid = region_mask > 0

    if np.count_nonzero(valid) == 0:
        return {
            "dominant": "unknown",
            "ratio": {"red": 0.0, "green": 0.0, "yellow": 0.0, "blue": 0.0},
            "count": 0
        }

    lab_img = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2LAB)
    pixels_lab = lab_img[valid].astype(np.float32)

    # RGB prototype -> LAB prototype
    proto_rgb = np.array(
        [[[255, 0, 0],
          [0, 255, 0],
          [255, 255, 0],
          [0, 0, 255]]],
        dtype=np.uint8
    )

    proto_lab = cv2.cvtColor(proto_rgb, cv2.COLOR_RGB2LAB)[0].astype(np.float32)

    color_names = ["red", "green", "yellow", "blue"]

    # L은 조명 영향이 크니까 약하게, a/b를 강하게 봄
    weights = np.array([0.25, 1.0, 1.0], dtype=np.float32)

    diff = pixels_lab[:, None, :] - proto_lab[None, :, :]
    dist = np.sqrt(np.sum((diff * weights) ** 2, axis=2))

    nearest = np.argmin(dist, axis=1)

    total = len(nearest)
    ratio = {}

    for i, name in enumerate(color_names):
        ratio[name] = float(np.count_nonzero(nearest == i) / total)

    dominant = max(ratio, key=ratio.get)

    return {
        "dominant": dominant,
        "ratio": ratio,
        "count": int(total)
    }


def analyze_axis_end_colors(
    color_rgb,
    contour,
    center_xy,
    axis_v,
    radius_px=8,
    margin_px=5
):
    """
    선택된 PCA 축의 양끝 endpoint 주변 색 비율 분석.
    """
    p_plus, p_minus = get_axis_endpoints_inside_contour(
        center_xy=center_xy,
        axis_v=axis_v,
        contour=contour,
        image_shape=color_rgb.shape[:2],
        margin_px=margin_px
    )

    if p_plus is None or p_minus is None:
        return None

    plus_mask = make_endpoint_region_mask(
        endpoint_xy=p_plus,
        contour=contour,
        image_shape=color_rgb.shape[:2],
        radius_px=radius_px
    )

    minus_mask = make_endpoint_region_mask(
        endpoint_xy=p_minus,
        contour=contour,
        image_shape=color_rgb.shape[:2],
        radius_px=radius_px
    )

    plus_color = get_lab_color_ratio_rgb4(color_rgb, plus_mask)
    minus_color = get_lab_color_ratio_rgb4(color_rgb, minus_mask)

    return {
        "plus_uv": p_plus,
        "minus_uv": p_minus,
        "plus_color": plus_color,
        "minus_color": minus_color,
        "plus_mask": plus_mask,
        "minus_mask": minus_mask
    }


def make_hsv_target_color_mask(color_rgb, target_color, base_mask=None):
    """
    RGB 이미지에서 red / green / yellow / blue 중 target_color 마스크 생성.
    base_mask가 있으면 그 내부에서만 계산.
    """
    hsv = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2HSV)

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    valid = (S > 40) & (V > 35)

    if target_color == "red":
        color_mask = ((H <= 10) | (H >= 170)) & valid

    elif target_color == "yellow":
        color_mask = (H >= 18) & (H <= 40) & valid

    elif target_color == "green":
        color_mask = (H >= 40) & (H <= 95) & valid

    elif target_color == "blue":
        color_mask = (H >= 90) & (H <= 135) & valid

    else:
        raise ValueError(f"Unknown target_color: {target_color}")

    color_mask = color_mask.astype(np.uint8) * 255

    if base_mask is not None:
        base_mask_255 = ((base_mask > 0).astype(np.uint8) * 255)
        color_mask = cv2.bitwise_and(color_mask, base_mask_255)

    return color_mask

def decide_bottom_side_by_color_projection(
    color_rgb,
    contour,
    center_xy,
    axis_v,
    bottom_color,
    min_pixels=30,
    side_gap_threshold=0.15,
    erode_px=3
):
    """
    contour 내부에서 bottom_color 픽셀을 찾고,
    그 픽셀들이 PCA 축의 plus/minus 어느 쪽에 많은지 판단.
    """
    h, w = color_rgb.shape[:2]

    contour_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)

    if erode_px > 0:
        kernel = np.ones((erode_px, erode_px), np.uint8)
        contour_mask = cv2.erode(contour_mask, kernel, iterations=1)

    target_mask = make_hsv_target_color_mask(
        color_rgb=color_rgb,
        target_color=bottom_color,
        base_mask=contour_mask
    )

    ys, xs = np.where(target_mask > 0)

    if len(xs) < min_pixels:
        return {
            "bottom_side": "unknown",
            "bottom_color": bottom_color,
            "target_pixel_count": int(len(xs)),
            "plus_ratio": 0.0,
            "minus_ratio": 0.0,
            "target_mask": target_mask,
            "target_centroid_uv": None
        }

    pts = np.stack([xs, ys], axis=1).astype(np.float64)

    center = np.array(center_xy, dtype=np.float64)
    v = np.array(axis_v, dtype=np.float64)

    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return {
            "bottom_side": "unknown",
            "bottom_color": bottom_color,
            "target_pixel_count": int(len(xs)),
            "plus_ratio": 0.0,
            "minus_ratio": 0.0,
            "target_mask": target_mask,
            "target_centroid_uv": None
        }

    v = v / norm

    # PCA 축 위로 projection
    proj = (pts - center) @ v

    plus_count = np.count_nonzero(proj > 0)
    minus_count = np.count_nonzero(proj < 0)
    total = len(proj)

    plus_ratio = plus_count / total
    minus_ratio = minus_count / total

    if plus_ratio - minus_ratio > side_gap_threshold:
        bottom_side = "plus"
    elif minus_ratio - plus_ratio > side_gap_threshold:
        bottom_side = "minus"
    else:
        bottom_side = "unknown"

    target_centroid = np.mean(pts, axis=0)
    target_centroid_uv = (
        int(round(target_centroid[0])),
        int(round(target_centroid[1]))
    )

    return {
        "bottom_side": bottom_side,
        "bottom_color": bottom_color,
        "target_pixel_count": int(total),
        "plus_ratio": float(plus_ratio),
        "minus_ratio": float(minus_ratio),
        "target_mask": target_mask,
        "target_centroid_uv": target_centroid_uv
    }



################################### 실행 함수

def get_target_grasp_pose(class_index, target_class_name):
    """
    class_index와 원하는 클래스 이름을 넣으면 광학축 최근접 객체의 (X, Y, Z, Yaw)를 반환합니다.
    """
    if target_class_name not in class_index:
        print(f"🚨 시야에 [{target_class_name}] 블록이 없습니다.")
        print(f"👉 현재 감지된 클래스 목록: {list(class_index.keys())}")
        return None, None, None, None

    if len(class_index[target_class_name]) == 0:
        print(f"🚨 [{target_class_name}] 클래스는 있지만 pose가 비어 있습니다.")
        return None, None, None, None

    # class_index[target_class_name]가 이미 광학축 가까운 순서로 정렬되어 있어야 함
    target_brick = class_index[target_class_name][0]

    pick_x = target_brick["x_mm"]
    pick_y = target_brick["y_mm"]
    pick_z = target_brick["z_mm"]
    pick_yaw = target_brick["yaw_deg"]

    print(f"🎯 타겟 [{target_class_name}] 포착 완료!")
    print(
        f"   ➔ 로봇 이동 좌표: "
        f"X={pick_x:.1f}, Y={pick_y:.1f}, Z={pick_z:.1f} / "
        f"회전: Yaw={pick_yaw:.2f}도"
    )

    return pick_x, pick_y, pick_z, pick_yaw

def fine_correct(final_obj_fine,
    color_rgb,
    depth,
    scale,
    intrinsics,
    V_visualize=True
    ):

    img_rgb = color_rgb.copy()
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

    # ============================================================
    # [NEW] Step 0: RANSAC 뎁스 기반 바닥 제거
    # 바닥면을 찾아 해당 영역을 검은색(0,0,0)으로 칠합니다.
    # ============================================================
    # 0-1. 뎁스를 3D 공간 좌표로 변환
    xyz_map, valid_mask = depth_to_xyz_map(depth, scale, intrinsics)
    valid_points = xyz_map[valid_mask]

    # 0-2. RANSAC 평면 피팅 (바닥 찾기)
    best_plane, _ = fit_plane_ransac_numpy(valid_points, distance_threshold=0.006)

    # 0-3. 바닥 평면으로부터의 Z축 거리(높이) 계산
    dist_map = compute_plane_distance_map(xyz_map, valid_mask, best_plane)

    # 0-4. 바닥에서 1cm(0.01m) 이상 튀어나온 영역만 마스킹 (1cm 미만은 바닥으로 간주)
    ransac_mask = (dist_map > 0.010).astype(np.uint8) * 255

    # 노이즈를 살짝 지워주기 위해 모폴로지 열기(Open) 적용
    kernel_open = np.ones((9, 9), np.uint8)
    ransac_mask = cv2.morphologyEx(ransac_mask, cv2.MORPH_OPEN, kernel_open)

    # 닫기(Close)를 통해 가까운 덩어리들을 1차로 뭉치기
    kernel_close = np.ones((9, 9), np.uint8)
    ransac_mask = cv2.morphologyEx(ransac_mask, cv2.MORPH_CLOSE, kernel_close)

    # --------------------------------------------------------
    # 1. Dilation (팽창) 적용: 테두리 복구 및 덩어리들 확실히 연결
    # --------------------------------------------------------
    kernel_dilate = np.ones((5, 5), np.uint8)
    ransac_mask = cv2.dilate(ransac_mask, kernel_dilate, iterations=1)

    # --------------------------------------------------------
    # 2. [NEW] Convex Hull (볼록 선체) 적용
    # 오목하게 파인 부분을 고무줄로 묶듯 팽팽하게 채워서 완벽한 한 덩어리로 만듭니다.
    # --------------------------------------------------------
    # 먼저 현재 마스크에서 윤곽선들을 찾습니다.
    contours_ransac, _ = cv2.findContours(ransac_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convex Hull을 그려넣을 빈 캔버스 생성
    hull_mask = np.zeros_like(ransac_mask)

    for cnt in contours_ransac:
        # 너무 작은 자잘한 노이즈는 무시 (필요시 수치 조절)
        if cv2.contourArea(cnt) < 200:
            continue
            
        # 윤곽선을 감싸는 최소한의 볼록 다각형(Convex Hull) 좌표 계산
        hull = cv2.convexHull(cnt)
        
        # 빈 캔버스에 볼록 다각형을 내부까지 꽉 채워서(thickness=-1) 흰색(255)으로 그림
        cv2.drawContours(hull_mask, [hull], -1, 255, thickness=-1)

    # --------------------------------------------------------
    # 3. [NEW] 최종 Padding (추가 팽창) 적용
    # Convex Hull로 묶인 객체의 바깥쪽에 여유 공간(패딩)을 줍니다.
    # --------------------------------------------------------
    kernel_pad = np.ones((7, 7), np.uint8) # 패딩 두께를 늘리려면 (7, 7) 등으로 조절
    ransac_mask = cv2.dilate(hull_mask, kernel_pad, iterations=1) # 다음 단계로 넘기기 위해 변수명 원상복구 
    # 0-5. RGB 이미지에 마스크 씌우기 (바닥 부분은 완전히 검정색으로)
    img_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=ransac_mask)


    # ============================================================
    # Step 1: 양방향 필터 (Bilateral Filter) 적용
    # ============================================================
    filtered_rgb = cv2.bilateralFilter(img_rgb, d=5, sigmaColor=50, sigmaSpace=50)

    # ============================================================

    # Step 2: LAB 색공간 변환 및 채널 분리
    # (참고: 앞서 배경을 칠한 검정색은 LAB에서 L=0, a=128, b=128이 됩니다!)
    # ============================================================

    lab = cv2.cvtColor(filtered_rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)

    # ============================================================
    # Step 3: 핵심 전처리 - 128(무채색 배경)과의 절대 거리 계산
    # ============================================================
    a_dist = cv2.absdiff(a, 128)
    b_dist = cv2.absdiff(b, 128)

    # ============================================================
    # Step 4: Min-Max 정규화 (상대평가 스케일링)
    # ============================================================

    a_norm = cv2.normalize(a_dist, None, 0, 255, cv2.NORM_MINMAX)
    b_norm = cv2.normalize(b_dist, None, 0, 255, cv2.NORM_MINMAX)

    # ============================================================
    # Step 5: a와 b 채널 병합
    # ============================================================
    ab_combined = cv2.max(a_norm, b_norm)

    # ============================================================
    # 시각화 1 (각 단계별 변화 및 RANSAC 마스크 확인)
    # ============================================================

    images = [
        ("0. RANSAC Floor Mask", ransac_mask, 'gray'),
        ("1. Masked RGB (Floor Removed)", img_rgb, None),
        ("2. Bilateral Filtered", filtered_rgb, None),
        ("3. Raw 'a' Channel", a, 'gray'),
        ("4. Raw 'b' Channel", b, 'gray'),
        ("5. |a - 128| Normalized", a_norm, 'gray'),
        ("6. |b - 128| Normalized", b_norm, 'gray'),
        ("7. Final Signal for K-Means", ab_combined, 'gray')
    ]
    if V_visualize:
        plt.figure(figsize=(20, 10))
        for i, (title, img, cmap) in enumerate(images):
            # 2행 4열 구조로 배치
            plt.subplot(2, 4, i + 1)
            plt.title(title)
            if cmap == 'gray':
                plt.imshow(img, cmap='gray', vmin=0, vmax=255)
            else:
                plt.imshow(img)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    # ============================================================
    # Step 8: 이진화 및 모폴로지 정리
    # ============================================================

    _, binary_mask = cv2.threshold(ab_combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    # ====================================================================


    # ============================================================
    # Step 9: 쿠키 틀 바인딩 및 규격화된 pose_table 생성
    # ============================================================
    result_img = color_rgb.copy()
    fine_pose_table = []  # <--- [NEW] 리턴할 테이블 배열

    for global_idx, obj in enumerate(final_obj_fine):
        raw_name = obj.get("class_name", "unknown")
        yolo_mask_bool = np.asarray(obj["mask"], dtype=bool)

        local_cookie_mask = np.zeros_like(binary_mask)
        local_cookie_mask[yolo_mask_bool] = binary_mask[yolo_mask_bool]

        contours_local, _ = cv2.findContours(local_cookie_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_local: continue

        best_cnt = max(contours_local, key=cv2.contourArea)
        if cv2.contourArea(best_cnt) < 200: continue

        rect = cv2.minAreaRect(best_cnt)
        M = cv2.moments(best_cnt)
        if M["m00"] == 0: continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

        width, height = rect[1][0], rect[1][1]
        aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 1.0

        if aspect_ratio >= 1.5:
            # ============================================================
            # [수정] PCA 대신 minAreaRect 박스의 단축 방향으로 yaw 계산
            # 기준:
            # - 영상 좌표계에서 위쪽 방향 = -Y
            # - 단축 벡터가 위쪽(-Y)을 향하도록 방향 선택
            # - atan2(x, -y)로 수직 위쪽 기준 yaw 계산
            # ============================================================

            box_pts = cv2.boxPoints(rect).astype(np.float64)  # shape: (4, 2)

            # ------------------------------------------------------------
            # 1. minAreaRect 박스의 인접 edge 2개만 보면 됨
            #    box_pts는 사각형 꼭짓점이 순서대로 들어있으므로
            #    edge0, edge1은 서로 수직인 두 변 방향
            # ------------------------------------------------------------
            edge0 = box_pts[1] - box_pts[0]
            edge1 = box_pts[2] - box_pts[1]

            len0 = np.linalg.norm(edge0)
            len1 = np.linalg.norm(edge1)

            # ------------------------------------------------------------
            # 2. 더 짧은 edge를 단축 방향 벡터로 선택
            # ------------------------------------------------------------
            if len0 <= len1:
                minor_v = edge0
            else:
                minor_v = edge1

            # ------------------------------------------------------------
            # 3. 방향 통일
            #    영상 좌표계는 y가 아래로 증가하므로,
            #    위쪽을 향하는 벡터는 y 성분이 음수여야 함.
            # ------------------------------------------------------------
            if minor_v[1] > 0:
                minor_v = -minor_v

            # y가 거의 0인 수평 단축일 경우 방향이 애매하므로
            # x 양수 방향을 +90도로 통일하고 싶으면 이 처리 추가
            if abs(minor_v[1]) < 1e-6 and minor_v[0] < 0:
                minor_v = -minor_v

            # ------------------------------------------------------------
            # 4. 정규화
            # ------------------------------------------------------------
            norm = np.linalg.norm(minor_v)
            if norm < 1e-6:
                continue

            chosen_v = minor_v / norm

            # ------------------------------------------------------------
            # 5. 영상 좌표계 -Y 방향 기준 yaw 계산
            #    chosen_v = [0, -1] 이면 yaw = 0도
            #    chosen_v = [1,  0] 이면 yaw = +90도
            #    chosen_v = [-1, 0] 이면 yaw = -90도
            # ------------------------------------------------------------
            yaw_deg = math.degrees(math.atan2(chosen_v[0], -chosen_v[1]))

            # ------------------------------------------------------------
            # 시각화
            # ------------------------------------------------------------
            if result_img is not None:
                p_center = np.array(rect[0], dtype=np.float64)

                # 객체 중심점
                cv2.circle(result_img, tuple(p_center.astype(int)), 5, (0, 0, 255), -1)

                # 기준선: 영상 좌표계 -Y 방향
                p_ref_end = p_center + np.array([0, -45], dtype=np.float64)
                cv2.line(
                    result_img,
                    tuple(p_center.astype(int)),
                    tuple(p_ref_end.astype(int)),
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA
                )

                # 선택된 단축 방향선
                p_minor_end = p_center + chosen_v * 45
                cv2.line(
                    result_img,
                    tuple(p_center.astype(int)),
                    tuple(p_minor_end.astype(int)),
                    (0, 0, 0),
                    7,
                    cv2.LINE_AA
                )

                # minAreaRect 박스도 같이 확인용으로 그림
                box_int = box_pts.astype(np.int32)
                cv2.drawContours(result_img, [box_int], -1, (255, 0, 255), 1)
        else:
            box_pts = cv2.boxPoints(rect)  # 사각형의 4개 꼭짓점 좌표 [shape: (4, 2)]

            # 1. 영상 좌표계 기준 Y값이 '가장 높은(즉, 화면상 제일 밑바닥에 있는)' 꼭짓점 찾기
            max_y_idx = int(np.argmax(box_pts[:, 1]))
            p_base = box_pts[max_y_idx]

            # 2. p_base와 연결된 양쪽 인접 꼭짓점 2개 구하기 (OpenCV는 꼭짓점이 순서대로 배열되어 있음)
            p_A = box_pts[(max_y_idx - 1) % 4]
            p_B = box_pts[(max_y_idx + 1) % 4]

            # 3. p_base에서 출발하여 p_A, p_B로 향하는 두 개의 선분 벡터 생성
            v_A = p_A - p_base
            v_B = p_B - p_base

            # 4. Y축 평행 기준선 벡터: p_base에서 '반대편 위쪽(-Y 방향)'으로 똑바로 뻗은 벡터 (0, -1)
            # math.atan2(x, -y) 공식을 쓰면 (0, -1) 벡터를 0도로 삼아 [우측 갸우뚱=+, 좌측 갸우뚱=-] 각도가 나옴
            angle_A = math.degrees(math.atan2(v_A[0], -v_A[1]))
            angle_B = math.degrees(math.atan2(v_B[0], -v_B[1]))

            # 5. Y축 평행선과 이루는 '절대 각도(abs)'가 더 짧은 쪽을 진짜 자세 각도로 채택!
            if abs(angle_A) < abs(angle_B):
                yaw_deg = angle_A
                chosen_p = p_A   # 시각화 강조용
            else:
                yaw_deg = angle_B
                chosen_p = p_B

            # (선택) 만약 사용하는 로봇 제어기가 [왼쪽 갸우뚱=(+)], [오른쪽 갸우뚱=(-)] 라면 아래 주석 해제
            # yaw_deg = -yaw_deg

            # ----------------------------------------------------------------
            # [NEW] 검은선을 '객체 정중앙'에서 뻗어나가도록 평행 이동
            # ----------------------------------------------------------------
            # 1. 사각형의 정중앙 좌표 (minAreaRect가 뱉는 도심값)
            p_center = np.array(rect[0])

            # 2. 알고리즘이 선택한 변의 '순수 방향 벡터' (길이와 기울기 정보)
            v_chosen = chosen_p - p_base

            # 3. 정중앙(p_center)에서 출발하여 v_chosen 방향으로 뻗어나간 끝점
            p_center_end = p_center + v_chosen


            # --- [시각화 드로잉 업데이트] ---
            # ① 바닥 기준점(p_base)은 '초록색 점' (유지)
            cv2.circle(result_img, tuple(p_base.astype(int)), 5, (0, 255, 0), -1)

            # ② [추가] 실제 로봇 그리퍼가 내려꽂힐 '객체 정중앙 파지점'에 빨간색 점 찍기
            cv2.circle(result_img, tuple(p_center.astype(int)), 5, (0, 0, 255), -1)

            # ③ 위로 뻗은 Y축 평행 기준선은 '하늘색 선' (유지)
            cv2.line(result_img, tuple(p_base.astype(int)), (int(p_base[0]), int(p_base[1] - 35)), (0, 200, 200), 1)

            # ④ [수정됨] 알고리즘이 선택한 각도를 '객체 정중앙에서 뻗어나가는 검은색 핀'으로 출력!
            cv2.line(result_img, tuple(p_center.astype(int)), tuple(p_center_end.astype(int)), (0, 0, 0), 7)

        yaw_deg = round(yaw_deg, 1)

        # 3D 좌표 및 광학 축 거리 계산
        x_m, y_m, top_z_m = xyz_map[cy, cx]
        h_mm = dist_map[cy, cx] * 1000.0
        center_z_mm = (top_z_m * 1000.0) + (h_mm / 2.0)
        axis_dist_m = float(np.sqrt(x_m**2 + y_m**2))

        real_yolo_id = obj.get("class_id", obj.get("cls", global_idx))

        # ★ 로봇 파이프라인 호환성을 위해 Coarse 규격과 100% 동일한 Key 부여
        item = {
            "global_idx": global_idx,
            "yolo_id": real_yolo_id,
            "class_name": raw_name,
            "axis_dist_m": axis_dist_m,
            "axis_dist_mm": axis_dist_m * 1000.0,
            "depth_m": float(center_z_mm / 1000.0),
            "x_m": float(x_m),
            "y_m": float(y_m),
            "z_m": float(center_z_mm / 1000.0),
            "x_mm": float(x_m * 1000.0),
            "y_mm": float(y_m * 1000.0),
            "z_mm": float(center_z_mm),
            "top_z_mm": float(top_z_m * 1000.0),
            "object_height_mm": float(h_mm),
            "roll_deg": 180.0,  # Top-down 고정 RPY 호환성 유지
            "pitch_deg": 0.0,
            "yaw_deg": float(yaw_deg),
            "aspect_ratio": float(aspect_ratio),
            "uv_center": (cx, cy),
            "contour": best_cnt,
            "object_ref": obj
        }

        obj["fine_pose"] = item
        fine_pose_table.append(item)

        cv2.drawContours(result_img, [best_cnt], -1, (0, 255, 0), 2)
        cv2.circle(result_img, (cx, cy), 4, (255, 0, 0), -1)
        cv2.drawContours(result_img, [np.intp(cv2.boxPoints(rect))], 0, (255, 165, 0), 2)
        cv2.putText(result_img, f"YOLO_ID:{real_yolo_id} ({raw_name})", (cx - 40, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result_img, f"Yaw: {yaw_deg}d", (cx - 40, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # --- 테이블 정렬 및 class_index 빌드 ---
    fine_pose_table = sorted(fine_pose_table, key=lambda x: x["axis_dist_m"])
    fine_class_index = {}

    for item in fine_pose_table:
        cls = item["class_name"]
        if cls not in fine_class_index: fine_class_index[cls] = []
        fine_class_index[cls].append(item)

    for cls, items in fine_class_index.items():
        items.sort(key=lambda x: x["axis_dist_m"])
        for local_id, item in enumerate(items): item["local_id"] = local_id

    if V_visualize:
        cv2.drawMarker(result_img, (int(intrinsics.ppx), int(intrinsics.ppy)), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
        plt.figure(figsize=(12, 8))
        plt.imshow(result_img)
        plt.title("Masterpiece Contours + YOLO ID Bind")
        plt.axis("off")
        plt.show()

    return fine_pose_table, fine_class_index

def search_bricks(mode, yolo_dir, color_rgb, depth, intrinsics, scale, V_visualize=True):

    if mode in ["coarse", "fine"]:
        pass
    else:
        raise ValueError(f"잘못된 mode 입력: {mode}. mode는 'coarse' 또는 'fine'만 가능합니다.")

    # 이미지 입력 확인

    if color_rgb is None or depth is None or intrinsics is None or scale is None:
        raise RuntimeError("RealSense 캡처 실패: color/depth/intrinsics/scale 중 None이 있습니다.")

    color_img_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)

    #======================================================================================
    # YOLO V8 세그멘테이션 기준 영역 잡기 + 겹침 전처리
    #======================================================================================

    MODEL_PATH = os.path.join(yolo_dir)

    if V_visualize:
        print("[DEBUG] MODEL_PATH:", MODEL_PATH)
        print("[DEBUG] MODEL_EXISTS:", os.path.exists(MODEL_PATH))

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)
    target_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    results, mask_binary, vis_yolo = detect_objects_yolo(
        model= model, 
        color_img_bgr=color_img_bgr, 
        target_classes=target_classes, 
        visualize=V_visualize
    )

    if V_visualize:

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        fig.suptitle("[Debug Brick 1. YOLO DETECTION]")

        axes[0].imshow(vis_yolo)
        axes[0].set_title("YOLO result")
        axes[0].axis("off")

        axes[1].imshow(mask_binary, cmap="gray")
        axes[1].set_title("Yolo Mask")
        axes[1].axis("off")

        p_title = 'original YOLO'
        plt.suptitle(p_title)
        plt.tight_layout()
        plt.show()


    # 바운딩 박스가 겹치는 부분을 억제 = 겹치는 마스크 깔끔하게 정리
    final_objects, clean_mask = filter_overlapping_masks(
        results=results, 
        overlap_threshold=0.70, 
        img_shape=(640, 480), 
        visualize=V_visualize
    )

    #======================================================================================
    # 높이 차이 확인 + 바운딩 박스 비율 기반 ID 수정 + Convex Hull로 패딩
    #======================================================================================

    # DBSCAN + RANSAC 바닥 검출
    mask_40mm_2d, refined_color, contours, plane_model = extract_3d_protruding_objects(
        depth_img=depth, 
        color_img_bgr=color_img_bgr, 
        intrinsics=intrinsics, 
        depth_scale=scale, 
        yolo_combined_mask=clean_mask,
        depth_trunc=5.0,
        height_threshold=0.040,
        visualize=V_visualize
    )

    # ID 판독 및 교정 4*2 or 2*2 변경
    final_objects, result_vis_img = correct_object_ids(
        detected_objects=final_objects, 
        mask_high_2d=mask_40mm_2d, 
        color_img_bgr=color_img_bgr, 
        ratio_threshold=1.5, 
        overlap_threshold=0.20, 
        visualize=V_visualize
    )

    # Convex Hull로 내부 채우기
    mask_before = np.zeros(color_img_bgr.shape[:2], dtype=np.uint8)
    mask_hull_after = np.zeros(color_img_bgr.shape[:2], dtype=np.uint8)

    for obj in final_objects:
        original_mask = (obj["mask"] > 0).astype(np.uint8)

        # Convex Hull로 객체 영역 재생성
        hull_mask = fill_object_mask_by_convex_hull(
            original_mask,
            min_area=20
        )

        # 객체별 mask를 hull 결과로 교체
        obj["mask"] = hull_mask.astype(bool)

        # 전체 before / after 통합 마스크
        mask_before = np.logical_or(mask_before, original_mask > 0).astype(np.uint8)
        mask_hull_after = np.logical_or(mask_hull_after, hull_mask > 0).astype(np.uint8)

    # ============================================================
    # Hull 적용 완료된 final_objects를 fine 처리용으로 복사
    # ============================================================
    final_obj_fine = copy.deepcopy(final_objects)

    if V_visualize:
        print(f"[INFO] final_obj_fine 복사 완료: {len(final_obj_fine)} objects")

    # ============================================================
    # Hull 적용 완료된 이미지 확인
    # ============================================================
    if V_visualize:
        color_img_rgb = cv2.cvtColor(color_img_bgr, cv2.COLOR_BGR2RGB)

        # 0/1 mask -> 0/255로 변환
        before_vis = ((mask_before > 0).astype(np.uint8) * 255)
        after_vis = ((mask_hull_after > 0).astype(np.uint8) * 255)

        # after 마스크를 원본 컬러에 씌운 결과
        refined_after = cv2.bitwise_and(color_img_bgr, color_img_bgr, mask=after_vis)
        # refined_after_rgb = cv2.cvtColor(refined_after, cv2.COLOR_BGR2RGB)

        # after contour 그리기
        overlay_rgb = color_img_rgb.copy()
        contours, _ = cv2.findContours(after_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_rgb, contours, -1, (0, 255, 0), 2)

        fig, axes = plt.subplots(1, 4, figsize=(18, 6))

        axes[0].imshow(color_img_rgb)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(before_vis, cmap="gray")
        axes[1].set_title("Mask Before Hull")
        axes[1].axis("off")

        axes[2].imshow(after_vis, cmap="gray")
        axes[2].set_title("Mask After Hull")
        axes[2].axis("off")

        axes[3].imshow(overlay_rgb)
        axes[3].set_title("Hull Overlay")
        axes[3].axis("off")

        p_title="Convex Hull Mask Result"
        plt.suptitle(p_title)
        plt.tight_layout()
        plt.show()

    # =================================================================
    # Convex Hull + 마스크 기준으로 바닥/PCD 재생성 + 3D OBB 생성
    # =================================================================

    pcd_data, plane_data, floor_pcd = build_floor_scene_data_from_depth(
        depth_img=depth,
        intrinsics=intrinsics,
        depth_scale=scale,
        object_mask_01=mask_hull_after,
        depth_trunc=5.0,
        voxel_size=0.003,
        plane_dist_thresh=0.015,
        floor_height_eps=0.005,
        visualize=V_visualize
    )

    # OBB 재생성
    objects_obb, vis_3d, overlay_3d, vis_2d_rgb, obb_results = generate_3d_obbs_from_hull_objects(
        objects=final_objects,
        refined_mask_01=mask_hull_after,
        pcd_data=pcd_data,
        plane_data=plane_data,
        intrinsics=intrinsics,
        color_img_rgb=color_rgb,
        floor_pcd=floor_pcd,
        min_height=0.024,
        max_height_limit=0.12,
        height_percentile=95,
        visualize_2d=V_visualize
    )

    # =================================================================
    # 3D OBB 기준 객체 좌표계 + Camera 기준 RPY 계산
    # =================================================================

    pose_results = []
    axes_geometries = []
    plane_normal = plane_data["normal"]

    for idx, obj in enumerate(objects_obb):
        obb_3d = obj.get("obb_3d", None)
        class_name = obj.get("class_name", "unknown")

        pose = estimate_pose_axes_from_obb3d(
            obb_3d=obb_3d,
            plane_normal=plane_normal,
            class_name=class_name,
            axis_size=0.04
        )

        if pose is None:
            print(f"[SKIP] idx {idx}: pose 계산 실패")
            continue

        obj["pose_cam"] = pose
        pose_results.append({
            "idx": idx,
            "class_name": class_name,
            "center_mm": pose["center_mm"],
            "roll_deg": pose["roll_deg"],
            "pitch_deg": pose["pitch_deg"],
            "yaw_deg": pose["yaw_deg"],
            "R_obj_cam": pose["R_obj_cam"],
        })

        axes_geometries.append(pose["axes_3d"])

        c = pose["center_mm"]

        # print(
        #     f"idx {idx:02d} | {class_name:20s} | "
        #     f"center(mm)=({c[0]:7.1f}, {c[1]:7.1f}, {c[2]:7.1f}) | "
        #     f"RPY(deg)=({pose['roll_deg']:7.2f}, "
        #     f"{pose['pitch_deg']:7.2f}, "
        #     f"{pose['yaw_deg']:7.2f})"
        # )

        # 기존 3D OBB geometry에 좌표축 추가
    vis_3d_with_axes = vis_3d + axes_geometries
    overlay_3d_with_axes = overlay_3d + axes_geometries

    if V_visualize:
        print("\n[INFO] 3D OBB + Object Coordinate Axes 표시")
        o3d.visualization.draw_geometries(
            vis_3d_with_axes,
            window_name="3D OBB + Object XYZ Axes"
        )

    color_o3d = o3d.geometry.Image(color_rgb)
    depth_o3d = o3d.geometry.Image(cv2.medianBlur(depth, 5))

    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        int(intrinsics.width),
        int(intrinsics.height),
        float(intrinsics.fx),
        float(intrinsics.fy),
        float(intrinsics.ppx),
        float(intrinsics.ppy)
    )

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0 / float(scale),
        depth_trunc=5.0,
        convert_rgb_to_intensity=False
    )

    rgb_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d_intr
    )

    rgb_pcd = rgb_pcd.voxel_down_sample(voxel_size=0.0015)

    final_overlay_elements = [rgb_pcd] + overlay_3d_with_axes

    if V_visualize:
        print("\n[INFO] RGB-D PointCloud + 3D OBB + Object Axes 표시")
        o3d.visualization.draw_geometries(
            final_overlay_elements,
            window_name="RGB-D PointCloud + Object XYZ Axes"
        )

    pose_table, class_index = build_class_sorted_pose_index(
        objects_obb=objects_obb,
        use_pose_cam=True,
        remove_c_prefix=True,
        remove_side2=False,
        verbose=False
    )

    if mode == 'fine':

        if V_visualize:
            print("이제 파인으로 진입합니다.")

        # final_obj_fine 기준으로 dilation 적용
        final_obj_fine_a, mask_dilate_before, mask_dilate_after, dilate_vis_img = dilate_final_objects(
            final_obj_fine=final_obj_fine,
            image_shape=color_img_bgr.shape[:2],
            kernel_size=7,
            iterations=2,
            kernel_type="ellipse",
            visualize=V_visualize,
            color_img_bgr=color_img_bgr
        )

        pose_table, class_index = fine_correct(
                    final_obj_fine=final_obj_fine_a,
                    color_rgb=color_rgb,
                    depth=depth,
                    scale=scale,
                    intrinsics=intrinsics,
                    V_visualize=V_visualize
                )

    return pose_table, class_index 

def search_assembly(
    color_rgb,
    depth,
    intrinsics,
    scale,
    yolo_model=None,
    yolo_dir=None,
    V_visualize=True,

    # YOLO 설정
    target_classes=None,       # 숫자 class id 리스트. 예: [0, 1, 2], None이면 전체
    target_class_names=None,   # 문자열 class name 리스트. 예: ["Magnet"], None이면 전체
    conf_thres=0.5,
    iou_thres=0.3,
    imgsz=640,
    device=0,

    # mask 후처리
    yolo_mask_thresh=0.5,
    morph_open_ksize=3,
    morph_close_ksize=5,
    min_contour_area=80,

    # depth 검사
    min_valid_depth_points=30
):
    """
    YOLOv8-seg 클래스 이름 기반 search_assembly.

    반환:
        pose_table:
            전체 객체 pose list

        class_index:
            {
                "YOLO_CLASS_NAME": [pose0, pose1, ...],
                ...
            }

    사용 예:
        yolo_dir = "yolo_models/Component_Model_ver1.0/Model_s_ver2.0/best.pt"
        yolo_model = YOLO(yolo_dir)

        pose_table, class_index = search_assembly(
            color_rgb=color_rgb,
            depth=depth,
            intrinsics=intrinsics,
            scale=scale,
            yolo_model=yolo_model,
            V_visualize=True
        )

        target_class = "Magnet"
        X, Y, Z, YAW = get_target_grasp_pose(class_index, target_class)
    """

    if color_rgb is None or depth is None or intrinsics is None or scale is None:
        raise RuntimeError("RealSense 캡처 실패: color/depth/intrinsics/scale 중 None이 있습니다.")

    if yolo_model is None and yolo_dir is None:
        raise ValueError("yolo_model 또는 yolo_dir 중 하나는 반드시 입력해야 합니다.")

    # ------------------------------------------------------------
    # 0. 이미지 정리
    # ------------------------------------------------------------
    color_img_rgb = color_rgb.copy()
    if color_img_rgb.dtype != np.uint8:
        color_img_rgb = np.clip(color_img_rgb, 0, 255).astype(np.uint8)

    depth_img = depth.copy()
    depth_scale = scale

    H, W = depth_img.shape[:2]

    # ------------------------------------------------------------
    # 1. YOLO 모델 준비
    # ------------------------------------------------------------
    if yolo_model is None:
        from ultralytics import YOLO
        yolo_model = YOLO(yolo_dir)

    # ultralytics 입력은 BGR도 가능하지만, 기존 OpenCV 흐름에 맞춰 BGR 사용
    color_img_bgr = cv2.cvtColor(color_img_rgb, cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------
    # 2. YOLOv8 segmentation 추론
    # ------------------------------------------------------------
    yolo_kwargs = dict(
        conf=conf_thres,
        iou=iou_thres,
        imgsz=imgsz,
        device=device,
        verbose=False
    )

    if target_classes is not None:
        yolo_kwargs["classes"] = target_classes

    results = yolo_model(color_img_bgr, **yolo_kwargs)

    if len(results) == 0 or results[0].masks is None:
        print("[WARN] YOLO segmentation mask가 없습니다.")
        return [], {}

    result0 = results[0]

    masks = result0.masks.data.cpu().numpy()
    boxes = result0.boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()

    # YOLO class id -> class name
    yolo_names = result0.names

    # ------------------------------------------------------------
    # 3. depth -> xyz_map 생성
    # ------------------------------------------------------------
    xyz_map, valid_mask = depth_to_xyz_map(
        depth_img=depth_img,
        depth_scale=depth_scale,
        intrinsics=intrinsics
    )

    # ------------------------------------------------------------
    # 4. YOLO instance mask별 contour/PCA/pose 생성
    # ------------------------------------------------------------
    pose_table = []
    combined_mask = np.zeros((H, W), dtype=np.uint8)
    all_contour_objects_for_vis = []

    for det_idx, mask in enumerate(masks):
        class_id = int(class_ids[det_idx])
        yolo_class_name = str(yolo_names[class_id])
        confidence = float(confidences[det_idx])

        # 문자열 class name으로도 필터링 가능
        if target_class_names is not None:
            if yolo_class_name not in target_class_names:
                continue

        # mask resize
        mask_resized = cv2.resize(
            mask,
            (W, H),
            interpolation=cv2.INTER_NEAREST
        )

        instance_mask = (mask_resized > yolo_mask_thresh).astype(np.uint8) * 255

        # morphology
        if morph_open_ksize is not None and morph_open_ksize > 1:
            k_open = np.ones((morph_open_ksize, morph_open_ksize), np.uint8)
            instance_mask = cv2.morphologyEx(instance_mask, cv2.MORPH_OPEN, k_open)

        if morph_close_ksize is not None and morph_close_ksize > 1:
            k_close = np.ones((morph_close_ksize, morph_close_ksize), np.uint8)
            instance_mask = cv2.morphologyEx(instance_mask, cv2.MORPH_CLOSE, k_close)

        combined_mask = cv2.bitwise_or(combined_mask, instance_mask)

        # 이 YOLO instance mask 안에서 contour PCA 수행
        contour_objects = extract_contour_pca_from_mask(
            object_mask=instance_mask,
            xyz_map=xyz_map,
            valid_mask=valid_mask,
            min_contour_area=min_contour_area
        )

        for contour_obj in contour_objects:
            contour_mask = contour_obj.get("contour_mask", None)

            if contour_mask is None:
                continue

            valid_count = np.count_nonzero((contour_mask > 0) & valid_mask)

            if valid_count < min_valid_depth_points:
                print(
                    f"[SKIP] {yolo_class_name} det_idx={det_idx}: "
                    f"valid depth points 부족 ({valid_count})"
                )
                continue

            if "center_xyz" not in contour_obj:
                print(
                    f"[SKIP] {yolo_class_name} det_idx={det_idx}: "
                    f"center_xyz 없음"
                )
                continue

            contour_obj["yolo_class_id"] = class_id
            contour_obj["yolo_class_name"] = yolo_class_name
            contour_obj["yolo_confidence"] = confidence
            contour_obj["yolo_det_idx"] = det_idx
            contour_obj["valid_depth_points"] = int(valid_count)

            all_contour_objects_for_vis.append(contour_obj)

            center_xyz_m = np.asarray(contour_obj["center_xyz"], dtype=np.float64)
            center_xyz_mm = center_xyz_m * 1000.0

            yaw_deg = (float(contour_obj.get("angle_deg", 0.0)) + 180.0) % 180.0 - 90.0

            pose = {
                # 핵심: 여기 class_name이 YOLO 클래스 이름이 됨
                "class_name": yolo_class_name,
                "class_id": class_id,
                "confidence": confidence,

                "local_id": -1,      # 나중에 class별로 다시 부여
                "global_idx": -1,    # 나중에 전체 순서로 다시 부여
                "yolo_det_idx": det_idx,

                "x_mm": float(center_xyz_mm[0]),
                "y_mm": float(center_xyz_mm[1]),
                "z_mm": float(center_xyz_mm[2]),

                "roll_deg": 0.0,
                "pitch_deg": 0.0,
                "yaw_deg": float(yaw_deg),

                "center_mm": center_xyz_mm,
                "center_xyz": center_xyz_m,
                "center_uv": contour_obj.get("center_uv", None),

                "major_axis_uv": contour_obj.get("major_axis_uv", None),
                "minor_axis_uv": contour_obj.get("minor_axis_uv", None),
                "angle_deg": contour_obj.get("angle_deg", None),

                "major_axis_xyz": contour_obj.get("major_axis_xyz", None),
                "middle_axis_xyz": contour_obj.get("middle_axis_xyz", None),
                "minor_axis_xyz": contour_obj.get("minor_axis_xyz", None),

                "major_length_mm": float(contour_obj.get("major_length_m", 0.0) * 1000.0)
                    if "major_length_m" in contour_obj else None,
                "middle_length_mm": float(contour_obj.get("middle_length_m", 0.0) * 1000.0)
                    if "middle_length_m" in contour_obj else None,
                "minor_length_mm": float(contour_obj.get("minor_length_m", 0.0) * 1000.0)
                    if "minor_length_m" in contour_obj else None,

                "area_px": float(contour_obj.get("area_px", 0.0)),
                "valid_depth_points": int(valid_count),

                "raw_contour_object": contour_obj
            }

            pose_table.append(pose)

    # ------------------------------------------------------------
    # 5. 가까운 순서로 전체 정렬
    # ------------------------------------------------------------
    pose_table = sorted(pose_table, key=lambda p: p["z_mm"])

    for global_idx, pose in enumerate(pose_table):
        pose["global_idx"] = global_idx

    # ------------------------------------------------------------
    # 6. YOLO class_name 기준 class_index 생성
    # ------------------------------------------------------------
    class_index = {}

    for pose in pose_table:
        cname = pose["class_name"]

        if cname not in class_index:
            class_index[cname] = []

        class_index[cname].append(pose)

    # class별 local_id 다시 부여
    for cname, poses in class_index.items():
        poses_sorted = sorted(poses, key=lambda p: p["z_mm"])
        class_index[cname] = poses_sorted

        for local_id, pose in enumerate(poses_sorted):
            pose["local_id"] = local_id

    # ------------------------------------------------------------
    # 7. 시각화
    # ------------------------------------------------------------
    if V_visualize:
        vis_yolo_bgr = result0.plot()
        vis_yolo_rgb = cv2.cvtColor(vis_yolo_bgr, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8, 6))
        plt.imshow(vis_yolo_rgb)
        plt.title("YOLOv8 Segmentation Result")
        plt.axis("off")
        plt.show()

        plt.figure(figsize=(7, 5))
        plt.imshow(combined_mask, cmap="gray")
        plt.title("YOLO Combined Mask")
        plt.axis("off")
        plt.show()

        if len(all_contour_objects_for_vis) > 0:
            vis_contour_pca = visualize_contour_pca_axes(
                color_img_rgb=color_img_rgb,
                object_mask=combined_mask,
                contour_objects=all_contour_objects_for_vis,
                draw_mask=True,
                draw_contour=True,
                draw_min_rect=True,
                axis_len_mode="pca_length",
                fixed_axis_len=80
            )

            plt.figure(figsize=(8, 6))
            plt.imshow(vis_contour_pca)
            plt.title("YOLO Class Mask Contour PCA")
            plt.axis("off")
            plt.show()

    # ------------------------------------------------------------
    # 8. 출력 확인
    # ------------------------------------------------------------
    print("\n[YOLO Class Pose Table]")
    for pose in pose_table:
        print(
            f"global {pose['global_idx']:02d} | "
            f"local {pose['local_id']:02d} | "
            f"{pose['class_name']:16s} | "
            f"conf={pose['confidence']:.2f} | "
            f"XYZ mm=({pose['x_mm']:7.1f}, {pose['y_mm']:7.1f}, {pose['z_mm']:7.1f}) | "
            f"YAW={pose['yaw_deg']:7.2f} | "
            f"area={pose['area_px']:.0f}"
        )

    print("\n[YOLO Class Index]")
    for cname, poses in class_index.items():
        print(f"{cname}: {len(poses)}개")

    # ------------------------------------------------------------
    # class_index 생성
    # YOLO class_name 기준으로 묶고,
    # 각 클래스 내부는 광학축 중심에 가까운 순서로 정렬
    # ------------------------------------------------------------

    # 전체 pose에 global_idx 부여
    for global_idx, pose in enumerate(pose_table):
        pose["global_idx"] = global_idx

        # 광학축과의 거리 [mm]
        # 카메라 좌표계에서 optical axis는 보통 Z축이므로,
        # X-Y 평면에서 원점에 가까운 정도를 사용
        pose["optical_axis_dist_mm"] = float(
            np.sqrt(pose["x_mm"] ** 2 + pose["y_mm"] ** 2)
        )

    class_index = {}

    for pose in pose_table:
        cname = pose["class_name"]   # 여기에는 YOLO class name이 들어가야 함

        if cname not in class_index:
            class_index[cname] = []

        class_index[cname].append(pose)

    # 클래스별로 광학축 가까운 순서 정렬
    for cname, poses in class_index.items():
        poses_sorted = sorted(
            poses,
            key=lambda p: p["optical_axis_dist_mm"]
        )

        class_index[cname] = poses_sorted

        # class 내부 local_id 재부여
        for local_id, pose in enumerate(poses_sorted):
            pose["local_id"] = local_id

    # pose_table도 보기 좋게 광학축 가까운 순서로 정렬하고 싶으면
    pose_table = sorted(
        pose_table,
        key=lambda p: p["optical_axis_dist_mm"]
    )

    print("\n[YOLO Class Pose Table]")
    for pose in pose_table:
        print(
            f"global {pose['global_idx']:02d} | "
            f"local {pose['local_id']:02d} | "
            f"{pose['class_name']:16s} | "
            f"conf={pose['confidence']:.2f} | "
            f"XYZ mm=({pose['x_mm']:7.1f}, {pose['y_mm']:7.1f}, {pose['z_mm']:7.1f}) | "
            f"YAW={pose['yaw_deg']:7.2f} | "
            f"axis_dist={pose['optical_axis_dist_mm']:.1f}"
        )

    print("\n[YOLO Class Index]")
    for cname, poses in class_index.items():
        print(f"{cname}: {len(poses)}개")

    return pose_table, class_index

def search_assembly_fine(color_rgb, depth, intrinsics, scale, V_visualize=True):
    img_rgb = color_rgb.copy()
    # 1. 원본 이미지 복사 및 타입 안전성 확보
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

    # ============================================================
    # [NEW] Step 0: RANSAC 뎁스 기반 바닥 제거
    # 바닥면을 찾아 해당 영역을 검은색(0,0,0)으로 칠합니다.
    # ============================================================
    # 0-1. 뎁스를 3D 공간 좌표로 변환
    xyz_map, valid_mask = depth_to_xyz_map(depth, scale, intrinsics)
    valid_points = xyz_map[valid_mask]

    # 0-2. RANSAC 평면 피팅 (바닥 찾기)
    best_plane, _ = fit_plane_ransac_numpy(valid_points, distance_threshold=0.006)

    # 0-3. 바닥 평면으로부터의 Z축 거리(높이) 계산
    dist_map = compute_plane_distance_map(xyz_map, valid_mask, best_plane)

    # 0-4. 바닥에서 1cm(0.01m) 이상 튀어나온 영역만 마스킹 (1cm 미만은 바닥으로 간주)
    ransac_mask = (dist_map > 0.010).astype(np.uint8) * 255
    
    # 노이즈를 살짝 지워주기 위해 모폴로지 열기(Open) 적용
    kernel_open = np.ones((9, 9), np.uint8)
    ransac_mask = cv2.morphologyEx(ransac_mask, cv2.MORPH_OPEN, kernel_open)
    
    # 닫기(Close)를 통해 가까운 덩어리들을 1차로 뭉치기
    kernel_close = np.ones((9, 9), np.uint8)
    ransac_mask = cv2.morphologyEx(ransac_mask, cv2.MORPH_CLOSE, kernel_close)

    # --------------------------------------------------------
    # 1. Dilation (팽창) 적용: 테두리 복구 및 덩어리들 확실히 연결
    # --------------------------------------------------------
    kernel_dilate = np.ones((5, 5), np.uint8)
    ransac_mask = cv2.dilate(ransac_mask, kernel_dilate, iterations=1)

    # --------------------------------------------------------
    # 2. [NEW] Convex Hull (볼록 선체) 적용
    # 오목하게 파인 부분을 고무줄로 묶듯 팽팽하게 채워서 완벽한 한 덩어리로 만듭니다.
    # --------------------------------------------------------
    # 먼저 현재 마스크에서 윤곽선들을 찾습니다.
    contours_ransac, _ = cv2.findContours(ransac_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convex Hull을 그려넣을 빈 캔버스 생성
    hull_mask = np.zeros_like(ransac_mask)
    
    for cnt in contours_ransac:
        # 너무 작은 자잘한 노이즈는 무시 (필요시 수치 조절)
        if cv2.contourArea(cnt) < 200:
            continue
            
        # 윤곽선을 감싸는 최소한의 볼록 다각형(Convex Hull) 좌표 계산
        hull = cv2.convexHull(cnt)
        
        # 빈 캔버스에 볼록 다각형을 내부까지 꽉 채워서(thickness=-1) 흰색(255)으로 그림
        cv2.drawContours(hull_mask, [hull], -1, 255, thickness=-1)

    # --------------------------------------------------------
    # 3. [NEW] 최종 Padding (추가 팽창) 적용
    # Convex Hull로 묶인 객체의 바깥쪽에 여유 공간(패딩)을 줍니다.
    # --------------------------------------------------------
    kernel_pad = np.ones((7, 7), np.uint8) # 패딩 두께를 늘리려면 (7, 7) 등으로 조절
    ransac_mask = cv2.dilate(hull_mask, kernel_pad, iterations=1) # 다음 단계로 넘기기 위해 변수명 원상복구 
    # 0-5. RGB 이미지에 마스크 씌우기 (바닥 부분은 완전히 검정색으로)
    img_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=ransac_mask)

    # ============================================================
    # Step 1: 양방향 필터 (Bilateral Filter) 적용
    # ============================================================
    filtered_rgb = cv2.bilateralFilter(img_rgb, d=5, sigmaColor=50, sigmaSpace=50)

    # ============================================================

    # Step 2: LAB 색공간 변환 및 채널 분리
    # (참고: 앞서 배경을 칠한 검정색은 LAB에서 L=0, a=128, b=128이 됩니다!)
    # ============================================================

    lab = cv2.cvtColor(filtered_rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)

    # ============================================================
    # Step 3: 핵심 전처리 - 128(무채색 배경)과의 절대 거리 계산
    # ============================================================
    a_dist = cv2.absdiff(a, 128)
    b_dist = cv2.absdiff(b, 128)

    # ============================================================
    # Step 4: Min-Max 정규화 (상대평가 스케일링)
    # ============================================================

    a_norm = cv2.normalize(a_dist, None, 0, 255, cv2.NORM_MINMAX)
    b_norm = cv2.normalize(b_dist, None, 0, 255, cv2.NORM_MINMAX)

    # ============================================================
    # Step 5: a와 b 채널 병합
    # ============================================================
    ab_combined = cv2.max(a_norm, b_norm)

    # ============================================================
    # 시각화 1 (각 단계별 변화 및 RANSAC 마스크 확인)
    # ============================================================
    if V_visualize:
        images = [
            ("0. RANSAC Floor Mask", ransac_mask, 'gray'),
            ("1. Masked RGB (Floor Removed)", img_rgb, None),
            ("2. Bilateral Filtered", filtered_rgb, None),
            ("3. Raw 'a' Channel", a, 'gray'),
            ("4. Raw 'b' Channel", b, 'gray'),
            ("5. |a - 128| Normalized", a_norm, 'gray'),
            ("6. |b - 128| Normalized", b_norm, 'gray'),
            ("7. Final Signal for K-Means", ab_combined, 'gray')
        ]

        plt.figure(figsize=(20, 10))
        for i, (title, img, cmap) in enumerate(images):
            # 2행 4열 구조로 배치
            plt.subplot(2, 4, i + 1)
            plt.title(title)
            if cmap == 'gray':
                plt.imshow(img, cmap='gray', vmin=0, vmax=255)
            else:
                plt.imshow(img)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    # ============================================================
    # Step 8: 이진화 및 모폴로지 정리
    # ============================================================

    _, binary_mask = cv2.threshold(ab_combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # ============================================================
    # Step 9: 윤곽선(Contours) 검출
    # ============================================================

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_img = color_rgb.copy() # 원본 이미지 위에 그리기 위해 다시 복사

    # ============================================================
    # Step 10: 주점(Principal Point) 기준 가장 가까운 타겟 객체 하나만 추출
    # ============================================================

    # 1. 주점 좌표 미리 추출
    ppx = int(intrinsics.ppx)
    ppy = int(intrinsics.ppy)

    # 가장 가까운 객체를 저장할 변수 초기화
    closest_dist = float('inf')  # 무한대로 초기화
    best_target = None           # 시각화 및 반환할 최종 데이터 딕셔너리

    # 2. 모든 윤곽선을 돌며 주점과의 거리가 가장 짧은 객체 찾기
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        # 무게중심 계산
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # 중심점(cx, cy)과 주점(ppx, ppy) 사이의 픽셀 거리 계산
        dist_to_pp = np.sqrt((cx - ppx)**2 + (cy - ppy)**2)

        # 기존 최소 거리보다 가깝다면 정보 갱신
        if dist_to_pp < closest_dist:
            closest_dist = dist_to_pp
            rect = cv2.minAreaRect(cnt) # OBB 계산

            # 나중에 그리기 위해 필요한 정보 모두 저장
            best_target = {
                "contour": cnt,
                "cx": cx,
                "cy": cy,
                "rect": rect,
                "dist": dist_to_pp
            }


    # 3. 가장 가까운 타겟 하나만 결과 이미지에 시각화 및 3D 좌표 계산
    target_pose_info = None  # 반환할 최종 데이터

    if best_target is not None:
        cnt = best_target["contour"]
        cx, cy = best_target["cx"], best_target["cy"]
        rect = best_target["rect"]

        # --------------------------------------------------------
        # 가로, 세로 길이 추출 및 비율 계산
        # --------------------------------------------------------
        width = rect[1][0]
        height = rect[1][1]
        if width > 0 and height > 0:
            aspect_ratio = max(width, height) / min(width, height)
        else:
            aspect_ratio = 1.0

        # --------------------------------------------------------
        # 비율에 따른 하이브리드 Yaw 각도 계산 (12시 방향 기준)
        # --------------------------------------------------------
        if aspect_ratio >= 1.5:
            # 1. 길쭉한 객체 (PCA 방식 적용: -90도 ~ +90도)
            # 윤곽선 좌표들을 PCA 연산에 맞게 변환
            pts = cnt.reshape(-1, 2).astype(np.float64)
            mean, eigenvectors = cv2.PCACompute(pts, mean=None)
            
            # 가장 분산이 큰 주성분 벡터 (x, y)
            vx, vy = eigenvectors[0][0], eigenvectors[0][1]

            # 벡터가 무조건 위쪽(-y 방향)을 향하도록 방향 조정
            if vy > 0:
                vx, vy = -vx, -vy

            # 12시 방향(0, -1)을 기준으로 각도 계산
            # math.atan2(x축, y축)을 사용하여 12시 기준 좌/우 각도 도출
            yaw_deg = math.degrees(math.atan2(vx, -vy))
            # yaw_deg = yaw_deg

        else:
            # 2. 정사각형에 가까운 객체 (OBB 방식 적용: 0도 ~ 90도)
            box_pts = cv2.boxPoints(rect)
            v1 = box_pts[1] - box_pts[0]
            v2 = box_pts[2] - box_pts[1]

            # 두 모서리 중 더 긴 쪽을 '세로선(수직 기준선)'으로 간주
            if math.hypot(v1[0], v1[1]) > math.hypot(v2[0], v2[1]):
                dx, dy = v1[0], v1[1]
            else:
                dx, dy = v2[0], v2[1]

            # 12시 방향(0, -1)과 세로선 사이의 사이각을 0~90도로 절대값 출력
            yaw_deg = math.degrees(math.atan2(abs(dx), abs(dy)))
            # yaw_deg =+ 90

        # 소수점 첫째 자리까지만 깔끔하게 정리
        yaw_deg = round(yaw_deg, 1)

        # --------------------------------------------------------
        # 3D 좌표(X, Y, Z) 및 높이 중심 계산 (m -> mm)
        # --------------------------------------------------------
        x_m, y_m, top_z_m = xyz_map[cy, cx]
        x_mm = x_m * 1000.0
        y_mm = y_m * 1000.0
        top_z_mm = top_z_m * 1000.0

        h_mm = dist_map[cy, cx] * 1000.0
        center_z_mm = top_z_mm + (h_mm / 2.0)

        # --------------------------------------------------------

        # 외곽선 및 무게중심 그리기 (초록/파랑)
        cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 2)
        cv2.circle(result_img, (cx, cy), 5, (255, 0, 0), -1)

        # OBB 및 OBB 중심점 그리기 (주황)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(result_img, [box], 0, (255, 165, 0), 2)

        obb_cx, obb_cy = int(rect[0][0]), int(rect[0][1])
        cv2.circle(result_img, (obb_cx, obb_cy), 5, (255, 165, 0), -1)

        # 텍스트 출력 (비율 정보 추가해서 3줄로 출력)
        text1 = f"TARGET: Yaw {yaw_deg}deg"
        text2 = f"XYZ: {x_mm:.1f}, {y_mm:.1f}, {center_z_mm:.1f}"
        text3 = f"Ratio: {aspect_ratio:.2f}"
        cv2.putText(result_img, text1, (cx - 40, cy - 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result_img, text2, (cx - 40, cy - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result_img, text3, (cx - 40, cy + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

        # 로봇 제어 및 다음 단계를 위한 최종 반환 딕셔너리
        target_pose_info = {
            "x_mm": float(x_mm),
            "y_mm": float(y_mm),
            "z_mm": float(center_z_mm),
            "yaw_deg": float(yaw_deg),
            "object_height_mm": float(h_mm),
            "top_z_mm": float(top_z_mm),
            "width_px": float(width),         # 가로(픽셀) 추가
            "height_px": float(height),       # 세로(픽셀) 추가
            "aspect_ratio": float(aspect_ratio), # 비율 추가
            "uv_center": (cx, cy)
        }

    # 주점(렌즈 중심)에 노란색 십자가 그리기
    cv2.drawMarker(result_img, (ppx, ppy), color=(255, 255, 0), 
                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    # ============================================================
    # 시각화 2 (최종 마스크 및 결과)
    # ============================================================
    if V_visualize:
        images_final = [
            ("8. Binary Mask (Otsu + Morphology)", binary_mask, 'gray'),
            ("9. Extracted Contours & Grasping Poses", result_img, None)
        ]

        plt.figure(figsize=(16, 8))
        for i, (title, img, cmap) in enumerate(images_final):
            plt.subplot(1, 2, i + 1)
            plt.title(title)
            if cmap == 'gray':
                plt.imshow(img, cmap='gray', vmin=0, vmax=255)
            else:
                plt.imshow(img)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    return result_img, target_pose_info