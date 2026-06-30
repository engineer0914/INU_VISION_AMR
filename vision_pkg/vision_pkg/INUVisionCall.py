import os
from vision_pkg import INUVisionLib as ivl

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))

class VisionManager:
    def __init__(self):
        self.color_rgb = None
        self.depth = None
        self.intrinsics = None
        self.scale = None
        
        self.pose_table = None
        self.class_index = None

        self.target = None

        self.yolo_dir_component = os.path.join(_PKG_DIR, 'yolo_models', 'Component_Model_ver1.0', 'Model_s_ver2.0', 'best.pt')
        self.yolo_dir_brick = os.path.join(_PKG_DIR, 'yolo_models', 'Block_m_ver1.0', 'Block_s_ver1.0', 'best.pt')
        self.id_to_class = {
            1: "2x2_red", 
            2: "2x2_green", 
            3: "2x2_blue", 
            4: "2x2_yellow",
            5: "4x2_red", 
            6: "4x2_green", 
            7: "4x2_blue", 
            8: "4x2_yellow",

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
            48132: "icecream"
        }

    @staticmethod
    def _normalize_class_name(name):
        return str(name).lower().replace(" ", "")

    @staticmethod
    def _same_color_size_fallback(name):
        if name.startswith("4x2_"):
            return name.replace("4x2_", "2x2_", 1)
        if name.startswith("2x2_"):
            return name.replace("2x2_", "4x2_", 1)
        return None

    def _find_class_index_key(self, target_class_name):
        if self.class_index is None:
            return None, None

        normalized_target = self._normalize_class_name(target_class_name)

        for key in self.class_index.keys():
            if self._normalize_class_name(key) == normalized_target:
                return key, None

        fallback_class_name = self._same_color_size_fallback(target_class_name)
        if fallback_class_name is None:
            return None, None

        normalized_fallback = self._normalize_class_name(fallback_class_name)
        for key in self.class_index.keys():
            if self._normalize_class_name(key) == normalized_fallback:
                return key, fallback_class_name

        return None, None


    def capture_camera(self, mode="mid_50", V_visualize=False):

        devices = ivl.get_realsense_ids()

        if len(devices) == 0:
            raise RuntimeError("연결된 RealSense 카메라가 없습니다.")
        target_serial = list(devices.keys())[0]

        # 캡처한 데이터를 클래스 내부 보관함(self)에 저장
        print("[INFO] 카메라 데이터 캡처 중...")
        self.color_rgb, self.depth, self.intrinsics, self.scale = ivl.capture_realsense_data(
            serial_number=target_serial, 
            mode=mode, 
            warmup_frames=10,
            visualize=V_visualize
        )
        return self.color_rgb, self.depth, self.intrinsics, self.scale

    def run_search(self, mode, V_visualize=False):

        if self.color_rgb is None:
            raise RuntimeError("카메라 데이터가 없습니다. 먼저 capture_camera()를 실행하세요.")

        self.pose_table, self.class_index = ivl.search_bricks(mode, 
                                                            self.yolo_dir_brick, 
                                                            self.color_rgb, 
                                                            self.depth, 
                                                            self.intrinsics, 
                                                            self.scale, 
                                                            V_visualize=V_visualize
                                                            )

        return self.pose_table, self.class_index

    def run_search_assembly(self,V_visualize=False):

        print("[INFO] 조립체 객체 탐색(Search Assembly) 실행 중...")

        if self.color_rgb is None:
            raise RuntimeError("카메라 데이터가 없습니다. 먼저 capture_camera()를 실행하세요.")

        self.pose_table, self.class_index = ivl.search_assembly(self.color_rgb,
                                                                self.depth,
                                                                self.intrinsics,
                                                                self.scale,
                                                                yolo_model=None,
                                                                yolo_dir=self.yolo_dir_component,
                                                                V_visualize=V_visualize,

                                                                # YOLO 설정
                                                                target_classes=None,
                                                                target_class_names=None,
                                                                conf_thres=0.7,
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
                                                            )

        return self.pose_table, self.class_index

    def get_pose_by_id(self, target_id, local_id=0):
        """
        target_id를 클래스 이름으로 변환한 뒤,
        이미 실행된 self.class_index에서 해당 객체 pose를 찾아
        X, Y, Z, YAW를 반환합니다.

        반환:
            X, Y, Z: mm
            YAW: deg

        실패 시:
            None, None, None, None
        """

        # ------------------------------------------------------------
        # 1. ID -> class name 변환
        # ------------------------------------------------------------
        target_class_name = self.id_to_class.get(target_id)

        if target_class_name is None:
            print(f"[ERROR] 등록되지 않은 ID 번호입니다: {target_id}")
            print(f"[INFO] 등록된 ID 목록: {list(self.id_to_class.keys())}")
            return None, None, None, None

        print(f"\n[INFO] 타겟 ID [{target_id}] ➔ 클래스명 ['{target_class_name}'] 변환 완료")

        # ------------------------------------------------------------
        # 2. class_index 존재 확인
        # ------------------------------------------------------------
        if self.class_index is None:
            print("[ERROR] class_index가 없습니다.")
            print("먼저 run_search() 또는 run_search_assembly()를 실행하세요.")
            return None, None, None, None

        # ------------------------------------------------------------
        # 3. class_index에서 target_class_name 찾기
        #    YOLO 모델 클래스명과 대소문자가 다를 수 있으므로
        #    소문자로 변환하여 매칭
        # ------------------------------------------------------------
        matched_key, fallback_class_name = self._find_class_index_key(target_class_name)

        if matched_key is None:
            print(f"🚨 시야에 [{target_class_name}] 블록이 없습니다.")
            print(f"👉 현재 감지된 클래스 목록: {list(self.class_index.keys())}")
            return None, None, None, None

        if fallback_class_name is not None:
            print(
                f"[WARNING] 요청 클래스 [{target_class_name}]가 없어 "
                f"같은 색상 후보 [{matched_key}]로 대체합니다."
            )

        X, Y, Z, YAW = ivl.get_target_grasp_pose(
            self.class_index,
            matched_key
        )

        return X, Y, Z, YAW

    def run_pipeline_by_id(
        self,
        target_id,
        local_id=0,
        camera_mode="mid_50",
        brick_search_mode="fine",
        V_visualize_capture=False,
        V_visualize_search=False
    ):
        """
        target_id를 받아서:
        1. ID -> class name 변환
        2. 카메라 캡처
        3. ID 그룹에 따라 run_search / run_search_assembly 분기
        4. get_pose_by_id로 X, Y, Z, YAW 반환

        반환:
            result dict

            성공:
            {
                "success": True,
                "target_id": int,
                "class_name": str,
                "x_mm": float,
                "y_mm": float,
                "z_mm": float,
                "yaw_deg": float
            }

            실패:
            {
                "success": False,
                "target_id": int,
                "class_name": str or None,
                "reason": str
            }
        """

        # ------------------------------------------------------------
        # 0. ID 그룹 정의
        # ------------------------------------------------------------
        BRICK_IDS = {
            1, 2, 3, 4,
            5, 6, 7, 8
        }

        COMPONENT_IDS = {
            13,     # Magnet
            34,     # Battery
            81,     # Estop
            241,    # Trafficlight
            442,    # carrot
            462,    # small tree
            711,    # hammer
            4482,   # bigcarrot
            8518,   # burger
            46262,  # bigtree
            48132   # icecream
        }

        # ------------------------------------------------------------
        # 1. target_id 정리
        # ------------------------------------------------------------
        try:
            target_id = int(target_id)
        except Exception:
            print(f"[ERROR] target_id를 int로 변환할 수 없습니다: {target_id}")
            return {
                "success": False,
                "target_id": None,
                "class_name": None,
                "reason": "target_id int 변환 실패"
            }

        # ------------------------------------------------------------
        # 2. ID -> class name 변환
        # ------------------------------------------------------------
        target_class_name = self.id_to_class.get(target_id)

        if target_class_name is None:
            print(f"[ERROR] 등록되지 않은 ID 번호입니다: {target_id}")
            print(f"[INFO] 등록된 ID 목록: {list(self.id_to_class.keys())}")
            return {
                "success": False,
                "target_id": target_id,
                "class_name": None,
                "reason": "등록되지 않은 ID"
            }

        print(f"\n[INFO] target_id={target_id} -> class='{target_class_name}'")

        # ------------------------------------------------------------
        # 3. 카메라 캡처
        # ------------------------------------------------------------
        self.capture_camera(
            mode=camera_mode,
            V_visualize=V_visualize_capture
        )

        # ------------------------------------------------------------
        # 4. ID 그룹에 따라 Search 분기
        # ------------------------------------------------------------
        if target_id in BRICK_IDS:
            print(f"[VISION] 일반 브릭 탐색 실행: ID={target_id}, class={target_class_name}")

            self.run_search(
                mode=brick_search_mode,
                V_visualize=V_visualize_search
            )

        elif target_id in COMPONENT_IDS:
            print(f"[VISION] 컴포넌트 YOLO-Seg 탐색 실행: ID={target_id}, class={target_class_name}")

            self.run_search_assembly(
                V_visualize=V_visualize_search
            )

        else:
            print(f"[ERROR] 탐색 분기 미지정 ID입니다: {target_id}")

            return {
                "success": False,
                "target_id": target_id,
                "class_name": target_class_name,
                "reason": "탐색 분기 미지정 ID"
            }

        # ------------------------------------------------------------
        # 5. Search 결과에서 pose 추출
        # ------------------------------------------------------------
        X, Y, Z, YAW = self.get_pose_by_id(
            target_id=target_id,
            local_id=local_id
        )

        if X is None or Y is None or Z is None or YAW is None:
            print(f"[WARNING] 시야에서 타겟을 찾지 못했습니다: ID={target_id}, class={target_class_name}")

            return {
                "success": False,
                "target_id": target_id,
                "class_name": target_class_name,
                "reason": "pose 추출 실패"
            }

        # ------------------------------------------------------------
        # 6. 성공 결과 반환
        # ------------------------------------------------------------
        result = {
            "success": True,
            "target_id": target_id,
            "class_name": target_class_name,
            "x_mm": float(X),
            "y_mm": float(Y),
            "z_mm": float(Z),
            "yaw_deg": float(YAW)
        }

        print("[VISION] run_pipeline_by_id 결과")
        print(
            f"  ID={target_id}, class={target_class_name}, "
            f"X={X:.1f}mm, Y={Y:.1f}mm, Z={Z:.1f}mm, Yaw={YAW:.2f}deg"
        )

        return result



# # ==========================================
# # 4. 단독 실행용 테스트 코드
# # ==========================================
# if __name__ == "__main__":
#     print("\n[INFO] ivc.py 라이브러리 단독 테스트 모드 실행\n")
    
#     vision = VisionManager()
    
#     try:
#         vision.capture_camera(V_visualize=False)

#         vision.run_search(mode='fine', V_visualize=True)

#         # vision.run_search_assembly(V_visualize=True)
        
#         test_pose = vision.get_pose_by_id(target_id=7)
        
#         if test_pose:
#             print("클래스를 이용한 포즈 추출 성공!")
            
#     except Exception as e:
#         print(f"[ERROR] 테스트 중 오류 발생: {e}")


# ==========================================
# 4. 단독 실행용 테스트 코드
# ==========================================
if __name__ == "__main__":
    print("\n[INFO] ivc.py 라이브러리 단독 테스트 모드 실행\n")
    
    vision = VisionManager()
    
    try:
        # 테스트할 ID
        # 1~8: 브릭
        # 13, 34, 81 ...: 컴포넌트
        # TEST_TARGET_ID = 7
        TEST_TARGET_ID = 13

        result = vision.run_pipeline_by_id(
            target_id=TEST_TARGET_ID,
            local_id=0,
            camera_mode="mid_50",
            brick_search_mode="fine",
            V_visualize_capture=False,
            V_visualize_search=True
        )
        
        if result["success"]:
            print("클래스를 이용한 포즈 추출 성공!")
            print(f"target_id : {result['target_id']}")
            print(f"class     : {result['class_name']}")
            print(f"X mm      : {result['x_mm']:.1f}")
            print(f"Y mm      : {result['y_mm']:.1f}")
            print(f"Z mm      : {result['z_mm']:.1f}")
            print(f"Yaw deg   : {result['yaw_deg']:.2f}")
        else:
            print("클래스를 이용한 포즈 추출 실패")
            print(f"reason: {result.get('reason')}")
            
    except Exception as e:
        print(f"[ERROR] 테스트 중 오류 발생: {e}")
