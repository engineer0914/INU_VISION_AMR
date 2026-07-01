# **AMR 비전 패키지**

0701 수정:
시각화 안정화
서비스 호출시 시각화
브릭, 컴포넌트 별 함수 실행

YOLO + 영상 기하 판별 구조

X,Y,Z,YAW 반환

각도 반환값 기준

12시 방향 기준 0도
8시 방향 -120도 
4시 방향 120도

브릭 -90~90 도
조립체 -180~180 도


## **서비스 호출**

### 기본 호출 구조
```
ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '1'}"
```

### 브릭들 1~8 순차 호출
```
sh aorder.sh
```

### 컴포넌트들 13 34 ... 8518 순차 호출
```
sh border.sh
```

**빌드 과정**
```
export ROS2_WS=$PWD
echo $ROS2_WS
```

```
colcon build --packages-select vision_pkg
```

```
cbsel vision_pkg
```

```
ros2 run vision_pkg vision_node
```


### vision_6d_manager.py 파일내 빈공간 찾기 관련 변수 설정구조

**DEFAULT_EMPTY_SPACE_GRID_DIVISIONS = 5**
- 활성화 영역 정방 수 ex) 5 = 5*5

**DEFAULT_EMPTY_SPACE_VISUALIZE_SEC = 3.0**
- 시각화 시간 초단위

**DEFAULT_EMPTY_SPACE_CELL_OCCUPIED_RATIO = 0.000001**
- 활성화 영역내 욜로 마스킹이 비활성화 되는 기준 비율
- ex) 영역이 100픽셀이라할때, 내부에 욜로 결과가 10픽셀이라면 = 0.1

