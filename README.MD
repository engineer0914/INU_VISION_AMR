# **AMR 비전 패키지**

0701 수정:
시각화 안정화
서비스 호출시 시각화
브릭, 컴포넌트 별 함수 실행

YOLO + 영상 기하 판별 구조

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

