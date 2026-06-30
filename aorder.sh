#!/usr/bin/env bash

ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '1'}"
read -p "1 완료. 다음 호출하려면 Enter..."

ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '2'}"
read -p "2 완료. 다음 호출하려면 Enter..."

ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '3'}"
read -p "3 완료. 다음 호출하려면 Enter..."

ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '4'}"
read -p "4 완료. 다음 호출하려면 Enter..."

ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '5'}"
read -p "5 완료. 다음 호출하려면 Enter..."

ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '6'}"
read -p "6 완료. 다음 호출하려면 Enter..."

ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '7'}"
read -p "7 완료. 다음 호출하려면 Enter..."

ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '8'}"
read -p "8 완료. 다음 호출하려면 Enter..."

echo "전체 호출 완료"
