#!/usr/bin/env bash

ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '13'}"
read -p "13 Magnet 완료. 다음 호출하려면 Enter..."

ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '34'}"
read -p "34 Battery 완료. 다음 호출하려면 Enter..."

ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '81'}"
read -p "81 Estop 완료. 다음 호출하려면 Enter..."

ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '241'}"
read -p "241 Trafficlight 완료. 다음 호출하려면 Enter..."

ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '442'}"
read -p "442 carrot 완료. 다음 호출하려면 Enter..."

ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '462'}"
read -p "462 small tree 완료. 다음 호출하려면 Enter..."

ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '711'}"
read -p "711 hammer 완료. 다음 호출하려면 Enter..."

ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '4482'}"
read -p "4482 bigcarrot 완료. 다음 호출하려면 Enter..."

ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '8518'}"
read -p "8518 burger 완료. 다음 호출하려면 Enter..."

ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '46262'}"
read -p "46262 bigtree 완료. 다음 호출하려면 Enter..."

ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '48132'}"
read -p "48132 icecream 완료. 다음 호출하려면 Enter..."

echo "전체 호출 완료"
