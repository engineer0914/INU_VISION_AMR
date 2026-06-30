#!/usr/bin/env bash

a="13"
repeat=10

for i in $(seq 1 $repeat); do
  echo "[$i/$repeat] target_color=${a} 호출"

  ros2 service call /get_target_pose arm_interfaces/srv/GetTargetPose "{target_color: '${a}'}"

  read -p "${a} ${i}회차 완료. 다음 호출하려면 Enter..."
done

echo "전체 반복 완료"
