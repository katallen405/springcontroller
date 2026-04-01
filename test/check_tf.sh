for link in base_link shoulder_link upper_arm_link forearm_link wrist_1_link wrist_2_link wrist_3_link tool0; do
    echo "=== ur3e_$link ==="
    ros2 run tf2_ros tf2_echo ur3e_base_link ur3e_$link 2>/dev/null | grep -A3 "Translation"
    echo ""
done
