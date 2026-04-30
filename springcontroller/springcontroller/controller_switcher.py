import rclpy
from rclpy.node import Node
from controller_manager_msgs.srv import SwitchController

class ControllerSwitcher(Node):
    def __init__(self):
        super().__init__('controller_switcher')
        self.client = self.create_client(SwitchController, '/controller_manager/switch_controller')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

    def switch(self, activate_list, deactivate_list):
        request = SwitchController.Request()
        request.activate_controllers = activate_list
        request.deactivate_controllers = deactivate_list
        request.strictness = 2  # STRICT
        
        future = self.client.call_async(request)
        return future


def main():
    rclpy.init()
    switcher = ControllerSwitcher()


    future = switcher.switch(
        activate_list=['joint_trajectory_controller'],    # controllers to START
        deactivate_list=['velocity_controller']           # controllers to STOP
    )

    rclpy.spin_until_future_complete(switcher, future)
    switcher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
