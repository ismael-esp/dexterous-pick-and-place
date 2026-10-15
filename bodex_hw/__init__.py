"""Hardware port of the BODex pick-and-place task.

Pure numpy + rclpy. No Isaac Lab / torch import anywhere in this package,
so it runs in the ROS2 system Python. BODex grasp synthesis is reached
through ``bodex_service.BODexClient`` over a Unix socket (the heavy
torch/cuRobo stack stays in the ``bodex`` conda env).

See ``HARDWARE_PORT_PLAN.md`` at the BODex repo root for the design.
"""
