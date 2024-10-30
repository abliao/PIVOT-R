prompt_scene = """
Given a task, which is for a mobile Franka panda robotic arm to learn a manipulation skill in the simulator. Your task is to help me break down the process of the robot performing the task into several actions to help the robot better understand and execute.

Capabilities: The task can only be completed with a robotic arm, which can move, rotate and clamp.

You should output response using the same format as the following json file
```
{{
    "scene": "You should description the scene"
}}
```

Here is one example:

Input:
Task: pick up the green bottle.

Output:
```
{{
    "scene": "On the table, there is a green bottle, a blue and white bottle."
}}
```

Can you do it for the following input:
Task: {task}
"""