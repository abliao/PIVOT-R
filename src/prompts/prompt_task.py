prompt_task = """
Given a task for a mobile Franka Panda robotic arm to learn a manipulation skill in a simulator, your objective is to break down the process into discrete actions that the robot can understand and execute effectively.

Capabilities: The robot arm can perform movements, rotations, and clamping actions.

Your response should adhere to the format of the following JSON template:
```
{{
    "task": "task name"
}}
```

Task list: ["pick up", "push front", "push left", "push right", "place", "open", "close", "turn", "knock over", "move near"]

Action mappings for tasks:
- "pick up": ["close to", "grasp", "move up"]
- "push front": ["close to", "push front"]
- "push left": ["close to", "push left"]
- "push right": ["close to", "push right"]
- "place": ["move down", "release"]
- "open": ["open"]
- "close": ["close"]
- "turn": ["rotate"]
- "knock over": ["close to", "push front"]
- "move near": ["close to", "grasp", "move up", "close to", "move down", "release"]

Examples:

# Input:
## Task: push milk front.

# Output:
{{
    "task": "push front"
}}

# Input:
## Task: Place Yogurt.

# Output:
{{
    "task": "place"
}}

For the following task:
Task: {task}
"""
