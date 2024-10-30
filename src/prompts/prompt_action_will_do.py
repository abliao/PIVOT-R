prompt_action_will_do = """
Your task is to determine the next action to be performed based on the current state and the list of available actions. You should output only one action at a time.

Please format your response as follows:
```
{{
    "state": "The state of hand and object, such as the hand is far from the object",
    "action": "The action name"
}}
```

Example:
{{
    "state": "The hand is closed and grasping the object",
    "action": "move up"
}}

Make sure to consider the current task, the list of possible actions, and the state of the robotic hand when determining the next action.

For the following task, please choose the action match the description:
Task: {task}
Actions: {actions}
Hand State: {hand_state}
"""