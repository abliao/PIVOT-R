# Due to VLMs perform poorly under simulation, simplify the setting of actions
task2actions={
    'place':{
        "actions": [
        {
            "action": "move down",
            "description": "chosen when ready to place the object."
        },
    ]
    },
    'push front':{
        "actions": [
        {
            "action": "close to",
            "description": "chosen when other action can\'t be done."
        },
        {
            "action": "push front",
            "description": "chosen when the hand is near the object."
        },
    ]
    },
    'push left':{
        "actions": [
        {
            "action": "close to",
            "description": "chosen when other action can\'t be done."
        },
        {
            "action": "push left",
            "description": "chosen when the hand is near the object."
        },
    ]
    },
    'push right':{
        "actions": [
        {
            "action": "close to",
            "description": "chosen when other action can\'t be done."
        },
        {
            "action": "push right",
            "description": "chosen when the hand is near the object."
        },
    ]
    },
    'pick up':{
        "actions": [
        {
            "action": "close to",
            "description": "chosen when the hand is far from the object."
        },
        {
            "action": "move up",
            "description": "chosen after the hand is closed."
        },
    ]
    },
    'move near':{
        "actions": [
        {
            "action": "close to",
            "description": "chosen when other action can\'t be done."
        },
        {
            "action": "grasp",
            "description": "chosen when the hand is open and the hand is near the object."
        },
        {
            "action": "move up",
            "description": "chosen when the hand is closed."
        },
        {
            "action": "close to",
            "description": "chosen when grasped the object and ready to move near another object."
        },
        {
            "action": "move down",
            "description": "chosen when ready to place the object."
        },
    ]
    },
    'knock over':{
        "actions": [
        {
            "action": "close to",
            "description": ""
        },
    ]
    },
}