from src.prompts.prompt_task import prompt_task
from src.prompts.task2actions import task2actions
from src.prompts.prompt_action_will_do import prompt_action_will_do
from PIL import Image
import io
import base64
import requests
import json
import time

url = 'http://127.0.0.1:8000/inference/'
def pil_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG") 
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
def send_request(image,prompt):
    data = {
        'image': pil_image_to_base64(image),  
        'prompt': prompt
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        answer = response.json()
        answer=answer['response'][3:-4].strip()
        return answer
    else:
        assert False, f"Failed to get valid response, status code: {response.status_code}" 
        self.logger.debug(f"Failed to get valid response, status code: {response.status_code}")


def test(image,task,hand_state):
    prompt = prompt_task.format(task=task)
    answer = send_request(image, prompt)
    answer = json.loads(answer)

    task_format = answer['task']
    actions = task2actions[task_format]
    st = time.time()
    prompt = prompt_action_will_do.format(task=task_format,actions=actions,hand_state=hand_state)
    # print('prompt',prompt)
    answer = send_request(image, prompt)
    answer = answer.replace('\_', '_')
    answer = json.loads(answer)
    action = answer['action']
    print('action',action)
    if 'state' in answer:
        print('state',answer['state'])
    # if 'reason' in answer:
    #     print('reason',answer['reason'])
    # print('time',time.time()-st)
    # answer = send_request(image, 'Has the hand grasped the object? If not, if it still has a long distance to grasp')
    # print(answer)

for i in range(0,50,2):
    image = Image.open(f'/data2/liangxiwen/zkd/datasets/dataGen/DATA/1_objs_graspTargetObj_Right_0512/000009/{i:03d}.jpg')
    # image = Image.open(f'/data2/liangxiwen/zkd/datasets/dataGen/DATA/1_objs_graspTargetObj_Right_0512/000007/{i:03d}.jpg')
    task = 'Pick a cup.'
    if i<=38:
        hand_state = 'open'
    else:
        hand_state = 'closed'
    print('index ',i)
    test(image,task,hand_state)

# image = Image.open('/data2/liangxiwen/zkd/datasets/dataGen/DATA/1_objs_placeTargetObj_Right_0512/000001/000.jpg')
# task = 'place the chip'
# hand_state = 'open'
# test(image,task,hand_state)

# image = Image.open('/data2/liangxiwen/zkd/datasets/dataGen/DATA/1_objs_pushFront_Right_0512/000001/000.jpg')
# task = 'push chip front'
# hand_state = 'open'
# test(image,task,hand_state)

# image = Image.open('/data2/liangxiwen/zkd/datasets/dataGen/DATA/1_objs_pushFront_Right_0512/000001/010.jpg')
# task = 'push chip front'
# hand_state = 'open'
# test(image,task,hand_state)

# image = Image.open('/data2/liangxiwen/zkd/datasets/dataGen/DATA/1_objs_pushFront_Right_0512/000001/020.jpg')
# task = 'push chip front'
# hand_state = 'open'
# test(image,task,hand_state)

# image = Image.open('/data2/liangxiwen/zkd/datasets/dataGen/DATA/1_objs_pushFront_Right_0512/000001/030.jpg')
# task = 'push chip front'
# hand_state = 'open'
# test(image,task,hand_state)

# image = Image.open('/data2/liangxiwen/zkd/datasets/dataGen/DATA/1_objs_pushFront_Right_0512/000001/040.jpg')
# task = 'push chip front'
# hand_state = 'open'
# test(image,task,hand_state)

# image = Image.open('/data2/liangxiwen/zkd/datasets/dataGen/DATA/3_objs_graspTargetObj_Right_0429/000002/000.jpg')
# task = 'Can you please give me the GlueStick'
# hand_state = 'open'
# test(image,task,hand_state)

# image = Image.open('/data2/liangxiwen/zkd/datasets/dataGen/DATA/3_objs_graspTargetObj_Right_0429/000002/020.jpg')
# task = 'Can you please give me the GlueStick'
# hand_state = 'open'
# test(image,task,hand_state)

# image = Image.open('/data2/liangxiwen/zkd/datasets/dataGen/DATA/3_objs_graspTargetObj_Right_0429/000002/030.jpg')
# task = 'Can you please give me the GlueStick'
# hand_state = 'open'
# test(image,task,hand_state)

# image = Image.open('/data2/liangxiwen/zkd/datasets/dataGen/DATA/3_objs_graspTargetObj_Right_0429/000002/040.jpg')
# task = 'Can you please give me the GlueStick'
# hand_state = 'closed'
# test(image,task,hand_state)

# image = Image.open('/data2/liangxiwen/zkd/datasets/dataGen/DATA/1_objs_graspTargetObj_Right_0512/000023/001.jpg')
# task = 'Can you please take the GlueStick off the table and hand it to me?'
# hand_state = 'open'
# test(image,task,hand_state)

# image = Image.open('/data2/liangxiwen/zkd/datasets/dataGen/DATA/1_objs_graspTargetObj_Right_0512/000023/021.jpg')
# task = 'Can you please take the GlueStick off the table and hand it to me?'
# hand_state = 'open'
# test(image,task,hand_state)

# image = Image.open('/data2/liangxiwen/zkd/datasets/dataGen/DATA/1_objs_graspTargetObj_Right_0512/000023/031.jpg')
# task = 'Can you please take the GlueStick off the table and hand it to me?'
# hand_state = 'open'
# test(image,task,hand_state)

# image = Image.open('/data2/liangxiwen/zkd/datasets/dataGen/DATA/1_objs_graspTargetObj_Right_0512/000023/041.jpg')
# task = 'Can you please take the GlueStick off the table and hand it to me?'
# hand_state = 'closed'
# test(image,task,hand_state)
