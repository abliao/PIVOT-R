# PIVOT-R
This repo contains code for the paper:
### PIVOT-R: Primitive-Driven Waypoint-Aware World Model for Robotic Manipulation ###

[Project Website](https://abliao.github.io/PIVOT-R/) || [Arxiv](https://arxiv.org/abs/2410.10394)

<p align="center">
<img src="images/pipeline.png" height=400px alt="Overview">
</p>

PIVOT-R is a primitive-driven waypoint-aware world model with asynchronous hierarchical executors. It only focuses on the prediction of waypoints related to the manipulation task, and it is easier to predict key nodes in the manipulation task than other methods. In addition, PIVOT-R sets different execution frequencies for different modules to have higher execution efficiency and lower redundancy.

## Abstract
Language-guided robotic manipulation is a challenging task that requires an embodied agent to follow abstract user instructions to accomplish various complex manipulation tasks. Previous work trivially fitting the data without revealing the relation between instruction and low-level executable actions, these models are prone to memorizing the surficial pattern of the data instead of acquiring the transferable knowledge, and thus are fragile to dynamic environment changes. To address this issue, we propose a PrIrmitive-driVen waypOinT-aware world model for Robotic manipulation (PIVOT-R) that focuses solely on the prediction of task-relevant waypoints. Specifically, PIVOT-R consists of a Waypoint-aware World Model (WAWM) and a lightweight action prediction module. The former performs primitive action parsing and primitive-driven waypoint prediction, while the latter focuses on decoding low-level actions. Additionally, we also design an asynchronous hierarchical executor (AHE), which can use different execution frequencies for different modules of the model, thereby helping the model reduce computational redundancy and improve model execution efficiency.

## Installation
- Install dependencies
```
pip install -r requirements.txt
```
- Download [simulator](https://drive.google.com/file/d/1GRe5OFmQdMJIIs8EU7kobWoCyFVfMHct/view?usp=sharing)
- Download [checkpoint](https://drive.google.com/file/d/12uDk9m4vxqkoZCd_7vNsz_NXhaRY49t7/view?usp=drive_link) and place it in "runs".
- Download [test dataset](https://drive.google.com/file/d/15V82C2RCyfEfJKA_HYomf-YuTaiG-4gK/view?usp=sharing) and place it in "datasets". if need training, please download [train dataset](https://pan.baidu.com/s/1u5u7-gS2RjUBprMArAyBZA?pwd=7mo4 )
## Evaluation
First, open the simulator
```
cd /path/to/simulator
./HARIX_RDKSim.sh -graphicsadapter=0 -port=30007 -RenderOffScreen
```
open a new command line, and run
```
cd LLAVA
python -m llava.serve.api --model-path liuhaotian/llava-v1.5-13b --temperature 0.0
```
Final, open a new command line
```
python src/tester.py
```

## Training
```
./train.sh
```

## Citation
If you find this code useful in your work, please consider citing
```shell
@misc{zhang2024pivotrprimitivedrivenwaypointawareworld,
      title={PIVOT-R: Primitive-Driven Waypoint-Aware World Model for Robotic Manipulation}, 
      author={Kaidong Zhang and Pengzhen Ren and Bingqian Lin and Junfan Lin and Shikui Ma and Hang Xu and Xiaodan Liang},
      year={2024},
      eprint={2410.10394},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2410.10394}, 
}
```
