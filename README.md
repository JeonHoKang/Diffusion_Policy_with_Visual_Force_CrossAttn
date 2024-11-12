# Compliant Object Manipulation for High Precision Prying Task Using <br \> Diffusion Policy with Force Modality

[project page](https://rros-lab.github.io/diffusion-with-force.github.io/) [data (Coming Soon)]

Jeon Ho Kang, Sagar Joshi, Ruopeng Huang, and Satyandra K. Gupta

University of Southern California

![System Architecture](imgs/overview_system.png)

Baseline code for diffusion policy was derived from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)

All  **Real** tags are for real robot implementation

However, [data_util.py](data_util.py) is shared amongst them.


## Dependencies

Create Conda Environment (Recommended) and run:


'''
pip install requirements.txt
'''

## Real Robot 

For all demonstrations, we used [KUKA IIWA 14 Robot](https://www.kuka.com/en-de/products/robot-systems/industrial-robots/lbr-iiwa)


## Real Robot Data for Prying Task (Zarr File)
(Coming Soon)
Data collected on Kuka IIWA 14 robot containing robot state, image, force and action will be published [here]


To collect your own data:

After obtaininig joint state from handguiding or any other methods,

Run

'''

$ python robot_data_collection_joint_state.py

'''


## Training Your Own Policy


After loading your own zarr file or ours in [real_robot_network.py](real_robot_network.py)

'''
$ python trtrain_real.py
'''

You can select or create your own [Config](config) file for training configuration


## Inference

'''
$ python inference_real.py
'''


## Acknowledgement

+ Diffusion policy was adapted from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)