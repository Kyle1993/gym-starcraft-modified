# Modification
It has been adapted to [Torchcraft v 1.3.0](https://github.com/TorchCraft/TorchCraft/releases) and [BWAPI v 4.1.2](https://github.com/bwapi/bwapi/releases).   
It doesn't need torchcraft-py anymore because [Torchcraft v 1.3.0](https://github.com/TorchCraft/TorchCraft/releases) supports python API.   
Support python3  
Wrap an easy_env for multi_units battle( need to change the game settings:bwapi.ini and add correaponding maps to /Mpas)  

reference:  
[alibaba/gym-starcraf](https://github.com/alibaba/gym-starcraft)  
[NoListen/gym-starcraft](https://github.com/NoListen/gym-starcraft)

# gym-starcraft
Gym StarCraft is an environment bundle for OpenAI Gym. It is based on [Facebook's TorchCraft](https://github.com/TorchCraft/TorchCraft), which is a bridge between Torch and StarCraft for AI research.

## Installation

1. Install [OpenAI Gym](https://github.com/openai/gym) and its dependencies.

2. Install [TorchCraft](https://github.com/TorchCraft/TorchCraft) and its dependencies. You can skip the torch client part. 

3. install /TorchCraft/py.
    ```
    cd /torchcraft/py
    pip install -e .
    ```

4. Install the package itself:
    ```
    git clone https://github.com/Kyle1993/gym-starcraft-modified.git
    cd gym-starcraft-modified
    pip install -e .
    ```

## Usage
1. Start StarCraft server with BWAPI by Chaoslauncher.

2. Run examples:

    ```
    cd examples
    python random_agent.py --ip $server_ip --port $server_port 
    ```
    
    The `$server_ip` and `$server_port` are the ip and port of the server running StarCraft.   
    
# Some Tips  
1. It's no need to install torchcraft-py  

2. Chose BWAPI v4.12 and StarCraft1 vxxx, don't forget install BroodWar patch.   

3. Replace the map folder `/xxx` with the same name folder in my project  

4. I have supplied some battle map in xxxx folder, the name like"3mv1z" means 3xxx vs 1xxx, you can chose them by change the bwapi config file: `$STARCRAFT/bwapi-data/bwapi.ini`. You can even create new battle maps , __But please make sure the env parameter`myself_num` and `enemy_num` is equal to the content of your map(number of the units you control and number of units computer control)__

5. The env will return the basic observation data in each step, you should extract the state info you need in you own code.  
the observation data look like this`{'myself':[M_unit0, M_unit1, ... , M_unitm], 'enemy':[E_unit0, E_unit1, ... ,E_unitn]}`  
6. Each unit have a 3-dim action, in alibaba code, it is represented like [move_or _attack, degree, distance], I offer a alternative representation [move_or_attack, position_x, position_y]