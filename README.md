# Modification
It has been adapted to [Torchcraft v 1.3.0](https://github.com/TorchCraft/TorchCraft/releases) and [BWAPI v 4.1.2](https://github.com/bwapi/bwapi/releases).   
It doesn't need torchcraft-py anymore because [Torchcraft v 1.3.0](https://github.com/TorchCraft/TorchCraft/releases) supports python API.   
Wrap an easy_env for multi_units battle

reference:
[alibaba/gym-starcraf](https://github.com/alibaba/gym-starcraft)
[NoListen/gym-starcraft](https://github.com/NoListen/gym-starcraft)

# gym-starcraft
Gym StarCraft is an environment bundle for OpenAI Gym. It is based on [Facebook's TorchCraft](https://github.com/TorchCraft/TorchCraft), which is a bridge between Torch and StarCraft for AI research.

## Installation

1. Install [OpenAI Gym](https://github.com/openai/gym) and its dependencies.

2. Install [TorchCraft](https://github.com/TorchCraft/TorchCraft) and its dependencies. You can skip the torch client part. 

3. Install the package itself:
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
    
