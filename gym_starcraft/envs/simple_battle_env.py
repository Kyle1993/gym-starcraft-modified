import numpy as np
import math

from gym import spaces
import torchcraft.Constants as tcc
# import gym_starcraft.utils as utils
import gym_starcraft.envs.starcraft_env as sc


class Unit_State(object):
    def __init__(self, unit, id):
        #print unit.groundRange, "Range"
        self.id = id
        self.health = unit.health
        self.x = unit.x
        self.y = unit.y
        self.shield = unit.shield
        self.attackCD = unit.groundCD
        self.velocityX = unit.velocityX        # speed
        self.velocityY = unit.velocityY
        self.type = unit.type
        self.groundATK = unit.groundATK
        self.groundRange = unit.groundRange    # this can be inferred from training
        self.under_attack = unit.under_attack  # True or Flase
        self.attacking = unit.attacking        # True or False
        self.moving = unit.moving              # True or False
        self.die = (unit.health <= 0)
        self.max_health = unit.max_health
        self.max_shield = unit.max_shield
        self.pixel_size_x = unit.pixel_size_x
        self.pixel_size_y = unit.pixel_size_y
        self.delta_health = 0

    def update(self,unit):
        self.delta_health = self.health-unit.health
        self.health = unit.health
        self.x = unit.x
        self.y = unit.y
        self.shield = unit.shield
        self.attackCD = unit.groundCD
        self.velocityX = unit.velocityX        # speed
        self.velocityY = unit.velocityY
        self.groundATK = unit.groundATK
        self.groundRange = unit.groundRange    # this can be inferred from training
        self.under_attack = unit.under_attack  # True or Flase
        self.attacking = unit.attacking        # True or False
        self.moving = unit.moving              # True or False
        self.die = (unit.health <= 0)
        self.pixel_size_x = unit.pixel_size_x
        self.pixel_size_y = unit.pixel_size_y

    def set_die(self):
        self.die = True
        self.delta_health = self.health        # if die, detal health = how much health it has before
        self.health = 0
        self.shield = 0
        


class SimpleBattleEnv(sc.StarCraftEnv):
    def __init__(self, server_ip, server_port,hyperparameters, speed=0, self_play=False):
        super(SimpleBattleEnv, self).__init__(server_ip, server_port, speed,
                                              hyperparameters['FRAME_SKIP'], self_play,
                                              hyperparameters['MAX_STEP'])

        self.mapname = None

        self.MYSELF_NUM = hyperparameters['MYSELF_NUM']
        self.ENEMY_NUM = hyperparameters['ENEMY_NUM']
        self.DISTANCE_FACTOR = hyperparameters['DISTANCE_FACTOR']
        self.ACTION_NUM = hyperparameters['NB_ACTION']
        self.SCREEN_BOX = hyperparameters['SCREEN_BOX']  # (left_top,rigth_down)
        self.POSITION_RANGE = hyperparameters['POSITION_RANGE']
        self.DIE_REWARD = hyperparameters['DIE_REWARD']  # 500 means max_step
        self.HEALTH_REWARD_WEIGHT = hyperparameters['HEALTH_REWARD_WEIGHT']
        self.WIN_REWARD_WEIGHT = hyperparameters['WIN_REWARD_WEIGHT']

        self.myself_alive_list = None    # like [1,1,0,0,1]
        self.enemy_alive_list = None

        self.current_state = None  # current_state['myself']: units state of myself; current_state['enemy']:units of enemy

        self.myself_total_hp = 0   # current myself_units total hp
        self.enemy_total_hp = 0
        self.win = False           # in this episode

        self.myself_unit_dict = None # map between myself unit.id and index of self.current_state['myself']
        self.enemy_unit_dict = None


        self.init_myself_total_hp = None  # in reward compute
        self.init_enemy_total_hp = None

        print('------------- Enveriment config ---------------')
        print('ip:',server_ip)
        print('port:',server_port)
        print('MYSELF_NUM:',self.MYSELF_NUM)
        print('ENEMY_NUM:',self.ENEMY_NUM)

    def _action_space(self):
        # attack or move, move_degree, move_distance
        action_low = [-1.0, -1.0, -1.0]
        action_high = [1.0, 1.0, 1.0]
        return spaces.Box(np.array(action_low), np.array(action_high))


    # needs to be modified
    def _observation_space(self):
        print('return the current_state, have no space, you can extract the state you want')
        return spaces.Box(np.array([1.]),np.array([1.]))

    def _make_commands(self, actions):
        cmds = []
        if self.state is None or actions is None:
            return cmds

        assert (len(actions) == self.MYSELF_NUM)
        assert (len(actions[0]) == self.ACTION_NUM)

        for uid in range(self.MYSELF_NUM):
            unit = self.current_state['myself'][uid]

            if not unit.die:   # if not die
                if actions[uid][0]>0:
                    # attack
                    enemy_id =self.get_closest_enemy2(unit,actions[uid])  # get the colse enemy of target position
                    if enemy_id is None:
                        x_ = actions[uid][1] * self.DISTANCE_FACTOR
                        y_ = actions[uid][2] * self.DISTANCE_FACTOR
                        x_target,y_target = self.get_position2(unit.x,unit.y,x_,y_)
                        x_target,y_target = max(x_target,self.SCREEN_BOX[0][0]),max(y_target,self.SCREEN_BOX[0][1])
                        x_target,y_target = min(x_target,self.SCREEN_BOX[1][0]),min(y_target,self.SCREEN_BOX[1][1])
                        cmd = [tcc.command_unit_protected, unit.id, tcc.unitcommandtypes.Move, -1, int(x_target), int(y_target)]
                    else:
                        cmd = [tcc.command_unit_protected, unit.id, tcc.unitcommandtypes.Attack_Unit, enemy_id]
                        # cmds.append(cmd)
                else:
                    # move
                    # degree = actions[uid][1] * 180
                    # distance = actions[uid][2] * DISTANCE_FACTOR
                    # x_target,y_target = self.get_position(unit.x,unit.y,degree,distance)
                    x_ = actions[uid][1] * self.DISTANCE_FACTOR
                    y_ = actions[uid][2] * self.DISTANCE_FACTOR
                    x_target,y_target = self.get_position2(unit.x,unit.y,x_,y_)
                    x_target,y_target = max(x_target,self.SCREEN_BOX[0][0]),max(y_target,self.SCREEN_BOX[0][1])
                    x_target,y_target = min(x_target,self.SCREEN_BOX[1][0]),min(y_target,self.SCREEN_BOX[1][1])
                    cmd = [tcc.command_unit_protected, unit.id, tcc.unitcommandtypes.Move, -1, int(x_target), int(y_target)]

                cmds.append(cmd)

        if len(cmds)==0:
            return [[tcc.noop]]

        return cmds

    def _make_observation(self):

        self.update_self()         # mainly in self.current_state, which is main state, we can extract all info from it
        obs = self._compute_obs()  # return the obs you want

        # self.print_attr()

        return obs

    # return the obs you want, compute by self.current_state
    # but in order to make the env stable, it is recommanded to compute in you own code
    def _compute_obs(self):
        return self.current_state

    # provide two options,total_reward or separate_reward, you can even calculate by yourself
    def _compute_reward(self):

        reward = self.compute_reward_separately2()
        # reward = self.compute_reward_total()

        return reward

    def compute_reward_total(self):
        # delta health between step
        myself_detla_health = 0
        enemy_detla_health = 0
        for unit in self.current_state['myself']:
            myself_detla_health += unit.delta_health
        for unit in self.current_state['enemy']:
            enemy_detla_health += unit.delta_health
        myself_detla_health = self.HEALTH_REWARD_WEIGHT * float(myself_detla_health) / float(self.init_myself_total_hp)
        enemy_detla_health = self.HEALTH_REWARD_WEIGHT * float(enemy_detla_health) / float(self.init_enemy_total_hp)
        health_reward = (enemy_detla_health - myself_detla_health)
        win_reward = float(self.win) * float(sum(self.myself_alive_list))
        reward = health_reward + win_reward

        return reward

    def compute_reward_separately(self):

        rewards = []

        myself_detla_health = 0
        enemy_detla_health = 0
        for unit in self.current_state['enemy']:
            enemy_detla_health += unit.delta_health
        enemy_detla_health = float(enemy_detla_health) / float(self.init_enemy_total_hp)

        for unit in self.current_state['myself']:
            if unit.die:
                rewards.append(self.DIE_REWARD)
            else:
                myself_detla_health += unit.delta_health
                unit_reward = self.ENEMY_NUM*enemy_detla_health-unit.delta_health/unit.max_health
                rewards.append(self.HEALTH_REWARD_WEIGHT*unit_reward)
        myself_detla_health = float(myself_detla_health) / float(self.init_myself_total_hp)

        health_reward = self.HEALTH_REWARD_WEIGHT * (enemy_detla_health - myself_detla_health)
        win_reward = float(self.win) * float(sum(self.myself_alive_list))
        global_reward = health_reward + win_reward
        rewards.insert(0,global_reward)

        return rewards

    def compute_reward_separately2(self):

        rewards = []

        myself_detla_health = 0
        enemy_detla_health = 0
        for unit in self.current_state['enemy']:
            enemy_detla_health += unit.delta_health
        enemy_detla_health = float(enemy_detla_health) / float(self.init_enemy_total_hp)

        for unit in self.current_state['myself']:
            if unit.die:
                rewards.append(self.DIE_REWARD)
            else:
                myself_detla_health += unit.delta_health
                health_reward = unit.attacking*max(1,float(self.ENEMY_NUM)/float(self.MYSELF_NUM))*enemy_detla_health # -unit.delta_health/unit.max_health
                win_reward = float(self.win) * float(sum(self.myself_alive_list)) * float(self.ENEMY_NUM)/float(self.MYSELF_NUM)
                rewards.append(self.HEALTH_REWARD_WEIGHT*health_reward+self.WIN_REWARD_WEIGHT*win_reward)

        rewards.insert(0,sum(rewards)/self.MYSELF_NUM)

        return rewards

    
    def reset_data(self):
        # skip the init state, there is no unit at the beginning of game
        while len(self.state.units) == 0 or len(self.state.units[0]) != self.MYSELF_NUM or len(self.state.units[1]) != self.ENEMY_NUM:
            self.client.send([])
            self.state = self.client.recv()

        # reset the variable
        self.current_state = None  # current_state['myself']: units state of myself; current_state['enemy']:units of enemy

        self.myself_total_hp = 0 
        self.enemy_total_hp = 0
        self.win = False 

        self.myself_alive_list = None
        self.enemy_alive_list = None

        self.myself_unit_dict = None
        self.enemy_unit_dict = None


        self.init_myself_total_hp = None
        self.init_enemy_total_hp = None

        self.advanced_termination = True

    # calculate by degree and distance
    def get_position(self,x,y,degree,distance):
        theta = math.radians(degree)
        return x + distance * math.cos(theta), y + distance * math.sin(theta)

    # get the closest enemy from out attack position, return unit.id
    def get_closest_enemy(self, unit, action):
        degree = action[1] * 180
        # TODO consider to change the target range.
        distance = (action[2] + 1)*self.DISTANCE_FACTOR
        tx, ty = self.get_position(unit.x,unit.y,degree, distance)
        target_id = self.compute_candidate(tx, ty)
        return target_id

    # calculate by x,y
    def get_position2(self,x,y,x_,y_):
        return x+x_,y+y_
        
    def get_closest_enemy2(self,unit,action):
        x_ = action[1] * self.DISTANCE_FACTOR
        y_ = action[2] * self.DISTANCE_FACTOR
        tx,ty = self.get_position2(unit.x,unit.y,x_,y_)
        target_id = self.compute_candidate(tx,ty)
        return target_id

    def compute_candidate(self, tx, ty):   # find the most nearly enemy unit in the POSITION_RANGE of target position
        target_id = None
        d = self.POSITION_RANGE
        for enemy_unit in self.state.units[1]:
            # if self.enemy_alive_list[self.enemy_unit_dict[enemy_unit.id]] == 0:
            #     continue
            td = (enemy_unit.x - tx)**2 + (enemy_unit.y - ty)**2
            if td < d :
                target_id = enemy_unit.id
                d = td
        return target_id

    def update_self(self):   # undate some info extract from self.current_state

        self.myself_current_hp = 0
        self.enemy_current_hp = 0
        self.mapname = self.state.map_name

        # if current_state is init_state
        if self.current_state is None:
            self.current_state = {}

        # update the info about myself_unit
        # if current_state is init_state
        if self.myself_unit_dict is None:
            self.myself_unit_dict = {}
            self.current_state['myself'] = [] 
            self.myself_alive_list = [1 for _ in range(self.MYSELF_NUM)]
            # self.init_myself_total_hp = 0
            for unit_index,unit in enumerate(self.state.units[0]):
                self.myself_unit_dict[unit.id]= unit_index
                self.current_state['myself'].append(Unit_State(unit,unit.id))
                self.myself_current_hp += unit.health
            self.init_myself_total_hp = self.myself_current_hp
                # self.init_myself_total_hp += unit.max_health
        # current_state is not init_state
        else:
            self.myself_alive_list = [0 for _ in range(self.MYSELF_NUM)]
            # self.current_state['myself'] = [None for _ in range(self.MYSELF_NUM)]
            for unit in self.state.units[0]:
                uid = self.myself_unit_dict[unit.id]
                self.current_state['myself'][uid].update(unit)
                self.myself_alive_list[uid] = 1
                self.myself_current_hp += unit.health
            for index in range(self.MYSELF_NUM):        # if the unit die, set None
                if self.myself_alive_list[index]==0:
                    self.current_state['myself'][index].set_die()


        # update the info about enemy_unit
        # if current_state is init_state
        if self.enemy_unit_dict is None:
            self.enemy_unit_dict = {}
            self.current_state['enemy'] = []
            self.enemy_alive_list = [1 for _ in range(self.ENEMY_NUM)]
            # self.init_enemy_total_hp = 0
            for unit_index,unit in enumerate(self.state.units[1]):
                self.enemy_unit_dict[unit.id]= unit_index
                self.current_state['enemy'].append(Unit_State(unit,unit.id))
                self.enemy_current_hp += unit.health
            self.init_enemy_total_hp = self.enemy_current_hp
                # self.init_enemy_total_hp += unit.max_health
        # current_state is not init_state
        else:
            self.enemy_alive_list = [0 for _ in range(self.ENEMY_NUM)]
            # self.current_state['enemy'] = [None for _ in range(self.ENEMY_NUM)]
            for unit in self.state.units[1]:
                uid = self.enemy_unit_dict[unit.id]
                self.current_state['enemy'][uid].update(unit)
                self.enemy_alive_list[uid] = 1
                self.enemy_current_hp += unit.health
            for index in range(self.ENEMY_NUM):
                if self.enemy_alive_list[index]==0:
                    self.current_state['enemy'][index].set_die()


        # # init the init_total_hp
        # if self.init_myself_total_hp is None:
        #     self.init_myself_total_hp = self.myself_current_hp
        # if self.init_enemy_total_hp is None:
        #     self.init_enemy_total_hp = self.enemy_current_hp

        assert len(self.current_state['myself']) == self.MYSELF_NUM
        assert len(self.myself_alive_list) == self.MYSELF_NUM
        assert int(sum(self.myself_alive_list)) == len(self.state.units[0])
        assert len(self.current_state['enemy']) == self.ENEMY_NUM
        assert len(self.enemy_alive_list) == self.ENEMY_NUM
        assert int(sum(self.enemy_alive_list)) == len(self.state.units[1])

    # return done
    def _check_done(self):
        # if int(sum(self.myself_alive_list)) == 0 or int(sum(self.enemy_alive_list)) == 0:
        if len(self.state.units[0])==0 or len(self.state.units[1])==0:
            if len(self.state.units[0])>0 and len(self.state.units[1])==0:
                self.win = True
                self.episode_wins += 1
                self.advanced_termination = True
            else:
                self.win = False
                self.advanced_termination = True
            return True
        if self.episode_steps >= self.max_episode_steps:
            print("max_episode_steps done!")
            self.advanced_termination = True
            return True
        return False

    # print info for debug
    def print_attr(self):

        print('-------------------------------------------')
        # print('self.state.units[0]:',len(self.state.units[0]))
        # print('self.state.units[1]:',len(self.state.units[1]))
        # print('myself_alive_list:',self.myself_alive_list)
        # print('enemy_alive_list:',self.enemy_alive_list)

        # print('myself_alive_list:',self.myself_alive_list)
        # print('enemy_alive_list:',self.enemy_alive_list)
        # print('current_state',len(self.current_state['myself']),len(self.current_state['enemy']))
        
        myself_info = []
        enemy_info = []
        for unit in self.state.units[0]:
            myself_info.append(unit.health)
        for unit in self.state.units[1]:
            enemy_info.append(unit.health)
        print(myself_info)
        print(enemy_info)

        alive_myselfinfo = []
        for unit in self.current_state['myself']:
            alive_myselfinfo.append(unit.max_health)
        print(alive_myselfinfo)
        # print(sum(myself_info),sum(enemy_info))
        # print(self.win)

        # print('myself_unit_dict:',self.myself_unit_dict)
        # print('enemy_unit_dict:',self.enemy_unit_dict)

        # print('myself_ids:',myself_ids)
        # print('enemy_ids:',enemy_ids)

    def getMapName(self):

        return self.mapname


