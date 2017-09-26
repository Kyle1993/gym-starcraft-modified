import numpy as np

from gym import spaces
import torchcraft.Constants as tcc
import gym_starcraft.utils as utils
import gym_starcraft.envs.starcraft_env as sc

DISTANCE_FACTOR = 16
# MYSELF_NUM = 5
# ENEMY_NUM = 5
ACTION_NUM = 3

class Unit_State(object):
    def __init__(self, unit, id):
        #print unit.groundRange, "Range"
        self.id = id
        self.health = unit.health
        self.x = unit.x
        self.y = unit.y
        self.shield = unit.shield
        self.attackCD = unit.groundCD
        # the type is encoded to be one-hot, but is not considered temporally
        self.type = unit.type
        self.groundATK = unit.groundATK
        self.groundRange = unit.groundRange # this can be inferred from training
        self.under_attack = unit.under_attack
        self.attacking = unit.attacking
        self.moving = unit.moving
        #TODO maintain a TOP-K list
        self.die = (unit.health <= 0)
        self.max_health = unit.max_health
        self.max_shield = unit.max_shield
        self.pixel_size_x = unit.pixel_size_x
        self.pixel_size_y = unit.pixel_size_y
        self._delta_health = 0


class EasyBattleEnv(sc.StarCraftEnv):
    def __init__(self, server_ip, server_port, myself_num, enemy_num, speed=0, frame_skip=0,
                 self_play=False, max_episode_steps=2000):
        super(EasyBattleEnv, self).__init__(server_ip, server_port, speed,
                                              frame_skip, self_play,
                                              max_episode_steps)

        self.MYSELF_NUM = myself_num
        self.ENMEY_NUM = enemy_num

        self.current_state = None  # current_state['myself']: units state of myself; current_state['enemy']:units of enemy

        self.myself_total_hp = 0   # current myself_units total hp
        self.enemy_total_hp = 0
        self.win = False           # if win in this episode

        self.myself_ids = None     # list of myself_units' unit.id
        self.enemy_ids = None      # list of enemy_units' unit.id

        self.myself_unit_dict = None # map between myself unit.id and index of self.current_state['myself']
        self.enemy_unit_dict = None


        self.init_myself_total_hp = None  # in reward compute
        self.init_enemy_total_hp = None

    def _action_space(self):
        # attack or move, move_degree, move_distance
        action_low = [-1.0, -1.0, -1.0]
        action_high = [1.0, 1.0, 1.0]
        return spaces.Box(np.array(action_low), np.array(action_high))


    # needs to be modified
    def _observation_space(self):
        # obs_low = [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # obs_high = [100.0, 100.0, 1.0, 1.0, 1.0, 50.0, 100.0, 100.0, 1.0, 1.0]
        # return spaces.Box(np.array(obs_low), np.array(obs_high))
        print('return the current_state, have no space, you can extract the state you want')
        return spaces.Box(np.array([1.]),np.array([1.]))

    def _make_commands(self, actions):
        cmds = []
        if self.state is None or actions is None:
            return cmds

        # print(actions)
        assert (len(actions) == self.MYSELF_NUM)
        assert (len(actions[0]) == ACTION_NUM)

        for uid in range(self.MYSELF_NUM):
            unit = self.current_state['myself'][uid]

            if unit is not None:   # if not die
                if actions[uid][0]>0:
                    # attack
                    enemy_id =self.get_closest_enemy(unit,actions[uid])  # input unit_state, return id of closest enemy
                    if enemy_id is None:
                        continue
                    cmd = [tcc.command_unit_protected, unit.id, tcc.unitcommandtypes.Attack_Unit, enemy_id]
                else:
                    # move
                    degree = actions[uid][1] * 180
                    distance = (actions[uid][2] + 1) * DISTANCE_FACTOR
                    x_target,y_target = utils.get_position2(degree,distance,unit.x,unit.y)  # get the colse enemy of our target position
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


    def _compute_reward(self):
        reward = 0
        hp_reward = float(self.myself_current_hp)/float(self.init_myself_total_hp)+(1.-float(self.enemy_current_hp)/self.init_enemy_total_hp)
        win_reward = float(self.win)
        # print(float(self.myself_total_hp)/float(self.init_myself_total_hp),(1.-float(self.enemy_total_hp)/self.init_enemy_total_hp),win_reward)
        reward = hp_reward/2.+win_reward
        return reward-0.8  # normalization
    
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

        self.myself_ids = None
        self.enemy_ids = None

        self.myself_unit_dict = None
        self.enemy_unit_dict = None


        self.init_myself_total_hp = None
        self.init_enemy_total_hp = None

        self.advanced_termination = True

    # get the closest enemy from out attack position, return unit.id
    def get_closest_enemy(self, unit, action):
        degree = action[1] * 180
        # TODO consider to change the target range.
        distance = (action[2] + 1) * unit.groundRange/2.  # at most 2*DISTANCE_FACTOR
        tx, ty = utils.get_position2(degree, distance, unit.x, unit.y)
        # TODO only consider those within the groundRange
        # limit the distance between target and target unit
        target_id = self.compute_candidate(tx, ty)
        return target_id

    def compute_candidate(self, tx, ty):
        target_id = None
        # set it to be a big positive integer
        d = 90000
        # not consider the same situation
        for enemy_unit in self.state.units[1]:
            if self.enemy_alive_list[self.enemy_unit_dict[enemy_unit.id]] == 0:
                continue
            # TODO if the value is normalized, pay attention to change the value here.
            td = (enemy_unit.x - tx)**2 + (enemy_unit.y - ty)**2
            if td < d :
                target_id = enemy_unit.id
                d = td
        return target_id

    def update_self(self):   # undate some info extract from self.current_state


        self.myself_current_hp = 0
        self.enemy_current_hp = 0

        # if current_state is init_state
        if self.current_state is None:
            self.current_state = {}

        # update the info about myself_unit(just covert it to Unit_State)
        # if current_state is init_state
        if self.myself_unit_dict is None:
            self.myself_unit_dict = {}
            self.current_state['myself'] = [] 
            self.myself_alive_list = [1 for _ in range(self.MYSELF_NUM)]
            for unit_index,unit in enumerate(self.state.units[0]):
                self.myself_unit_dict[unit.id]= unit_index
                self.current_state['myself'].append(Unit_State(unit,unit.id))
                self.myself_current_hp += unit.health
        # current_state is not init_state
        else:
            self.myself_alive_list = [0 for _ in range(self.MYSELF_NUM)]
            self.current_state['myself'] = [None for _ in range(self.MYSELF_NUM)]
            for unit in self.state.units[0]:
                uid = self.myself_unit_dict[unit.id]
                self.current_state['myself'][uid] = Unit_State(unit,unit.id)
                self.myself_alive_list[uid] = 1
                self.myself_current_hp += unit.health

        # update the info about enemy_unit(just covert it to Unit_State)
        # if current_state is init_state
        if self.enemy_unit_dict is None:
            self.enemy_unit_dict = {}
            self.current_state['enemy'] = []
            self.enemy_alive_list = [1 for _ in range(self.ENEMY_NUM)]
            for unit_index,unit in enumerate(self.state.units[1]):
                self.enemy_unit_dict[unit.id]= unit_index
                self.current_state['enemy'].append(Unit_State(unit,unit.id))
                self.enemy_current_hp += unit.health
        # current_state is not init_state
        else:
            self.enemy_alive_list = [0 for _ in range(self.ENEMY_NUM)]
            self.current_state['enemy'] = [None for _ in range(self.ENEMY_NUM)]
            for unit in self.state.units[1]:
                uid = self.enemy_unit_dict[unit.id]
                self.current_state['enemy'][uid] = Unit_State(unit,unit.id)
                self.enemy_alive_list[uid] = 1
                self.enemy_current_hp += unit.health


        # init the init_total_hp
        if self.init_myself_total_hp is None:
            self.init_myself_total_hp = self.myself_current_hp
        if self.init_enemy_total_hp is None:
            self.init_enemy_total_hp = self.enemy_current_hp


    # return done
    def _check_done(self):
        if int(sum(self.myself_alive_list)) == 0 or int(sum(self.enemy_alive_list)) == 0:
            if sum(self.myself_alive_list)>0 and sum(self.enemy_alive_list) == 0:
                self.win = True
                self.episode_wins += 1
            else:
                self.win = False
            return True
        if self.episode_steps >= self.max_episode_steps:
            print("max_episode_steps done!")
            self.advanced_termination = True
            return True
        return False

    # print info for debug
    def print_attr(self):

        print('self.state.units[0]:',len(self.state.units[0]))
        print('self.state.units[1]:',len(self.state.units[1]))

        # print('myself_alive_list:',self.myself_alive_list)
        # print('enemy_alive_list:',self.enemy_alive_list)
        # print('current_state',len(self.current_state['myself']),len(self.current_state['enemy']))
        
        myself_ids = []
        enemy_ids = []
        for unit in self.state.units[0]:
            myself_ids.append(unit.id)
        for unit in self.state.units[1]:
            enemy_ids.append(unit.id)

        # print('myself_unit_dict:',self.myself_unit_dict)
        # print('enemy_unit_dict:',self.enemy_unit_dict)

        print('myself_ids:',myself_ids)
        print('enemy_ids:',enemy_ids)



