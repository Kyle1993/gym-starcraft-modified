import numpy as np
import math

from gym import spaces
import torchcraft.Constants as tcc
# import gym_starcraft.utils as utils
import gym_starcraft.envs.starcraft_env as sc


def get_position(degree, distance, x1, y1):
    theta = math.pi / 2 - math.radians(degree)
    return x1 + distance * math.sin(theta), y1 + distance * math.cos(theta)

class Unit_State(object):
    def __init__(self, unit):
        # print unit.groundRange, "Range"
        self.id = unit.id
        self.health = unit.health
        self.x = unit.x
        self.y = unit.y
        self.shield = unit.shield
        self.attackCD = unit.groundCD
        self.velocityX = unit.velocityX  # speed
        self.velocityY = unit.velocityY
        self.type = unit.type
        self.groundATK = unit.groundATK
        self.groundRange = unit.groundRange  # this can be inferred from training
        self.under_attack = unit.under_attack  # True or Flase
        self.attacking = unit.attacking  # True or False
        self.moving = unit.moving  # True or False
        self.die = (unit.health <= 0)
        self.max_health = unit.max_health
        self.max_shield = unit.max_shield
        self.pixel_size_x = unit.pixel_size_x
        self.pixel_size_y = unit.pixel_size_y
        self.delta_health = 0
        self.delta_shield = 0

    def update(self, unit):
        self.delta_health = self.health - unit.health
        self.health = unit.health
        self.delta_shield = self.shield - unit.shield
        self.shield = unit.shield
        self.x = unit.x
        self.y = unit.y
        self.attackCD = unit.groundCD
        self.velocityX = unit.velocityX  # speed
        self.velocityY = unit.velocityY
        self.groundATK = unit.groundATK
        self.groundRange = unit.groundRange  # this can be inferred from training
        self.under_attack = unit.under_attack  # True or Flase
        self.attacking = unit.attacking  # True or False
        self.moving = unit.moving  # True or False
        self.die = (unit.health <= 0)
        self.pixel_size_x = unit.pixel_size_x
        self.pixel_size_y = unit.pixel_size_y

    def set_die(self):
        self.die = True
        self.delta_health = self.health  # if die, detal health = how much health it has before
        self.health = 0
        self.shield = 0


class SimpleBattleEnv(sc.StarCraftEnv):
    def __init__(self, server_ip, server_port, MYSELF_NUM, ENEMY_NUM, ACTION_NUM=3, DISTANCE_FACTOR=10,
                 POSITION_RANGE=400, SCREEN_BOX=((40, 105), (150, 170)),
                 DIE_REWARD=2e-3, HEALTH_REWARD_WEIGHT=4, WIN_REWARD_WEIGHT=1, MY_HEALTH_WEIGHT=1,
                 ENEMY_HEALTH_WEIGHT=4, FRAME_SKIP=2, MAX_STEP=700, speed=0, self_play=False):
        super(SimpleBattleEnv, self).__init__(server_ip, server_port, speed, FRAME_SKIP, self_play, MAX_STEP)
        self.mapname = None
        # basic config
        self.MYSELF_NUM = MYSELF_NUM
        self.ENEMY_NUM = ENEMY_NUM
        self.ACTION_NUM = ACTION_NUM

        # env config
        self.DISTANCE_FACTOR = DISTANCE_FACTOR
        self.SCREEN_BOX = SCREEN_BOX  # (left_top,rigth_down)
        self.POSITION_RANGE = POSITION_RANGE

        # reward compute
        self.DIE_REWARD = DIE_REWARD
        self.HEALTH_REWARD_WEIGHT = HEALTH_REWARD_WEIGHT
        self.WIN_REWARD_WEIGHT = WIN_REWARD_WEIGHT
        self.MY_HEALTH_WEIGHT = MY_HEALTH_WEIGHT
        self.ENEMY_HEALTH_WEIGHT = ENEMY_HEALTH_WEIGHT

        # self.myself_alive_list = None  # like [1,1,0,0,1],init by first state
        # self.enemy_alive_list = None

        self.current_state = [{},{}]  # current_state['myself']: units state of myself; current_state['enemy']:units of enemy

        self.myself_total_hp = 0  # current myself_units total hp
        self.myself_total_shield = 0
        self.enemy_total_hp = 0
        self.enemy_total_shield = 0
        self.myself_detal_hp = 0
        self.myself_detal_shield = 0
        self.enemy_detal_hp = 0
        self.enemy_detal_shield = 0

        self.myself_alive = {}
        self.enemy_alive = {}
        self.myself_id = []
        self.enemy_id = []

        self.win = False  # in this episode

        # self.myself_unit_dict = None  # map between myself unit.id and index of self.current_state['myself']
        # self.enemy_unit_dict = None


        print('------------- Enveriment config ---------------')
        print('ip:', server_ip)
        print('port:', server_port)
        print('MYSELF_NUM:', self.MYSELF_NUM)
        print('ENEMY_NUM:', self.ENEMY_NUM)

    def _action_space(self):
        # attack or move, move_degree, move_distance
        action_low = [-1.0, -1.0, -1.0]
        action_high = [1.0, 1.0, 1.0]
        return spaces.Box(np.array(action_low), np.array(action_high))

    # return the current_state, have no space, you can extract the state you want
    def _observation_space(self):
        # print('return the current_state, have no space, you can extract the state you want')
        return spaces.Box(np.array([1.]), np.array([1.]))

    def _make_commands(self, actions):
        cmds = []
        if self.state is None or actions is None:
            return cmds

        # assert ((len(actions) == self.MYSELF_NUM) and (len(actions[0]) == self.ACTION_NUM))
        # print(actions.shape,(self.MYSELF_NUM,self.ACTION_NUM))
        assert actions.shape == (self.MYSELF_NUM,self.ACTION_NUM)

        for index,uid in enumerate(self.myself_id):
            unit = self.current_state[0][uid]

            if not unit.die:  # if not die
                if actions[index][0] >= 0:
                    # attack
                    enemy_id = self.get_closest_enemy(unit, actions[index])  # get the colse enemy of target position
                    if enemy_id is not None:
                        cmd = [tcc.command_unit_protected, unit.id, tcc.unitcommandtypes.Attack_Unit, enemy_id]
                    else:
                        # don't do any thing
                        cmd = [tcc.command_unit_protected, unit.id, tcc.unitcommandtypes.Move, -1, unit.x,
                               unit.y]

                        # # attack around
                        # continue

                        # move to target position
                        # x_target, y_target = unit.x + actions[index][1] * self.DISTANCE_FACTOR, unit.y + actions[index][
                        #     2] * self.DISTANCE_FACTOR
                        # x_target, y_target = max(x_target, self.SCREEN_BOX[0][0]), max(y_target, self.SCREEN_BOX[0][1])
                        # x_target, y_target = min(x_target, self.SCREEN_BOX[1][0]), min(y_target, self.SCREEN_BOX[1][1])
                        # cmd = [tcc.command_unit_protected, unit.id, tcc.unitcommandtypes.Move, -1, int(x_target),
                        #        int(y_target)]
                else:
                    # move
                    # degree = actions[index][1] * 180
                    # distance = (actions[index][2]+1) * self.DISTANCE_FACTOR
                    # x_target,y_target = get_position(degree,distance,unit.x,unit.y)
                    x_target,y_target = unit.x+actions[index][1]*self.DISTANCE_FACTOR,unit.y+actions[index][2]*self.DISTANCE_FACTOR

                    x_target, y_target = max(x_target, self.SCREEN_BOX[0][0]), max(y_target, self.SCREEN_BOX[0][1])
                    x_target, y_target = min(x_target, self.SCREEN_BOX[1][0]), min(y_target, self.SCREEN_BOX[1][1])
                    cmd = [tcc.command_unit_protected, unit.id, tcc.unitcommandtypes.Move, -1, int(x_target),
                           int(y_target)]

                cmds.append(cmd)

        if len(cmds) == 0:
            return [[tcc.noop]]

        return cmds

    def _make_observation(self):

        self.update_self()  # mainly in self.current_state, which is main state, we can extract all info from it
        obs = self._compute_obs()  # return the obs you want

        # self.print_attr()

        return obs

    # return the obs you want, compute by self.current_state
    # but in order to make the env stable, it is recommanded to compute in you own code
    def _compute_obs(self):
        state = {'myself':[],'enemy':[]}

        # # only return alive agent of mine
        # for uid in self.myself_id:
        #     if self.myself_alive[uid] == 1:
        #         state['myself'].append(self.current_state[0][uid])

        # return all unit of mine
        for uid in self.myself_id:
            state['myself'].append(self.current_state[0][uid])

        for uid in self.enemy_id:
            state['enemy'].append(self.current_state[1][uid])

        assert len(state['myself']) == self.MYSELF_NUM
        assert len(state['enemy']) == self.ENEMY_NUM

        return state

    # provide two options,total_reward or separate_reward, you can even calculate by yourself
    def _compute_reward(self):
        rewards = self.compute_reward_separately()
        reward = []
        for uid in self.myself_id:
            reward.append(rewards[uid])
        return reward


    def compute_reward_separately(self):

        rewards = {}
        alive = 0
        for ua in self.myself_alive.values():
            alive += ua

        for uid,unit in self.current_state[0].items():
            if unit.die:
                rewards[uid] = self.DIE_REWARD
            else:
                health_reward = self.ENEMY_HEALTH_WEIGHT*(self.enemy_detal_hp+self.enemy_detal_shield) - self.MY_HEALTH_WEIGHT*(unit.delta_health+unit.delta_shield)
                win_reward = self.win * float(alive)
                range_reward = -1*self.DIE_REWARD*self.in_saferange(unit)
                rewards[uid] = self.HEALTH_REWARD_WEIGHT * health_reward + self.WIN_REWARD_WEIGHT * win_reward + range_reward

        return rewards


    def reset_data(self):
        # skip the init state, there is no unit at the beginning of game
        while len(self.state.units) == 0 or len(self.state.units[0]) != self.MYSELF_NUM or len(
                self.state.units[1]) != self.ENEMY_NUM:
            self.client.send([])
            self.state = self.client.recv()

        # reset the variable
        self.current_state = [{},{}]  # current_state['myself']: units state of myself; current_state['enemy']:units of enemy

        self.myself_total_hp = 0  # current myself_units total hp
        self.myself_total_shield = 0
        self.enemy_total_hp = 0
        self.enemy_total_shield = 0
        self.myself_detal_hp = 0
        self.myself_detal_shield = 0
        self.enemy_detal_hp = 0
        self.enemy_detal_shield = 0
        self.myself_alive = {}
        self.enemy_alive = {}
        self.myself_id = []
        self.enemy_id = []
        self.win = False

        for unit in self.state.units[0]:
            self.myself_id.append(unit.id)
            self.myself_alive[unit.id] = 1
            self.current_state[0][unit.id] = Unit_State(unit)
            self.myself_total_hp += unit.health
            self.myself_total_shield += unit.shield

        for unit in self.state.units[1]:
            self.enemy_id.append(unit.id)
            self.enemy_alive[unit.id] = 1
            self.current_state[1][unit.id] = Unit_State(unit)
            self.enemy_total_hp += unit.health
            self.enemy_total_shield += unit.shield

        self.init_myself_total_hp = self.myself_total_hp
        self.init_myself_total_shield = self.myself_total_shield
        self.init_enemy_total_hp = self.enemy_total_hp
        self.init_enemy_total_shield = self.enemy_total_shield

        self.advanced_termination = True


    # get the closest enemy from out attack position, return unit.id
    def get_closest_enemy(self, unit, action):
        # degree = action[1] * 180
        # distance = (action[2] + 1) * self.DISTANCE_FACTOR
        # tx, ty = get_position(degree,distance,unit.x,unit.y)
        tx,ty = unit.x + action[1]*self.DISTANCE_FACTOR,unit.y + action[2]*self.DISTANCE_FACTOR
        target_id = self.compute_candidate(tx, ty)
        return target_id


    def compute_candidate(self, tx, ty):  # find the most nearly enemy unit in the POSITION_RANGE of target position
        target_id = None
        d = self.POSITION_RANGE
        for enemy_unit in self.state.units[1]:
            td = math.sqrt((enemy_unit.x - tx) ** 2 + (enemy_unit.y - ty) ** 2)
            if td < d:
                target_id = enemy_unit.id
                d = td
        return target_id

    def in_saferange(self,myself_unit,saferange=10):
        x,y = myself_unit.x,myself_unit.y
        min_d = saferange
        for uid,enemy in self.current_state[1].items():
            d = math.sqrt((x-enemy.x)**2 + (y-enemy.y)**2)
            if d < min_d:
                min_d = d
        return 1-min_d/saferange

    def update_self(self):

        myself_current_hp = 0
        enemy_current_hp = 0
        myself_current_shield = 0
        enemy_current_shield = 0

        self.mapname = self.state.map_name

        myslef_alive = []
        enemy_alive = []
        for unit in self.state.units[0]:
            myslef_alive.append(unit.id)
            self.current_state[0][unit.id].update(unit)
            myself_current_hp += unit.health
            myself_current_shield += unit.shield

        for unit in self.state.units[1]:
            enemy_alive.append(unit.id)
            self.current_state[1][unit.id].update(unit)
            enemy_current_hp += unit.health
            enemy_current_shield += unit.shield

        for k in self.myself_alive.keys():
            if not (k in myslef_alive):
                self.myself_alive[k] = 0
                self.current_state[0][k].set_die()
        for k in self.enemy_alive.keys():
            if not (k in enemy_alive):
                self.enemy_alive[k] = 0
                self.current_state[1][k].set_die()

        self.myself_detal_hp = max(0,self.myself_total_hp - myself_current_hp)
        self.myself_detal_shield = max(0,self.myself_total_shield - myself_current_shield)
        self.enemy_detal_hp = max(0,self.enemy_total_hp - enemy_current_hp)
        self.enemy_detal_shield = max(0,self.enemy_total_shield - enemy_current_shield)

        self.myself_total_hp = myself_current_hp
        self.myself_total_shield = myself_current_shield
        self.enemy_total_hp = enemy_current_hp
        self.enemy_total_shield = enemy_current_shield


        assert len(self.current_state[0]) == self.MYSELF_NUM
        assert len(self.current_state[1]) == self.ENEMY_NUM

    # return done
    def _check_done(self):
        # if int(sum(self.myself_alive_list)) == 0 or int(sum(self.enemy_alive_list)) == 0:
        if len(self.state.units[0]) == 0 or len(self.state.units[1]) == 0:
            if len(self.state.units[0]) > 0 and len(self.state.units[1]) == 0:
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

        myself_info = []
        enemy_info = []
        for unit in self.state.units[0]:
            myself_info.append(unit.health)
        for unit in self.state.units[1]:
            enemy_info.append(unit.health)
        print(myself_info)
        print(enemy_info)

        alive_myselfinfo = []
        for unit in self.current_state[0].values():
            alive_myselfinfo.append(unit.max_health)
        print(alive_myselfinfo)

    def getMapName(self):

        return self.mapname


