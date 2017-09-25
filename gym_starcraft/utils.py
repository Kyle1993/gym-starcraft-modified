import math
import numpy as np

def total_reward(unit_dict_list):
    health_delta_list = []
    flag_list = []
    num_list = []
    for unit_dict in unit_dict_list:
        flag = unit_dict.flag
        total_delta_health = 0
        for i in unit_dict.id_list:
            t = unit_dict.units_dict[i]
            if t.die:
                continue
            total_delta_health += t.delta_health*100./unit.max_health
        health_delta_list.append(total_delta_health)
        flag_list.append(flag)
        num_list.append(unit_dict.num)
    health_delta_norm = np.array(health_delta_list)/(np.array(num_list) + 0.)
    reward = np.sum(health_delta_norm * (np.array(flag_list)*2 -1))
    return reward

def top_k_enemy_reward(k, unit, unit_dict):
    d_list = []
    health_delta_list = []
    for i in unit_dict.id_list:
        t = unit_dict.units_dict[i]
        if t.die:
            continue
        d = get_distance(unit.x, unit.y, t.x, t.y)
        health_delta_list.append(t.delta_health*100./unit.max_health)
        d_list.append(d)
    top_k_idxes = np.argsort(np.array(d_list))[:k]

    num_enemy = len(top_k_idxes)

    top_k_delta = np.array(health_delta_list)[top_k_idxes]
    enemy_delta_norm = 0.
    reward_added = 0
    if num_enemy:
        if np.array(d_list)[top_k_idxes][0] > 100:
            reward_added = -0.5
        enemy_delta_norm = np.sum(top_k_delta)/num_enemy
    myself_delta_norm = (unit.delta_health*100./unit.max_health)
    unit_reward  = enemy_delta_norm - myself_delta_norm + reward_added
    return unit_reward

# only consider two groups.
# no alliance.
# TODO in complicated scenes, need to judge whether one unit is alliance.
def unit_top_k_reward(k, unit, unit_dict_list):
    # f - highest distance.
    d_list = []
    flag_list = []
    health_delta_list = []
    for unit_dict in unit_dict_list:
        flag = unit_dict.flag
        for i in unit_dict.id_list:
            t = unit_dict.units_dict[i]
            if t.die:
                continue
            d = get_distance(unit.x, unit.y, t.x, t.y)
            health_delta_list.append(t.delta_health*100./unit.max_health)
            d_list.append(d)
            flag_list.append(flag)
    # the unit itself will be included as well. 0 distance
    top_k_idxes = np.argsort(np.array(d_list))[:k]
    top_k_flags = np.array(flag_list)[top_k_idxes]
    # enemy flag 1
    num_enemy = np.sum(top_k_flags) + 0.
    num_myself = len(top_k_idxes) - num_enemy + 0.
    top_k_delta = np.array(health_delta_list)[top_k_idxes]
    myself_delta_norm = 0
    enemy_delta_norm = 0

    if num_myself > 0:
        myself_delta_norm = np.sum(np.multiply(top_k_delta, 1-top_k_idxes))/num_myself
    if enemy_delta_norm > 0:
        enemy_delta_norm = np.sum(np.multiply(top_k_delta, top_k_idxes))/num_enemy

    unit_reward = enemy_delta_norm - myself_delta_norm
    return unit_reward


def get_degree(x1, y1, x2, y2):
    radians = math.atan2(y2 - y1, x2 - x1)
    return math.degrees(radians)


def get_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def get_position(degree, distance, x1, y1):
    theta = math.pi / 2 - math.radians(degree)
    return x1 + distance * math.sin(theta), y1 + distance * math.cos(theta)


# TODO change the action degree range
# I can't understand the negtive operation.
def get_position2(degree, distance, x1, y1):
    theta = math.radians(degree) # -180-180 not -1->1
    return x1 + distance * math.cos(theta), y1 + distance * math.sin(theta)



def print_progress(episodes, wins):
    print("Episodes: %4d | Wins: %4d | WinRate: %1.3f" % (
        episodes, wins, wins / (episodes + 1E-6)))

# modified from https://github.com/torchcraft/torchcraft/lua/utils.lua
def hsv_to_rgb(h, s, v):
    h = h / 360.
    s = s / 100.
    v = v / 100.

    i = math.floor(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    i = i % 6

    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    elif i == 5:
        r, g, b = v, p, q
    t = (math.floor(r * 255), math.floor(g * 255), math.floor(b * 255))
    return t

players_color_table = [
    (255, 0, 0),    # red
    (0, 0, 255),    # blue
    (0, 128, 128),  # teal
    (128, 0, 128),  # purple
    (255,165,0),    # orange
    (165,42,42),    # brown
    (255,255,255),  # white
    (255,255,0)     # Yellow
]

html_color_table = [
    [255,0,0],      # Red
    [0,255,0],      # Lime
    [0,0,255],      # Blue
    [255,255,0],    # Yellow
    [0,255,255],    # Cyan / Aqua
    [255,0,255],    # Magenta / Fuchsia
    [192,192,192],  # Silver
    [128,128,128],  # Gray
    [128,0,0],      # Maroon
    [128,128,0],    # Olive
    [0,128,0],      # Green
    [128,0,128],    # Purple
    [0,128,128],    # Teal
    [0,0,128],      # Navy
    [128,0,0],      # maroon
    [139,0,0],      # dark red
    [165,42,42],    # brown
    [178,34,34],    # firebrick
    [220,20,60],    # crimson
    [255,0,0],      # red
    [255,99,71],    # tomato
    [255,127,80],   # coral
    [205,92,92],    # indian red
    [240,128,128],  # light coral
    [233,150,122],  # dark salmon
    [250,128,114],  # salmon
    [255,160,122],  # light salmon
    [255,69,0],     # orange red
    [255,140,0],    # dark orange
    [255,165,0],    # orange
    [255,215,0],    # gold
    [184,134,11],   # dark golden rod
    [218,165,32],   # golden rod
    [238,232,170],  # pale golden rod
    [189,183,107],  # dark khaki
    [240,230,140],  # khaki
    [128,128,0],    # olive
    [255,255,0],    # yellow
    [154,205,50],   # yellow green
    [85,107,47],    # dark olive green
    [107,142,35],   # olive drab
    [124,252,0],    # lawn green
    [127,255,0],    # chart reuse
    [173,255,47],   # green yellow
    [0,100,0],      # dark green
    [0,128,0],      # green
    [34,139,34],    # forest green
    [0,255,0],      # lime
    [50,205,50],    # lime green
    [144,238,144],  # light green
    [152,251,152],  # pale green
    [143,188,143],  # dark sea green
    [0,250,154],    # medium spring green
    [0,255,127],    # spring green
    [46,139,87],    # sea green
    [102,205,170],  # medium aqua marine
    [60,179,113],   # medium sea green
    [32,178,170],   # light sea green
    [47,79,79],     # dark slate gray
    [0,128,128],    # teal
    [0,139,139],    # dark cyan
    [0,255,255],    # aqua
    [0,255,255],    # cyan
    [224,255,255],  # light cyan
    [0,206,209],    # dark turquoise
    [64,224,208],   # turquoise
    [72,209,204],   # medium turquoise
    [175,238,238],  # pale turquoise
    [127,255,212],  # aqua marine
    [176,224,230],  # powder blue
    [95,158,160],   # cadet blue
    [70,130,180],   # steel blue
    [100,149,237],  # corn flower blue
    [0,191,255],    # deep sky blue
    [30,144,255],   # dodger blue
    [173,216,230],  # light blue
    [135,206,235],  # sky blue
    [135,206,250],  # light sky blue
    [25,25,112],    # midnight blue
    [0,0,128],      # navy
    [0,0,139],      # dark blue
    [0,0,205],      # medium blue
    [0,0,255],      # blue
    [65,105,225],   # royal blue
    [138,43,226],   # blue violet
    [75,0,130],     # indigo
    [72,61,139],    # dark slate blue
    [106,90,205],   # slate blue
    [123,104,238],  # medium slate blue
    [147,112,219],  # medium purple
    [139,0,139],    # dark magenta
    [148,0,211],    # dark violet
    [153,50,204],   # dark orchid
    [186,85,211],   # medium orchid
    [128,0,128],    # purple
    [216,191,216],  # thistle
    [221,160,221],  # plum
    [238,130,238],  # violet
    [255,0,255],    # magenta / fuchsia
    [218,112,214],  # orchid
    [199,21,133],   # medium violet red
    [219,112,147],  # pale violet red
    [255,20,147],   # deep pink
    [255,105,180],  # hot pink
    [255,182,193],  # light pink
    [255,192,203],  # pink
    [250,235,215],  # antique white
    [245,245,220],  # beige
    [255,228,196],  # bisque
    [255,235,205],  # blanched almond
    [245,222,179],  # wheat
    [255,248,220],  # corn silk
    [255,250,205],  # lemon chiffon
    [250,250,210],  # light golden rod yellow
    [255,255,224],  # light yellow
    [139,69,19]     # saddle brownH
]

map_channels_table = {
    "unit_density": 1,
    "unit_location": 1,
    "unit_data": 0,
    "health": 3,
    "shield": 3,
    "type": 3,
    "flag": 3
}

obs_dtype = {
    'ul': "uint8",
    'ud':"float32",
    's': "uint8",
    'mask':"uint8",
    'au': "int32"
}

obs_cls_table = {
    "unit_location": "ul",
    "unit_density": "s",
    "unit_data": "ud",
    "health": "s",
    "shield": "s",
    "type": "s",
    "flag": "s"
}
