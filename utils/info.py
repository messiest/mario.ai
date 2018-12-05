PLAYER_STATES = {
    0x00 : 'Leftmost of screen',
    0x01 : 'Climbing vine',
    0x02 : 'Entering reversed-L pipe',
    0x03 : 'Going down a pipe',
    0x04 : 'Auto-walk',
    0x05 : 'Auto-walk',
    0x06 : 'Dead',
    0x07 : 'Entering area',
    0x08 : 'Normal',
    0x09 : 'Cannot move',
    0x0B : 'Dying',
    0x0C : "Palette cycling, can't move",
}


def decode_info(env):
    env = env.unwrapped
    info = {
        'level': env._level,
        'world': env._world,
        'stage': env._stage,
        'area': env._area,
        'score': env._score,
        'time': env._time,
        'coins': env._coins,
        'life': env._life,
        'x_position': env._x_position,
        'left_x_position': env._left_x_position,
        'y_position': env._y_position,
        'y_viewport': env._y_viewport,
        'player_status': env._player_status,
        'player_state': PLAYER_STATES[env._player_state],
        'is_dying': env._is_dying,
        'is_dead': env._is_dead,
        'is_game_over': env._is_game_over,
        'is_busy': env._is_busy,
        'is_world_over': env._is_world_over,
        'is_stage_over': env._is_stage_over,
        'flag_get': env._flag_get,
    }

    return info
