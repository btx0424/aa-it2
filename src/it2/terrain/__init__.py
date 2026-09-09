import active_adaptation as aa

if aa.get_backend() in ("isaac", "isaaclab"):
    from . import terrain_configs
