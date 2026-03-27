import active_adaptation as aa

if aa.get_backend() == "isaac":
    from . import terrain_configs
