from isaaclab.terrains import (
    TerrainImporterCfg,
    HfTerrainBaseCfg,
    HfRandomUniformTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfInvertedPyramidSlopedTerrainCfg,
    TerrainGeneratorCfg,
    MeshPlaneTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfInvertedPyramidStairsTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    MeshRandomGridTerrainCfg,
    HfDiscreteObstaclesTerrainCfg,
    MeshRepeatedBoxesTerrainCfg,
    MeshGapTerrainCfg,
    MeshPitTerrainCfg,
    MeshRailsTerrainCfg,
    MeshFloatingRingTerrainCfg,
    MeshStarTerrainCfg,
    MeshRepeatedCylindersTerrainCfg,
    MeshBoxTerrainCfg,
    height_field,
    FlatPatchSamplingCfg
)
from dataclasses import MISSING
from active_adaptation import ROBOT_MODEL_DIR
from active_adaptation.envs.terrain import BetterTerrainImporter, BetterTerrainGenerator

import isaaclab.sim as sim_utils

ROUGH_GAME = TerrainGeneratorCfg(
    class_type=BetterTerrainGenerator,
    seed=0,
    size=(10.0, 10.0),
    border_width=65.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(
            proportion=0.20,
        ),
        "gap": MeshGapTerrainCfg(
            proportion=0.20,
            gap_width_range=(0.1, 0.4),
            platform_width=6.0,
        ),
        "ring": MeshFloatingRingTerrainCfg(
            proportion=0.10,
            ring_height_range=(0.0, 0.1),
            ring_width_range=(0.3, 0.5),
            ring_thickness=0.2,
            platform_width=6.0,
        ),
        "star": MeshStarTerrainCfg(
            proportion=0.20,
            num_bars=3,
            bar_width_range=(0.8, 1.2),
            bar_height_range=(0.2, 0.8),
            platform_width=6.0,
        ),
        "pit": MeshPitTerrainCfg(
            proportion=0.10,
            pit_depth_range=(0.1, 0.2),
            platform_width=6.0,
        ),
        "grid": MeshRandomGridTerrainCfg(
            proportion=0.20,
            grid_width=0.45,
            grid_height_range=(0.02, 0.05),
            platform_width=2.0,
        )
    },
)

# add stairs terrain
ROUGH_GAME_323 = TerrainGeneratorCfg(
    class_type=BetterTerrainGenerator,
    seed=0,
    size=(10.0, 10.0),
    border_width=65.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(
            proportion=0.10,
        ),
        "gap": MeshGapTerrainCfg(
            proportion=0.15,
            gap_width_range=(0.1, 0.4),
            platform_width=6.0,
        ),
        "ring": MeshFloatingRingTerrainCfg(
            proportion=0.15,
            ring_height_range=(0.0, 0.1),
            ring_width_range=(0.3, 0.5),
            ring_thickness=0.2,
            platform_width=6.0,
        ),
        "star": MeshStarTerrainCfg(
            proportion=0.20,
            num_bars=3,
            bar_width_range=(0.8, 1.2),
            bar_height_range=(0.2, 0.8),
            platform_width=6.0,
        ),
        "pit": MeshPitTerrainCfg(
            proportion=0.10,
            pit_depth_range=(0.1, 0.2),
            platform_width=6.0,
        ),
        "grid": MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.45,
            grid_height_range=(0.02, 0.05),
            platform_width=2.0,
        ),
        "stairs": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.20),
            step_width=0.35,
            platform_width=3.5,
            holes=False,
        ),
    },
)

ROUGH_GAME_410 = TerrainGeneratorCfg(
    class_type=BetterTerrainGenerator,
    seed=0,
    size=(10.0, 10.0),
    border_width=65.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(
            proportion=0.10,
        ),
        "gap": MeshGapTerrainCfg(
            proportion=0.15,
            gap_width_range=(0.1, 0.4),
            platform_width=6.0,
        ),
        "ring": MeshFloatingRingTerrainCfg(
            proportion=0.10,
            ring_height_range=(0.0, 0.1),
            ring_width_range=(0.3, 0.5),
            ring_thickness=0.2,
            platform_width=6.0,
        ),
        "star": MeshStarTerrainCfg(
            proportion=0.15,
            num_bars=3,
            bar_width_range=(0.8, 1.2),
            bar_height_range=(0.2, 0.8),
            platform_width=6.0,
        ),
        "pit": MeshPitTerrainCfg(
            proportion=0.10,
            pit_depth_range=(0.1, 0.2),
            platform_width=6.0,
        ),
        "grid": MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.45,
            grid_height_range=(0.02, 0.05),
            platform_width=2.0,
        ),
        "stairs": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.20),
            step_width=0.35,
            platform_width=3.5,
            holes=False,
        ),
        "stairs_holes": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.20),
            step_width=0.35,
            platform_width=3,
            holes=True,
        ),
        "cylinders": MeshRepeatedCylindersTerrainCfg(
            proportion=0.10,
            object_params_start=MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=5,
                height=0.2,
                radius=0.1,
                max_yx_angle=0.0,
            ),
            object_params_end=MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=20,
                height=1.0,
                radius=0.3,
                max_yx_angle=30.0,
            ),
            abs_height_noise=(0.0, 1.0),
            platform_width=3.5,
            platform_height=0.1,
        ),
    },
)

ROUGH_TERRAIN_BASE_CFG = TerrainImporterCfg(
    class_type=BetterTerrainImporter,
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=MISSING,
    max_init_terrain_level=None,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ROBOT_MODEL_DIR}/scene/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
    ),
    debug_vis=False,
)

from active_adaptation.registry import Registry

registry = Registry.instance()
registry.register("terrain", "game", ROUGH_TERRAIN_BASE_CFG.replace(terrain_generator=ROUGH_GAME))
registry.register("terrain", "game_323", ROUGH_TERRAIN_BASE_CFG.replace(terrain_generator=ROUGH_GAME_323))
registry.register("terrain", "game_410", ROUGH_TERRAIN_BASE_CFG.replace(terrain_generator=ROUGH_GAME_410))
