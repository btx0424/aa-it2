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
    height_field,
    FlatPatchSamplingCfg
)
from dataclasses import MISSING
from active_adaptation.envs.terrain import BetterTerrainImporter, BetterTerrainGenerator

import isaaclab.sim as sim_utils

ROUGH_GAME = TerrainGeneratorCfg(
    class_type=BetterTerrainGenerator,
    seed=0,
    size=(12.0, 12.0),
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
            proportion=0.20,
            ring_height_range=(0.1, 0.2),
            ring_width_range=(0.3, 0.5),
            ring_thickness=0.4,
            platform_width=6.0,
        ),
        "star": MeshStarTerrainCfg(
            proportion=0.20,
            num_bars=3,
            bar_width_range=(0.8, 1.2),
            bar_height_range=(0.2, 0.8),
            platform_width=6.0,
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
    # visual_material=sim_utils.MdlFileCfg(
    #     mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
    #     project_uvw=True,
    # ),
    debug_vis=False,
)

from active_adaptation.registry import Registry

registry = Registry.instance()
registry.register("terrain", "game", ROUGH_TERRAIN_BASE_CFG.replace(terrain_generator=ROUGH_GAME))
