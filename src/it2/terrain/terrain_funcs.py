"""
Two agents will play a chaser-evader game on various terrains to learn locomotion skills.
The terrains should incentivize diverse locomotion skills, such as walking, running, jumping, and climbing, and
discourage repetitive circling around.

Note that each terrain function generates a sub-terrain mesh. Add proper boundaries when necessary.
"""

import math
import trimesh
import numpy as np


def intersection_terrain(
    size: tuple[float, float],
    platform_width: float,
    platform_height: float,
    curb_heights: tuple[float, float, float, float],
    path_width: float,
) -> trimesh.Trimesh:
    """Create a terrain with a central platform and four connecting paths.

    The terrain is centered at the origin in the XY-plane. The central
    platform is axis-aligned, while the four paths are sloped ramps that
    connect the platform height down to ground level.

    - A square platform (box) of width ``platform_width`` is placed at the center.
    - Four rectangular paths (also boxes) of width ``path_width`` connect each side
      of the platform to the corresponding edge of the terrain bounding box.

    Args:
        size: Full terrain size in X and Y as ``(size_x, size_y)``.
        platform_width: Width of the central square platform (X and Y).
        platform_height: Height of the central square platform.
        curb_heights: Heights of the curbs for the four paths in the
            order (north, south, east, west). Each curb is a low wall
            placed across the middle of the corresponding path so that
            the robot must jump over it.
        path_width: Width of the connecting paths (their "thickness").

    Returns:
        A single `trimesh.Trimesh` representing the union of the platform and paths.
    """
    size_x, size_y = size

    # Basic sanity checks: platform and paths must fit inside the terrain,
    # and all lengths (including curb heights) must be non-negative.
    assert size_x > 0.0 and size_y > 0.0
    assert platform_width > 0.0 and path_width > 0.0
    assert platform_width < size_x and platform_width < size_y
    assert len(curb_heights) == 4
    north_curb_h, south_curb_h, east_curb_h, west_curb_h = curb_heights
    assert north_curb_h >= 0.0 and south_curb_h >= 0.0
    assert east_curb_h >= 0.0 and west_curb_h >= 0.0

    # Half extents for the terrain and central platform.
    half_x = size_x / 2.0
    half_y = size_y / 2.0
    half_platform = platform_width / 2.0

    # Thickness of the sloped paths and curbs.
    path_thickness = 0.2

    meshes: list[trimesh.Trimesh] = []

    # Central square platform. Its top surface is at z = platform_height and
    # its base sits on the ground (z = 0).
    platform = trimesh.creation.box(
        extents=[platform_width, platform_width, path_thickness]
    )
    platform.apply_translation([0.0, 0.0, platform_height - path_thickness / 2.0])
    meshes.append(platform)

    # Horizontal distance from the platform edge to terrain boundary.
    north_south_span = half_y - half_platform
    east_west_span = half_x - half_platform

    assert north_south_span > 0.0
    assert east_west_span > 0.0

    # Sloped ramp geometry: for a ramp that connects height `platform_height`
    # down to ground (0) over horizontal distance `span`, the ramp's length L
    # and tilt angle theta satisfy:
    #   L = sqrt(span^2 + platform_height^2)
    #   theta = atan2(platform_height, span)
    north_south_L = math.hypot(north_south_span, platform_height)
    north_south_theta = math.atan2(platform_height, north_south_span)

    east_west_L = math.hypot(east_west_span, platform_height)
    east_west_theta = math.atan2(platform_height, east_west_span)

    # North path (positive Y direction): high at the platform edge, low at the
    # outer edge of the terrain. The path is a box whose long axis is along Y
    # and which is rotated about the X axis.
    north_center_y = (half_platform + half_y) / 2.0
    north_center_z = platform_height / 2.0
    north = trimesh.creation.box(
        extents=[path_width, north_south_L, path_thickness]
    )
    # Negative angle so that the platform side (smaller Y) is high.
    rot_north = trimesh.transformations.rotation_matrix(
        -north_south_theta, [1.0, 0.0, 0.0]
    )
    north.apply_transform(rot_north)
    north.apply_translation([0.0, north_center_y, north_center_z - path_thickness / 2.0])
    meshes.append(north)

    # North curb (wall across the path).
    if north_curb_h > 0.0:
        curb_thickness = path_width * 0.2
        north_curb = trimesh.creation.box(
            extents=[path_width, curb_thickness, north_curb_h]
        )
        # Place the curb at the middle of the path in Y, sitting on top of the ground (z=0).
        north_curb_center_y = north_center_y
        north_curb_center_z = north_curb_h / 2.0
        north_curb.apply_translation([0.0, north_curb_center_y, north_curb_center_z])
        meshes.append(north_curb)

    # South path (negative Y direction).
    south_center_y = -(half_platform + half_y) / 2.0
    south_center_z = platform_height / 2.0
    south = trimesh.creation.box(
        extents=[path_width, north_south_L, path_thickness]
    )
    # Positive angle so that the platform side (larger Y in local coords,
    # mapped to more positive Z) is high.
    rot_south = trimesh.transformations.rotation_matrix(
        north_south_theta, [1.0, 0.0, 0.0]
    )
    south.apply_transform(rot_south)
    south.apply_translation([0.0, south_center_y, south_center_z - path_thickness / 2.0])
    meshes.append(south)

    # South curb (wall across the path).
    if south_curb_h > 0.0:
        curb_thickness = path_width * 0.2
        south_curb = trimesh.creation.box(
            extents=[path_width, curb_thickness, south_curb_h]
        )
        south_curb_center_y = south_center_y
        south_curb_center_z = south_curb_h / 2.0
        south_curb.apply_translation([0.0, south_curb_center_y, south_curb_center_z])
        meshes.append(south_curb)

    # East path (positive X direction).
    east_center_x = (half_platform + half_x) / 2.0
    east_center_z = platform_height / 2.0
    east = trimesh.creation.box(
        extents=[east_west_L, path_width, path_thickness]
    )
    # Negative angle about Y so that the platform side (smaller X) is high.
    rot_east = trimesh.transformations.rotation_matrix(
        east_west_theta, [0.0, 1.0, 0.0]
    )
    east.apply_transform(rot_east)
    east.apply_translation([east_center_x, 0.0, east_center_z - path_thickness / 2.0])
    meshes.append(east)

    # East curb (wall across the path).
    if east_curb_h > 0.0:
        curb_thickness = path_width * 0.2
        east_curb = trimesh.creation.box(
            extents=[curb_thickness, path_width, east_curb_h]
        )
        east_curb_center_x = east_center_x
        east_curb_center_z = east_curb_h / 2.0
        east_curb.apply_translation([east_curb_center_x, 0.0, east_curb_center_z])
        meshes.append(east_curb)

    # West path (negative X direction).
    west_center_x = -(half_platform + half_x) / 2.0
    west_center_z = platform_height / 2.0
    west = trimesh.creation.box(
        extents=[east_west_L, path_width, path_thickness]
    )
    # Positive angle about Y so that the platform side (larger X in local
    # coords, mapped to more positive Z) is high.
    rot_west = trimesh.transformations.rotation_matrix(
        -east_west_theta, [0.0, 1.0, 0.0]
    )
    west.apply_transform(rot_west)
    west.apply_translation([west_center_x, 0.0, west_center_z - path_thickness / 2.0])
    meshes.append(west)

    # West curb (wall across the path).
    if west_curb_h > 0.0:
        curb_thickness = path_width * 0.2
        west_curb = trimesh.creation.box(
            extents=[curb_thickness, path_width, west_curb_h]
        )
        west_curb_center_x = west_center_x
        west_curb_center_z = west_curb_h / 2.0
        west_curb.apply_translation([west_curb_center_x, 0.0, west_curb_center_z])
        meshes.append(west_curb)

    # Combine all pieces into a single mesh.
    terrain_mesh = trimesh.util.concatenate(meshes)
    terrain_mesh.merge_vertices()
    return terrain_mesh


def stair_flight_meshes(
    num_steps: int,
    step_height: float,
    step_depth: float,
    width: float,
    platform_depth: float,
) -> trimesh.Trimesh:
    """Box treads for one straight flight along **±X** (horizontal plane).

    The bottom of the first tread lies at ``z = z0``; each next tread rises by
    ``step_height``. Treads are centered on ``y = y_center`` with extent ``width``
    in **Y** and ``step_depth`` in **X**.

    Args:
        num_steps: Number of treads.
        step_height: Rise (extent along **Z**).
        step_depth: Run (extent along **X**).
        width: Tread width along **Y**.

    Returns:
        One ``trimesh.Trimesh`` box per tread (not concatenated).
    """
    assert num_steps >= 1
    assert step_height > 0.0 and step_depth > 0.0 and width > 0.0

    treads: list[trimesh.Trimesh] = []
    depths = [platform_depth] + (num_steps - 2) * [step_depth] + [platform_depth]
    xs = [0, platform_depth] + [step_depth] * (num_steps-1)
    xs = np.cumsum(xs)
    
    for j in range(num_steps):
        tread = trimesh.creation.box(
            extents=[depths[j], width, step_height]
        )
        tread.apply_translation([xs[j] + depths[j] / 2.0, 0.0, (j + 0.5) * step_height])
        treads.append(tread)
    return trimesh.util.concatenate(treads)


def landing_stairs_terrain(
    size: tuple[float, float],
    platform_width: float,
) -> trimesh.Trimesh:
    num_steps = 10
    step_height = 0.12
    step_depth = 0.35
    width = 2
    platform_depth = 2

    meshes: list[trimesh.Trimesh] = []
    total_length = (num_steps - 2) * step_depth + platform_depth * 2
    flight_1 = stair_flight_meshes(
        num_steps=num_steps,
        step_height=step_height,
        step_depth=step_depth,
        width=width,
        platform_depth=platform_depth,
    )
    flight_1.apply_translation([0.0, -width/2, 0.0])
    meshes.append(flight_1)

    flight_2 = stair_flight_meshes(
        num_steps=num_steps,
        step_height=step_height,
        step_depth=step_depth,
        width=width,
        platform_depth=platform_depth,
    )
    flight_2.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [0.0, 0.0, 1.0]))
    flight_2.apply_translation([total_length, width/2, num_steps * step_height])
    meshes.append(flight_2)

    ground = trimesh.creation.box(
        extents=[size[0], size[1], 0.2]
    )
    ground.apply_translation([0.0, 0.0, -0.1])
    meshes.append(ground)

    terrain_mesh = trimesh.util.concatenate(meshes)
    terrain_mesh.merge_vertices()
    return terrain_mesh


def curved_corridor_terrain(
    size: tuple[float, float],
    turn_radius: float,
    wall_height: float,
    wall_thickness: float,
) -> trimesh.Trimesh:
    """Build an S-shaped corridor from two mirrored **180° turn** wall pieces.

    Each piece is a thick **annular arc** (half of an annulus after a plane cut)
    with **straight end caps**, forming a smooth U-like bend. One turn sits in the
    ``+X`` half-space and the other is rotated and shifted into the ``-X``
    half-space so the inner channel links in an **S** (two opposing bends).

    A flat ground slab covers the full ``size`` footprint (top at ``z = 0``).

    Args:
        size: Full terrain size in X and Y as ``(size_x, size_y)`` (ground plane).
        turn_radius: Inner radius of each curved wall (annulus ``r_min``).
        wall_height: Vertical extent of the walls.
        wall_thickness: Radial thickness of the annulus and width of the end-cap boxes.

    Returns:
        One ``trimesh.Trimesh`` combining both turn meshes and the ground.
    """
    size_x, size_y = size
    meshes = []

    def turn(radius: float) -> trimesh.Trimesh:
        """Create a turn by cutting a cylinder."""
        mesh: trimesh.Trimesh = trimesh.creation.annulus(
            r_min=radius,
            r_max=radius + wall_thickness,
            height=wall_height,
            sections=12,
        )
        # cut out the half of it
        mesh = mesh.slice_plane([0., 0., 0.], [1., 0., 0.])
        wall_length = radius * 2.0
        # add walls at the ends
        wall_a = trimesh.creation.box(extents=[wall_length, wall_thickness, wall_height])
        wall_a.apply_translation([-0.5 * wall_length, radius + 0.5 * wall_thickness, 0.0])
        wall_b = trimesh.creation.box(extents=[wall_length, wall_thickness, wall_height])
        wall_b.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [0.0, 0.0, 1.0]))
        wall_b.apply_translation([-0.5 * wall_length, -radius - 0.5 * wall_thickness, 0.0])
        return trimesh.util.concatenate([mesh, wall_a, wall_b])
        
    turn_a = turn(turn_radius)
    turn_a.apply_translation([turn_radius, 0.5 * turn_radius, 0.5 * wall_height])
    turn_b = turn(turn_radius)
    turn_b.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [0.0, 0.0, 1.0]))
    turn_b.apply_translation([-turn_radius, -0.5 * turn_radius, 0.5 * wall_height])
    meshes.append(turn_a)
    meshes.append(turn_b)

    ground = trimesh.creation.box(extents=[size_x, size_y, 0.2])
    ground.apply_translation([0.0, 0.0, -0.1])
    meshes.append(ground)

    terrain_mesh = trimesh.util.concatenate(meshes)
    terrain_mesh.merge_vertices()
    return terrain_mesh


if __name__ == "__main__":
    # Example: 10x10 terrain with a 3x3 central platform of height 0.5, 2-wide
    # sloped paths, and 0.3-high curbs on all four paths.
    # mesh = intersection_terrain(
    #     (10.0, 10.0),
    #     platform_width=3.0,
    #     platform_height=0.0,
    #     curb_heights=(0.3, 0.3, 0.3, 0.3),
    #     path_width=2.0,
    # )
    # mesh.show()

    # Ridge example: 10x10 terrain, 4 m wide ridge peaking at 0.35 m.
    # mesh = ridge_corridor_terrain((10.0, 10.0), ridge_width=4.0, ridge_height=0.35)
    # mesh.show()

    # # Chicane + slalom: 14x10, five offset gates (2.2 m passage), hurdles between gates.
    # mesh = chicane_slalom_terrain(
    #     (14.0, 10.0),
    #     num_gates=5,
    #     passage_width=2.2,
    #     wall_depth=0.55,
    #     wall_height=0.9,
    # )
    # mesh.show()

    # Stairs: switchback (+X lower Y, −X upper Y); fixed step_run / stair_width.
    # mesh = landing_stairs_terrain(
    #     (10.0, 10.0),
    #     platform_width=3.0,
    # )
    # mesh.show()

    # Winding wall: 10x10 terrain, 0.35 m high wall 0.2 m thick.
    mesh = curved_corridor_terrain(
        (10.0, 10.0),
        turn_radius=2.4,
        wall_height=1.0,
        wall_thickness=0.1,
    )
    mesh.show()
