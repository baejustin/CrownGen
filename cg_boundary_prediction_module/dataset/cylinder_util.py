

import numpy as np
import trimesh
import os

def minimum_enclosing_circle(points_xy):
    """
    Returns (cx, cy, r) for the minimal bounding circle of 2D points.
    Uses a simple O(n^3) fallback method for clarity,
    or you can implement Welzl's algorithm for O(n) average.
    """
    # Shuffle points to improve average performance of the naive approach
    shuffled = np.random.permutation(points_xy)
    
    # Start with the trivial circle
    cx, cy = shuffled[0]
    r = 0.0
    
    # Progressively enlarge circle
    for i, p in enumerate(shuffled[1:], start=1):
        px, py = p
        # If p is not in the circle, update the circle
        if np.hypot(px - cx, py - cy) > r:
            # Circle must be defined by p and some other point
            cx, cy = px, py
            r = 0.0
            # Check all previous points
            for j in range(i):
                qx, qy = shuffled[j]
                if np.hypot(qx - cx, qy - cy) > r:
                    # Define circle passing through p and q
                    cx, cy, r = circle_from_2_points(p, shuffled[j])
                    # Check if any earlier points are outside circle
                    for k in range(j):
                        rx, ry = shuffled[k]
                        if np.hypot(rx - cx, ry - cy) > r:
                            # Define circle from p, q, r
                            cx, cy, r = circle_from_3_points(p, shuffled[j], shuffled[k])
    return cx, cy, r

def circle_from_2_points(a, b):
    """
    Return center and radius for circle passing through points a, b.
    """
    ax, ay = a
    bx, by = b
    cx = 0.5 * (ax + bx)
    cy = 0.5 * (ay + by)
    r = 0.5 * np.hypot(ax - bx, ay - by)
    return cx, cy, r

def circle_from_3_points(a, b, c):
    """
    Return center and radius for circle passing through non-collinear points a, b, c.
    """
    # Reference: standard circumcircle formula
    ax, ay = a
    bx, by = b
    cx, cy = c
    
    d = 2 * (ax*(by - cy) + bx*(cy - ay) + cx*(ay - by))
    # Handle nearly collinear points gracefully
    if abs(d) < 1e-14:
        return circle_from_2_points(a, b)

    ux = ((ax**2 + ay**2)*(by - cy) + (bx**2 + by**2)*(cy - ay) + (cx**2 + cy**2)*(ay - by))/d
    uy = ((ax**2 + ay**2)*(cx - bx) + (bx**2 + by**2)*(ax - cx) + (cx**2 + cy**2)*(bx - ax))/d
    r = np.hypot(ax - ux, ay - uy)
    return ux, uy, r


def bounding_cylinder_z_axis(points_3d):
    """
    Given an (N,3) array of points, return the 5 parameters of
    the bounding cylinder aligned with the z-axis.

    Returns (cx, cy, r, zmin, zmax)
    """
    # 1. Project onto XY plane
    points_xy = points_3d[:, :2]

    # 2. Compute minimal bounding circle
    cx, cy, r = minimum_enclosing_circle(points_xy)

    # 3. Compute z-min and z-max
    zmin = np.min(points_3d[:, 2])
    zmax = np.max(points_3d[:, 2])

    cz = (zmin+zmax)/2.0
    h = zmax-zmin

    return cx, cy, cz, h, r

def unnormalize_bound(pred_bound_arr, shift, scale):
    pred_bound_center = (pred_bound_arr[:3]*scale) + shift
    pred_bound_size = pred_bound_arr[3:]*scale

    return np.concatenate([pred_bound_center, pred_bound_size],1).astype(float)

def create_cylinder_trimesh(cx, cy, cz, h, r):
    """
    Create a trimesh cylinder aligned with the z-axis such that:
      - The cylinder center in xy is (cx, cy).
      - The radius is r.
      - The bottom is at zmin and the top is at zmax.
    """
    # Create a cylinder of given radius and height centered at origin
    cylinder_mesh = trimesh.primitives.Cylinder(
        radius=r,
        height=h,
        sections=32  # number of radial segments, adjust as needed
    )

    transform = np.eye(4)
    transform[0, 3] = cx
    transform[1, 3] = cy
    transform[2, 3] = cz
    
    # Apply the transformation
    cylinder_mesh.apply_transform(transform)
    
    return cylinder_mesh


def export_dentition(id, dir, points, shift, scale, combine_teeth=False):

    # points: (nT, 3, P)
    
    os.makedirs(dir, exist_ok=True)
    points = points.transpose(0,2,1)
    points = points * scale + shift  

    if combine_teeth or points.shape[0] == 1:
        points = points.reshape(-1,3)
        save_path = os.path.join(dir, '{}.npy'.format(id))
        np.save(save_path, points)
        trimesh.PointCloud(points).export(save_path.replace('.npy', '.ply'))
    else:
        for i in range(points.shape[0]):
            points_tooth = points[i].reshape(-1,3)
            save_path = os.path.join(dir, '{}_{}.npy'.format(id, i))
            np.save(save_path, points_tooth)
            trimesh.PointCloud(points_tooth).export(save_path.replace('.npy', '.ply'))
