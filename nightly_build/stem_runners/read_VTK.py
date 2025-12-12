import os
import json

import numpy as np
from matplotlib.path import Path
from scipy.spatial import KDTree
from scipy.interpolate import LinearNDInterpolator


def read_vtk_local_interpolated(file, coordinate_points, data_name, is_vector_data, k_neighbors=12):
    """
    Read a VTK file and estimate data at specified coordinate points
    using KDTree to find local neighbors, followed by local linear interpolation.

    Parameters:
    file (str): Path to the VTK file.
    coordinate_points (list): List of coordinate points to interpolate data at.
    data_name (str): Name of the data to extract.
    k_neighbors (int): Number of neighbors to use for local interpolation (default 12 to ensure 3D hull).

    Returns:
    tuple: A tuple containing the list of target coordinates and the
           corresponding interpolated data (list of lists).
    """

    # --- 1. FAST FILE READING ---
    with open(file, 'r') as f:
        lines = f.read().splitlines()

    # Parse Coordinates
    idx_points = [i for i, line in enumerate(lines) if line.startswith("POINTS")][0]
    nb_points = int(lines[idx_points].split()[1])

    # Vectorized read of coordinates
    coord_lines = ' '.join(lines[idx_points + 1:idx_points + 1 + nb_points])
    all_coordinates = np.fromstring(coord_lines, sep=' ').reshape((nb_points, -1))

    # Parse Elements
    idx_cells = [i for i, line in enumerate(lines) if line.startswith("CELLS")][0]
    nb_cells = int(lines[idx_cells].split()[1])

    # Vectorized read of cells
    cell_lines = ' '.join(lines[idx_cells + 1:idx_cells + 1 + nb_cells])
    all_cells = np.fromstring(cell_lines, sep=' ').reshape((nb_cells, -1))
    cell_type = all_cells[0, 0]
    all_cells = all_cells[:, 1:]

    # Parse Data
    idx_data = [i for i, line in enumerate(lines) if line.startswith(data_name.upper())][0]
    data_format = lines[idx_data].split()

    # Determine components (Scalar vs Vector)
    nb_data_components = 3  # Default to vector
    if 'SCALARS' in data_format or (len(data_format) > 2 and data_format[2] == '1'):
        nb_data_components = 1
        size_ = nb_points
    if 'CAUCHY_STRESS_VECTOR' in data_format:
        nb_data_components = int(data_format[1])
        size_ = nb_cells

    # Vectorized read of data
    idx_data_start = idx_data + 1
    data_lines = lines[idx_data_start:idx_data_start + size_]
    all_data_str = ' '.join(data_lines)
    all_data = np.fromstring(all_data_str, sep=' ').reshape((size_, nb_data_components))

    # Convert targets to numpy
    target_points = np.array(coordinate_points)

    # --- 3. LOCAL INTERPOLATION LOOP ---
    final_results = []

    # Build tree once O(N log N)
    tree = KDTree(all_coordinates)
    _, indices = tree.query(target_points, k=k_neighbors)
    if is_vector_data:
        # Iterate over each target point and its specific neighbors
        for i, target_point in enumerate(target_points):

            # Get local cluster of points
            local_idx = indices[i]
            local_coords = all_coordinates[local_idx]
            local_values = all_data[local_idx]

            # --- CRITICAL STABILITY FIX: JITTER ---
            # Add tiny noise (1e-10) to prevent QHull errors if points are flat/coplanar.
            # This is required because KDTree often picks points on a single face/plane.
            jitter = np.random.normal(scale=1e-10, size=local_coords.shape)
            local_coords_jittered = local_coords + jitter

            try:
                # Create ONE interpolator for all data components at once
                # LinearNDInterpolator handles multidimensional values (N, 3) natively
                lip = LinearNDInterpolator(local_coords_jittered, local_values)

                # Interpolate
                result = lip(target_point)[0]  # returns shape (nb_components,)

                # Handle case where target is outside the convex hull of neighbors (returns NaN)
                if np.any(np.isnan(result)):
                    # Fallback: Nearest Neighbor (average of 3 closest)
                    result = np.mean(local_values[:3], axis=0)

            except Exception:
                # Fallback if triangulation completely fails
                result = np.mean(local_values[:3], axis=0)

            final_results.append(result)
    else:
        # find element containing each target point
        for i, target_point in enumerate(target_points):

            local_idx = indices[i]
            # Find elements that contain any of the nearby nodes
            idx_elements_containing_node = [np.where(np.any(all_cells == n, axis=1))[0] for n in local_idx]
            # flatten and unique
            idx_elements_containing_node = np.unique(np.concatenate(idx_elements_containing_node))

            found_element = None
            for el_idx in idx_elements_containing_node:
                if point_in_element_coord(all_cells[el_idx], all_coordinates, target_point, cell_type):
                    found_element = el_idx
                    break

            if found_element is not None:
                # Get the data for this element (for cell-based data like stress)
                final_results.append(all_data[found_element])
            else:
                # Fallback: use nearest element's data
                final_results.append(all_data[idx_elements_containing_node[0]])

    return target_points.tolist(), np.array(final_results).tolist()


def point_in_element_coord(element_nodes, all_coordinates, target_point, cell_type):
    """
    Check if target_point is inside the element defined by element_nodes.

    Parameters:
    element_nodes: array of node indices (0-based) for this element
    all_coordinates: array of all node coordinates
    target_point: the point to check
    cell_type: VTK cell type

    Returns:
    bool: True if point is inside element, False otherwise
    """
    coords = np.array([all_coordinates[int(i)] for i in element_nodes])
    return point_in_element(target_point, coords)


def point_in_element(point, element):
    point = np.asarray(point)
    element = np.asarray(element)

    # --- Case 1: Element is 2D in 3D (all Z, or Y or X coordinates equal) ---
    # Detect if vertices lie in a plane
    if element.shape[1] == 3:
        v0 = element[1] - element[0]
        v1 = element[2] - element[0]
        v2 = element[3] - element[0]
        normal = np.cross(v0, v1)
        if np.linalg.norm(normal) < 1e-8:
            v2 = element[3] - element[0]
            normal = np.cross(v0, v2)
            # raise ValueError("Degenerate polygon")

        # Check if all points satisfy the plane equation
        d = -np.dot(normal, element[0])
        distances = np.dot(element, normal) + d

        if np.all(np.abs(distances) < 1e-6):
            # Element is planar → project to 2D
            return _point_in_planar_polygon(point, element, normal)

    # --- Case 2: Convex 3D polyhedron ---
    return _point_in_convex_polyhedron(point, element)


def _point_in_planar_polygon(point, element, normal):
    """
    Projects polygon and point to 2D coordinate system and checks inclusion.
    """
    # Make a local coordinate system in plane
    normal = normal / np.linalg.norm(normal)
    v1 = element[1] - element[0]
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)

    # Projection (3D → 2D local coords)
    def proj(p):
        return np.array([np.dot(p - element[0], v1), np.dot(p - element[0], v2)])

    poly_2d = np.array([proj(v) for v in element])
    p_2d = proj(point)

    return Path(poly_2d).contains_point(p_2d)


def _point_in_convex_polyhedron(point, element):
    """
    Uses half-space test. Assumes convexity.
    """
    # Create faces by triangulating around centroid
    centroid = np.mean(element, axis=0)

    for i in range(len(element)):
        a = element[i]
        b = element[(i + 1) % len(element)]
        c = centroid

        normal = np.cross(b - a, c - a)
        if np.dot(point - a, normal) > 1e-9:
            return False

    return True


def read_vtk_nearest_neighbour(file, coordinate_points, data_name):
    """
    Read a VTK file and extract data at specified coordinate points using KDTree for speed.

    Parameters:
    file (str): Path to the VTK file.
    coordinate_points (list): List of coordinate points to extract data from.
    data_name (str): Name of the data to extract (e.g., "DISPLACEMENT", "VELOCITY", "ACCELERATION").

    Returns:
    tuple: A tuple containing the list of coordinates and the corresponding extracted data.
    """

    # 1. READ FILE CONTENT EFFICIENTLY
    with open(file, 'r') as f:
        lines = f.read().splitlines()

    # Find the coordinates section
    idx_points = [i for i, line in enumerate(lines) if line.startswith("POINTS")][0]
    nb_points = int(lines[idx_points].split()[1])

    # 2. VECTORIZED COORDINATE READING
    # Read all coordinate lines and concatenate them into a single string/list
    coord_lines = ' '.join(lines[idx_points + 1:idx_points + 1 + nb_points])

    # Convert all coordinates into a single NumPy array (nb_points x 3)
    all_coordinates = np.fromstring(coord_lines, sep=' ').reshape((nb_points, 3))

    # Find the data section index
    idx_data = [i for i, line in enumerate(lines) if line.startswith(data_name.upper())][0]

    # Assume data is vector (3 components) or scalar (1 component)
    data_format = lines[idx_data].split()
    nb_data_components = int(data_format[1])  # Assuming structure like: 'VECTORS field_name float'

    # 3. VECTORIZED DATA READING
    # Read all data lines, including the header line, to get the total chunk.
    # We must find where the data chunk ends (usually at the next header line or end of file)

    # Simple reading approach (assuming data is contiguous without line breaks in between components):
    idx_data_start = idx_data + 1

    # We rely on the coordinates array size (nb_points) to determine the data size.
    data_lines = lines[idx_data_start:idx_data_start + nb_points]

    # Join and convert all data into a single NumPy array (nb_points x N)
    all_data_lines = ' '.join(data_lines)
    all_data = np.fromstring(all_data_lines, sep=' ').reshape((nb_points, nb_data_components))

    # 4. VECTORIZED INDEXING using KDTree (The biggest speedup)

    # Build the KDTree from ALL coordinates in the file. This is O(N log N).
    tree = KDTree(all_coordinates)

    # Convert target points to NumPy array
    target_points = np.array(coordinate_points)

    # Query the tree for the nearest neighbor index of each target point.
    # This is O(M log N) where M is number of target points, much faster than O(M * N).
    # d: distances, idx: indices
    distances, indices = tree.query(target_points)

    # 5. VECTORIZED DATA EXTRACTION
    # Extract the coordinates and data corresponding to the found indices
    coords = all_coordinates[indices].tolist()
    data = all_data[indices].tolist()

    return coords, data


def read(folder, coordinate_points, data_name, output_file):
    """
    Read VTK files from a folder and extract data at specified coordinate points.

    Parameters:
    folder (str): Path to the folder containing VTK files.
    coordinate_points (list): List of coordinate points to extract data from.
    data_name (str): Name of the data to extract (e.g., "DISPLACEMENT", "VELOCITY", "ACCELERATION").
    output_file (str): Path to the output JSON file.
    """

    # check data name
    valid_vector_data_names = ["DISPLACEMENT", "VELOCITY", "ACCELERATION"]
    valid_tensor_data_names = ["CAUCHY_STRESS_VECTOR"]
    if data_name.upper() not in valid_vector_data_names and data_name.upper() not in valid_tensor_data_names:
        raise ValueError(f"Invalid data name. Choose from {valid_vector_data_names + valid_tensor_data_names}")

    # check if vector or tensor data
    is_vector_data = data_name.upper() in valid_vector_data_names

    # list all vtk files in the folder
    files = os.listdir(folder)
    files = [file for file in files if file.endswith('.vtk')]

    # sort files based on numeric value in filename
    files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or -1))

    # read each file
    coord_list = []
    data_list = []
    for file in files:
        coordinates, data = read_vtk_local_interpolated(os.path.join(folder, file), coordinate_points, data_name,
                                                        is_vector_data)
        data_list.append(data)
        coord_list.append(coordinates)

    # check uniqueness of the coordinates
    for i, _ in enumerate(coordinate_points):
        for c in coord_list:
            if not np.allclose(c[i], coord_list[0][i]):
                raise ValueError("The coordinate points do not match across all files.")

    data = {
        "STEP": np.linspace(0,
                            len(files) - 1, len(files)).astype(int).tolist(),
        "COORDINATES": coord_list[0],
        "DATA": data_list
    }

    # save the data
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    coordinate_x_coords = np.linspace(0, 20, 21)
    coordinate_y_coords = np.ones(21) * 9
    coordinate_z_coords = np.zeros(21)
    coordinate_points = np.array([coordinate_x_coords, coordinate_y_coords, coordinate_z_coords]).T.tolist()

    vtk_file_dir = r"benchmark_tests/test_strip_load/inputs_kratos/output/output_vtk_porous_computational_model_part"

    read(vtk_file_dir, coordinate_points, "CAUCHY_STRESS_VECTOR", "results.json")

    with open("results.json", "r") as f:
        data = json.load(f)

    coord = data["COORDINATES"]
    x_coords = [c[0] for c in coord]
    velocity_data = data["DATA"]

    # calculate velocity magnitudes at each step
    velocity_magnitudes = []
    for step_data in velocity_data:
        magnitudes = [np.linalg.norm(v) for v in step_data]
        velocity_magnitudes.append(magnitudes)

    # get max velocity magnitude at each coordinate over all time steps
    max_velocity_magnitudes = np.max(velocity_magnitudes, axis=0)

    plt.plot(x_coords, max_velocity_magnitudes)
    plt.xlabel("x coord")
    plt.ylabel("Velocity Magnitude")
    plt.show()
