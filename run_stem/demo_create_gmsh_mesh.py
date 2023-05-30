

#todo group input into geometry, settings, ...

from gmsh_utils.gmsh_IO import GmshIO

# define the points of the surface as a list of tuples
input_points = [(0, 0, 0), (1, 0, 0), (1, 3, 0), (0, 3, 0), (-1, 1.5, 0)]
# define the element size
element_size = 2
# define geometry dimension; input "3" for 3D to extrude the 2D surface, input "2" for 2D
dims = 3
# if 3D, input depth of geometry to be extruded from 2D surface
extrusion_length = [0, 0, 2]
# set a name label for the surface
name_label = "Soil Layer"
# if "True", saves mesh data to separate mdpa files; otherwise "False"
save_file = False
# if "True", opens gmsh interface; otherwise "False"
open_gmsh_gui = False
# set a name for mesh output file
mesh_name = "geometry"
# set output directory
mesh_output_dir = "./"



gmsh_io = GmshIO()

gmsh_io.generate_gmsh_mesh(input_points, extrusion_length, element_size, dims, name_label, mesh_name, mesh_output_dir,
                           save_file, open_gmsh_gui)

mesh_data = gmsh_io.mesh_data

