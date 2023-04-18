
from stem.gmsh_IO import GmshIO

# define the points of the surface as a list of tuples
input_points = [(0, 0, 0), (1, 0, 0), (1, 3, 0), (0, 3, 0), (-1, 1.5, 0)]
# define the mesh size
mesh_size = 2
# define geometry dimension; input "3" for 3D to extrude the 2D surface, input "2" for 2D
dims = 3
# if 3D, input depth of geometry to be extruded from 2D surface
depth = 2
# set a name label for the surface
name_label = "Soil Layer"
# if "True", saves mesh data to separate mdpa files; otherwise "False"
save_file = True
# if "True", opens gmsh interface; otherwise "False"
gmsh_interface = True
# set a name for mesh output file
mesh_output_name = "geometry"

gmsh_io = GmshIO()

mesh_data = gmsh_io.generate_gmsh_mesh(input_points, depth, mesh_size, dims, name_label, mesh_output_name, save_file,
                                       gmsh_interface)


a=1+1
