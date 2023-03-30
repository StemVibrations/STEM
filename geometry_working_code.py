import gmsh
import sys
import tkinter as tk

answer = input("Do you want to change the default values for number of slots? (y/n): ")
if answer == "y":
    print("Enter the number of input slots that you need for: ")
    num1 = int(input("points: "))
    num2 = int(input("lines: "))
    num3 = int(input("surfaces: "))
    lc = float(input("Enter the mesh size(with base 10): "))
    print("Entered values:", num1, num2, num3, "lc=", lc)
elif answer == "n":
    num1 = num2 = 5
    num3 = 2
    lc = 1e-1
    # Print the integers entered by user
    print("*Default values*")
    print("Max #ofpoints:",num1, "Max #oflines:", num2, "Max #ofsurfaces:", num3, "lc=", lc)
else:
    # handle invalid input
    print("Invalid input.")

"""
test points
p1 = [0, 0, 0, lc]
p2 = [1, 0, 0, lc]
p3 = [1, 3, 0, lc]
p4 = [0, 3, 0, lc]

test lines
l1 = create_line(1, 2)
l2 = create_line(2, 3)
l3 = create_line(3, 4)
l4 = create_line(4, 1)

test surface
s = [1, 2, 3, 4]
"""

class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

class Curve:
    def __init__(self, *args):
        for i, arg in enumerate(args):
            setattr(self, f'l{i+1}', arg)

#gmsh functions
def create_point(input):
    x = input[0]
    y = input[1]
    z = input[2]
    lc = input[3]
    gmsh.model.geo.addPoint(x, y, z, lc)

def create_line(input):
    point1 = input[0]
    point2 = input[1]
    l = gmsh.model.geo.addLine(point1, point2)
    return l
#
def create_surface(shape):
    gmsh.model.geo.addCurveLoop(shape, 1)
    s = gmsh.model.geo.addPlaneSurface([1],1)
    return s

def submit_points():
    num_points = int(app.num_points_entry.get())
    points = []

    for i in range(num_points):
        x = int(app.point_entries[i]['x'].get())
        y = int(app.point_entries[i]['y'].get())
        z = int(app.point_entries[i]['z'].get())

        point = Point(x, y, z)
        points.append(point)

    return points, num_points

def submit_lines():
    num_lines = int(app.num_lines_entry.get())
    lines = []

    for i in range(num_lines):
        try:
            p1 = int(app.lines_entries[i]['p1'].get())
        except ValueError:
            print("Enter integer number!")
            p1 = 0  # Set a default value if the user did not input a valid integer
        try:
            p2 = int(app.lines_entries[i]['p2'].get())
        except ValueError:
            print("Enter integer number!")
            p2 = 0  # Set a default value if the user did not input a valid integer

        line = Line(p1, p2)
        lines.append(line)

    return lines, num_lines

def submit_curves():
    num_curves = int(app.num_curves_entry.get())
    curves = []

    for i in range(num_curves):
        try:
            l1 = int(app.curves_entries[i]['l1'].get())
        except ValueError:
            print("Enter integer number!")
            l1 = 0  # Set a default value if the user did not input a valid integer
        try:
            l2 = int(app.curves_entries[i]['l2'].get())
        except ValueError:
            print("Enter integer number!")
            l2 = 0  # Set a default value if the user did not input a valid integer
        try:
            l3 = int(app.curves_entries[i]['l3'].get())
        except ValueError:
            print("Enter integer number!")
            l3 = 0  # Set a default value if the user did not input a valid integer
        try:
            l4 = int(app.curves_entries[i]['l4'].get())
        except ValueError:
            print("Enter integer number!")
            l4 = 0  # Set a default value if the user did not input a valid integer

        curve = Curve(l1, l2, l3, l4)
        curves.append(curve)

    return curves, num_curves

def make_geometry(points, np, lc, lines, nl, curves, nc):
    for i in range(np):
        p = [points[i].x, points[i].y, points[i].z, lc]
        create_point(p)
    for i in range(nl):
        l = [lines[i].p1, lines[i].p2]
        create_line(l)
    for i in range(nc):
        s = [curves[i].l1, curves[i].l2, curves[i].l3, curves[i].l4]
        # print(s)
        create_surface(s)

# Create a tkinter window
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets_points()
        # self.create_widgets_lines()

    def create_widgets_points(self):
        # Add a label and entry widget for the number of points
        num_points_label = tk.Label(self, text="Number of points:")
        num_points_label.grid(row=0, column=0)
        self.num_points_entry = tk.Entry(self)
        self.num_points_entry.grid(row=0, column=1)

        # Add a label and entry widgets for each point
        self.point_entries = []
        for i in range(num1):
            x_label = tk.Label(self, text=f"Point {i + 1} x:")
            x_label.grid(row=i + 1, column=0)
            x_entry = tk.Entry(self)
            x_entry.grid(row=i + 1, column=1)

            y_label = tk.Label(self, text=f"Point {i + 1} y:")
            y_label.grid(row=i + 1, column=2)
            y_entry = tk.Entry(self)
            y_entry.grid(row=i + 1, column=3)

            z_label = tk.Label(self, text=f"Point {i + 1} z:")
            z_label.grid(row=i + 1, column=4)
            z_entry = tk.Entry(self)
            z_entry.grid(row=i + 1, column=5)
            self.point_entries.append({'x': x_entry, 'y': y_entry, 'z': z_entry})

        num_lines_label = tk.Label(self, text="Number of lines:")
        num_lines_label.grid(row=num1+1, column=0)
        self.num_lines_entry = tk.Entry(self)
        self.num_lines_entry.grid(row=num1+1, column=1)
        self.lines_entries = []
        for i in range(num1+2, num1+num2+2):
            x_label = tk.Label(self, text=f"line {i - (num1+2) + 1} p1:")
            x_label.grid(row=i + 1, column=0)
            x_entry = tk.Entry(self)
            x_entry.grid(row=i + 1, column=1)

            y_label = tk.Label(self, text=f"line {i - (num1+2) + 1} p2:")
            y_label.grid(row=i + 1, column=2)
            y_entry = tk.Entry(self)
            y_entry.grid(row=i + 1, column=3)

            self.lines_entries.append({'p1': x_entry, 'p2': y_entry})

        num_curves_label = tk.Label(self, text="Number of curves:")
        num_curves_label.grid(row=num1+num2+3, column=0)
        self.num_curves_entry = tk.Entry(self)
        self.num_curves_entry.grid(row=num1+num2+3, column=1)
        self.curves_entries = []

        for i in range(num1+num2+4, num1+num2+num3+4):
            x_label = tk.Label(self, text=f"curve {i - (num1+num2+4) + 1} l1:")
            x_label.grid(row=i + 1, column=0)
            x_entry = tk.Entry(self)
            x_entry.grid(row=i + 1, column=1)

            y_label = tk.Label(self, text=f"curve {i - (num1+num2+4) + 1} l2:")
            y_label.grid(row=i + 1, column=2)
            y_entry = tk.Entry(self)
            y_entry.grid(row=i + 1, column=3)

            z_label = tk.Label(self, text=f"curve {i - (num1+num2+4) + 1} l3:")
            z_label.grid(row=i + 1, column=4)
            z_entry = tk.Entry(self)
            z_entry.grid(row=i + 1, column=5)

            t_label = tk.Label(self, text=f"curve {i - (num1+num2+4) + 1} l4:")
            t_label.grid(row=i + 1, column=6)
            t_entry = tk.Entry(self)
            t_entry.grid(row=i + 1, column=7)

            self.curves_entries.append({'l1': x_entry, 'l2': y_entry, 'l3': z_entry, 'l4': t_entry})

        self.submit_button = tk.Button(self, text="Submit", command=self.submit_callback)
        self.submit_button.grid(row=30, column=0, columnspan=20)

    def submit_callback(self):
        points, np = submit_points()
        lines, nl = submit_lines()
        curves, nc = submit_curves()

        gmsh.initialize()
        gmsh.model.add("geometry")

        make_geometry(points, np, lc, lines, nl, curves, nc)

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.write("geometry.msh")
        if '-nopopup' not in sys.argv:
            gmsh.fltk.run()
        gmsh.finalize()

root = tk.Tk()
app = Application(master=root)
app.mainloop()

