// Gmsh project: created with gmsh-3.0.6-Windows64

// Create 2D square mesh
Mesh.ElementOrder = 1;
Point(1) = {0, 0, 0};
Point(2) = {1, 0, 0};
Point(3) = {1, 1, 0};
Point(4) = {0, 1, 0};

// create lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// create surface
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = 1;

// create new points of second surface
Point(5) = {0, 2, 0};
Point(6) = {1, 2, 0};

// create new lines
Line(5) = {4, 5};
Line(6) = {5, 6};
Line(7) = {6, 3};

// create second surface
Line Loop(2) = {3, 5, 6, 7};
Plane Surface(2) = 2;

// Define the physical groups
Physical Surface("group_1") = 1;
Physical Surface("group_2") = 2;
