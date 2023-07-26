// Gmsh project: created with gmsh-3.0.6-Windows64
boxdim = 0.5;
column_height = 2;
gridsize = 0.5;

// Create 2D square mesh
Mesh.ElementOrder = 1;
Point(1) = {0, 0, 0, gridsize};
Point(2) = {boxdim, 0, 0, gridsize};
Point(3) = {boxdim, column_height/2, 0, gridsize};
Point(4) = {0, column_height/2, 0, gridsize};

// create lines
Line(5) = {1, 2};
Line(6) = {2, 3};
Line(7) = {3, 4};
Line(8) = {4, 1};

// create surface
Line Loop(9) = {5, 6, 7, 8};
Plane Surface(10) = 9;

// create new points of second surface
Point(11) = {0, column_height, 0, gridsize};
Point(12) = {boxdim, column_height, 0, gridsize};

// create new lines
Line(13) = {4, 11};
Line(14) = {11, 12};
Line(15) = {12, 3};

// create second surface
Line Loop(16) = {-13, -14, -15, -7};
Plane Surface(17) = 16;

// divide de lines for the elements
Transfinite Line{6, 8} = column_height/2 / gridsize + 1;
Transfinite Line{5, 7} = boxdim / gridsize + 1;

Transfinite Line{13, 15} = column_height/2 / gridsize + 1;
Transfinite Line{14, 7} = boxdim / gridsize + 1;

Transfinite Surface{10};
Transfinite Surface{17};

// Make 3D by extrusion
group1[] = Extrude {0, 0, -boxdim}{Surface{10};
                                        Layers{boxdim / gridsize};
                                        };

group2[] = Extrude {0, 0, -boxdim}{Surface{17};
                                        Layers{boxdim / gridsize};
                                        };

// Define the volume
Physical Volume("group_1") = {group1[1]};
Physical Volume("group_2") = {group2[1]};