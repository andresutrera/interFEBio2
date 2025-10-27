Mesh.MshFileVersion = 2.2;
// Parametric unit cube using newp/newl/... and Extrude; all faces named; hex mesh.
// Save as cube.geo, then: gmsh -3 cube.geo

// --- Parameters ---
Lx = 1; Ly = 1; Lz = 1;        // dimensions
nx = 10; ny = 10; nz = 10;     // elements along x, y, z

// --- Base rectangle in z = 0 ---
p1 = newp; Point(p1) = {0,  0,  0, 1};
p2 = newp; Point(p2) = {Lx, 0,  0, 1};
p3 = newp; Point(p3) = {Lx, Ly, 0, 1};
p4 = newp; Point(p4) = {0,  Ly, 0, 1};

l1 = newl; Line(l1) = {p1, p2}; // +x
l2 = newl; Line(l2) = {p2, p3}; // +y
l3 = newl; Line(l3) = {p3, p4}; // -x
l4 = newl; Line(l4) = {p4, p1}; // -y

ll = newll; Line Loop(ll) = {l1, l2, l3, l4};
s0 = news; Plane Surface(s0) = {ll}; // -z face

// --- Structured mesh settings on base ---
Transfinite Curve {l1, l3} = nx + 1; // nx elements along x
Transfinite Curve {l2, l4} = ny + 1; // ny elements along y
Transfinite Surface {s0};
Recombine Surface {s0};

// --- Extrude to make volume (returns: topSurf, vol, side1, side2, side3, side4) ---
out[] = Extrude {0, 0, Lz} { Surface{s0}; Layers{nz}; Recombine; };
sTop = out[0];
vol  = out[1];
sSide1 = out[2]; // from l1 -> y = 0  (-y)
sSide2 = out[3]; // from l2 -> x = Lx (+x)
sSide3 = out[4]; // from l3 -> y = Ly (+y)
sSide4 = out[5]; // from l4 -> x = 0  (-x)

// --- Transfinite volume for hex mesh ---
Transfinite Volume {vol};

// --- Physical groups ---
Physical Surface("-z") = {s0};
Physical Surface("+z") = {sTop};
Physical Surface("-y") = {sSide1};
Physical Surface("+x") = {sSide2};
Physical Surface("+y") = {sSide3};
Physical Surface("-x") = {sSide4};
Physical Volume("volume") = {vol};

Mesh 3;
Save "cube.msh";