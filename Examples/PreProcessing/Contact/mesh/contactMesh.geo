// Two unit cubes separated by a parametric gap, using newp/newl/... and Extrude.
// Hex mesh. Save as stacked_gap.geo, then: gmsh -3 stacked_gap.geo

// --- Parameters ---
Lx = 1; Ly = 1; Lz = 1;
gap = 0.1;            // vertical gap between cubes
nx = 10; ny = 10; nz = 10; // elements along x, y, z

// --- Base rectangle at z = 0 (bottom cube) ---
p1 = newp; Point(p1) = {0,  0,  0, 1};
p2 = newp; Point(p2) = {Lx, 0,  0, 1};
p3 = newp; Point(p3) = {Lx, Ly, 0, 1};
p4 = newp; Point(p4) = {0,  Ly, 0, 1};

l1 = newl; Line(l1) = {p1, p2}; // y = 0
l2 = newl; Line(l2) = {p2, p3}; // x = Lx
l3 = newl; Line(l3) = {p3, p4}; // y = Ly
l4 = newl; Line(l4) = {p4, p1}; // x = 0

ll1 = newll; Line Loop(ll1) = {l1, l2, l3, l4};
s_bot_bottom = news; Plane Surface(s_bot_bottom) = {ll1}; // bottom cube -z

// Structured 2D on base
Transfinite Curve {l1, l3} = nx + 1;
Transfinite Curve {l2, l4} = ny + 1;
Transfinite Surface {s_bot_bottom};
Recombine Surface {s_bot_bottom};

// --- Extrude #1: bottom cube (z: 0 -> 1) ---
out1[] = Extrude {0, 0, Lz} { Surface{s_bot_bottom}; Layers{nz}; Recombine; };
s_bot_top = out1[0];     // +z of bottom cube
v_bot     = out1[1];
s_bot_y0  = out1[2];     // -y
s_bot_xL  = out1[3];     // +x
s_bot_yL  = out1[4];     // +y
s_bot_x0  = out1[5];     // -x

// --- Second rectangle at z = 1 + gap (top cube base) ---
p5 = newp; Point(p5) = {0,  0,  Lz + gap, 1};
p6 = newp; Point(p6) = {Lx, 0,  Lz + gap, 1};
p7 = newp; Point(p7) = {Lx, Ly, Lz + gap, 1};
p8 = newp; Point(p8) = {0,  Ly, Lz + gap, 1};

m1 = newl; Line(m1) = {p5, p6}; // y = 0
m2 = newl; Line(m2) = {p6, p7}; // x = Lx
m3 = newl; Line(m3) = {p7, p8}; // y = Ly
m4 = newl; Line(m4) = {p8, p5}; // x = 0

ll2 = newll; Line Loop(ll2) = {m1, m2, m3, m4};
s_top_bottom = news; Plane Surface(s_top_bottom) = {ll2}; // top cube -z

// Structured 2D on second base
Transfinite Curve {m1, m3} = nx + 1;
Transfinite Curve {m2, m4} = ny + 1;
Transfinite Surface {s_top_bottom};
Recombine Surface {s_top_bottom};

// --- Extrude #2: top cube (z: 1+gap -> 2+gap) ---
out2[] = Extrude {0, 0, Lz} { Surface{s_top_bottom}; Layers{nz}; Recombine; };
s_top_top = out2[0];     // +z of top cube
v_top     = out2[1];
s_top_y0  = out2[2];     // -y
s_top_xL  = out2[3];     // +x
s_top_yL  = out2[4];     // +y
s_top_x0  = out2[5];     // -x

// Transfinite volumes for hex mesh
Transfinite Volume {v_bot, v_top};

// --- Physical groups ---
// Bottom cube
Physical Surface("bottom_-z") = {s_bot_bottom};
Physical Surface("bottom_+z") = {s_bot_top};
Physical Surface("bottom_-y") = {s_bot_y0};
Physical Surface("bottom_+x") = {s_bot_xL};
Physical Surface("bottom_+y") = {s_bot_yL};
Physical Surface("bottom_-x") = {s_bot_x0};
Physical Volume("bottom")     = {v_bot};

// Top cube
Physical Surface("top_-z") = {-s_top_bottom};
Physical Surface("top_+z") = {s_top_top};
Physical Surface("top_-y") = {s_top_y0};
Physical Surface("top_+x") = {s_top_xL};
Physical Surface("top_+y") = {s_top_yL};
Physical Surface("top_-x") = {s_top_x0};
Physical Volume("top")     = {v_top};

Mesh 3;
Save "contactMesh.msh";