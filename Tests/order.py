# put this anywhere and run once with your Mesh loaded as m

import numpy as np
from collections import Counter


def _rot_eq(a, b):
    # a,b are length-4 lists; equal up to rotation (not reversal)
    for s in range(4):
        if all(a[(i + s) % 4] == b[i] for i in range(4)):
            return True
    return False


def _hex_faces_from_perm(ids8, perm):
    i = list(ids8[list(perm)])
    # VTK face definitions for HEX8, using permuted corners
    return [
        [i[0], i[1], i[2], i[3]],  # bottom
        [i[4], i[5], i[6], i[7]],  # top
        [i[0], i[1], i[5], i[4]],
        [i[1], i[2], i[6], i[5]],
        [i[2], i[3], i[7], i[6]],
        [i[3], i[0], i[4], i[7]],
    ]


def _candidate_perms():
    # Build 24 plausible FEBio→VTK permutations:
    # rotate ring by r and choose bottom winding cw/ccw; top follows pillars.
    base_ccw = (0, 1, 2, 3, 4, 5, 6, 7)  # VTK
    base_cw = (0, 1, 3, 2, 4, 5, 7, 6)  # common FE
    for base in (base_ccw, base_cw):
        for r in range(4):
            b = [base[(i + r) % 4] for i in range(4)]
            t = [base[4 + ((i + r) % 4)] for i in range(4)]
            yield tuple(b + t)


def determine_hex8_febio_to_vtk_perm(mesh, max_hex_samples=2000):
    # collect oriented QUAD faces present in the mesh (surfaces + QUAD domains)
    quads = []
    for sa in mesh.surfaces.values():
        F = sa.faces
        k = sa.nper
        sel = k == 4
        quads += [F[i, :4].tolist() for i in np.nonzero(sel)[0]]

    # also include QUAD elements that are in 'parts' with 4-noded faces
    EA = mesh.elements
    for e in range(len(EA)):
        if EA.nper[e] == 4:
            quads.append(EA.conn[e, :4].tolist())

    if not quads:
        raise RuntimeError("No oriented QUAD faces found to calibrate against.")

    # speed: convert to tuples for hashing of unordered sets
    quad_sets = {frozenset(q) for q in quads}

    # pick a subset of boundary hexes (those sharing any 4-node set with a quad)
    hex_rows = np.nonzero(
        np.isin(
            np.char.upper(mesh.elements.etype.astype(str)),
            ["ELEM_HEX", "HEX", "HEXA", "BRICK"],
        )
    )[0]
    rng = np.random.default_rng(0)
    rng.shuffle(hex_rows)
    sample = []
    for e in hex_rows:
        ids8 = mesh.elements.conn[e, :8]
        # quick boundary check
        has_boundary = False
        # try all 6 unordered faces from raw ids8
        cand_faces = [
            frozenset(ids8[[0, 1, 2, 3]]),
            frozenset(ids8[[4, 5, 6, 7]]),
            frozenset(ids8[[0, 1, 5, 4]]),
            frozenset(ids8[[1, 2, 6, 5]]),
            frozenset(ids8[[2, 3, 7, 6]]),
            frozenset(ids8[[3, 0, 4, 7]]),
        ]
        if any(cf in quad_sets for cf in cand_faces):
            has_boundary = True
        if has_boundary:
            sample.append(e)
        if len(sample) >= max_hex_samples:
            break
    if not sample:
        raise RuntimeError("Could not find boundary HEX elements to calibrate.")

    # score candidate permutations by oriented matches with QUADs
    scores = Counter()
    for perm in _candidate_perms():
        sc = 0
        for e in sample:
            ids8 = mesh.elements.conn[e, :8]
            for f in _hex_faces_from_perm(ids8, perm):
                if frozenset(f) in quad_sets:
                    # oriented comparison: match any quad up to rotation
                    if any(
                        _rot_eq(f, q) for q in quads if frozenset(q) == frozenset(f)
                    ):
                        sc += 1
        scores[perm] = sc

    best_perm, best_score = max(scores.items(), key=lambda kv: kv[1])
    print("Suggested FEBio HEX8 -> VTK perm:", best_perm, "score:", best_score)
    return best_perm  # put this anywhere and run once with your Mesh loaded as m


import numpy as np
from collections import Counter


def _rot_eq(a, b):
    # a,b are length-4 lists; equal up to rotation (not reversal)
    for s in range(4):
        if all(a[(i + s) % 4] == b[i] for i in range(4)):
            return True
    return False


def _hex_faces_from_perm(ids8, perm):
    i = list(ids8[list(perm)])
    # VTK face definitions for HEX8, using permuted corners
    return [
        [i[0], i[1], i[2], i[3]],  # bottom
        [i[4], i[5], i[6], i[7]],  # top
        [i[0], i[1], i[5], i[4]],
        [i[1], i[2], i[6], i[5]],
        [i[2], i[3], i[7], i[6]],
        [i[3], i[0], i[4], i[7]],
    ]


def _candidate_perms():
    # Build 24 plausible FEBio→VTK permutations:
    # rotate ring by r and choose bottom winding cw/ccw; top follows pillars.
    base_ccw = (0, 1, 2, 3, 4, 5, 6, 7)  # VTK
    base_cw = (0, 1, 3, 2, 4, 5, 7, 6)  # common FE
    for base in (base_ccw, base_cw):
        for r in range(4):
            b = [base[(i + r) % 4] for i in range(4)]
            t = [base[4 + ((i + r) % 4)] for i in range(4)]
            yield tuple(b + t)


def determine_hex8_febio_to_vtk_perm(mesh, max_hex_samples=2000):
    # collect oriented QUAD faces present in the mesh (surfaces + QUAD domains)
    quads = []
    for sa in mesh.surfaces.values():
        F = sa.faces
        k = sa.nper
        sel = k == 4
        quads += [F[i, :4].tolist() for i in np.nonzero(sel)[0]]

    # also include QUAD elements that are in 'parts' with 4-noded faces
    EA = mesh.elements
    for e in range(len(EA)):
        if EA.nper[e] == 4:
            quads.append(EA.conn[e, :4].tolist())

    if not quads:
        raise RuntimeError("No oriented QUAD faces found to calibrate against.")

    # speed: convert to tuples for hashing of unordered sets
    quad_sets = {frozenset(q) for q in quads}

    # pick a subset of boundary hexes (those sharing any 4-node set with a quad)
    hex_rows = np.nonzero(
        np.isin(
            np.char.upper(mesh.elements.etype.astype(str)),
            ["ELEM_HEX", "HEX", "HEXA", "BRICK"],
        )
    )[0]
    rng = np.random.default_rng(0)
    rng.shuffle(hex_rows)
    sample = []
    for e in hex_rows:
        ids8 = mesh.elements.conn[e, :8]
        # quick boundary check
        has_boundary = False
        # try all 6 unordered faces from raw ids8
        cand_faces = [
            frozenset(ids8[[0, 1, 2, 3]]),
            frozenset(ids8[[4, 5, 6, 7]]),
            frozenset(ids8[[0, 1, 5, 4]]),
            frozenset(ids8[[1, 2, 6, 5]]),
            frozenset(ids8[[2, 3, 7, 6]]),
            frozenset(ids8[[3, 0, 4, 7]]),
        ]
        if any(cf in quad_sets for cf in cand_faces):
            has_boundary = True
        if has_boundary:
            sample.append(e)
        if len(sample) >= max_hex_samples:
            break
    if not sample:
        raise RuntimeError("Could not find boundary HEX elements to calibrate.")

    # score candidate permutations by oriented matches with QUADs
    scores = Counter()
    for perm in _candidate_perms():
        sc = 0
        for e in sample:
            ids8 = mesh.elements.conn[e, :8]
            for f in _hex_faces_from_perm(ids8, perm):
                if frozenset(f) in quad_sets:
                    # oriented comparison: match any quad up to rotation
                    if any(
                        _rot_eq(f, q) for q in quads if frozenset(q) == frozenset(f)
                    ):
                        sc += 1
        scores[perm] = sc

    best_perm, best_score = max(scores.items(), key=lambda kv: kv[1])
    print("Suggested FEBio HEX8 -> VTK perm:", best_perm, "score:", best_score)
    return best_perm


from interFEBio.XPLT.XPLT import xplt

xp = xplt("ring.xplt")
# xp.readAllStates()
perm = determine_hex8_febio_to_vtk_perm(xp.mesh)
