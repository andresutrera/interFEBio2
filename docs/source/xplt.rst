xplt: FEBio ``.xplt`` Reader
============================

Slice-first results API on top of your standalone :class:`Mesh` container.

Overview
--------

``xplt`` parses FEBio ``.xplt`` files, builds a :class:`~interFEBio.Mesh.Mesh.Mesh`
object, and exposes results through thin views:

- Nodal results: :class:`NodeResultView` with shape ``(T, N, C)``
- Element results: :class:`ElemResultView` per **domain/part**, shape ``(T, Ne, C)``
- Surface results: :class:`FaceResultView` per **surface**, shape ``(T, Nf, C)``

Here

- ``T`` = number of time steps
- ``N`` = number of mesh nodes
- ``Ne`` = number of elements in a domain
- ``Nf`` = number of surface facets
- ``C`` = number of components of the field (e.g., 3 for a vector, 6 for a symmetric tensor)

Install and import
------------------

.. code-block:: python

    from interFEBio.XPLT.XPLT import xplt
    from interFEBio.Mesh.Mesh import Mesh  # only needed if you use Mesh helpers

Quick start
-----------

.. code-block:: python

    xp = xplt("run.xplt")     # parse header, dictionary, mesh geometry
    xp.readAllStates()        # stream all states into memory
    m = xp.mesh               # Mesh instance
    res = xp.results          # Results registry

    print(m.nnodes, m.nelems) # mesh sizes
    print(res.times().shape)  # (T,)

Core objects
------------

``xplt``
^^^^^^^^

- ``mesh``: :class:`Mesh`
- ``results``: :class:`Results`
- ``dictionary``: ``dict`` of available variables with FE storage types
- ``version``: integer file version
- ``compression``: integer compression flag (as in file)

Public methods:

- ``readAllStates()``: read every state in sequence
- ``readSteps(stepList: List[int])``: read specific state indices (1-based in file order)

``Mesh``
^^^^^^^^

- ``nodes``: :class:`NodeArray`
- ``elements``: :class:`ElementArray`
- ``parts``: ``Dict[str, np.ndarray]`` mapping part name → element-row indices
- ``surfaces``: ``Dict[str, SurfaceArray]`` mapping name → facet table
- ``nodesets``: ``Dict[str, np.ndarray]`` mapping name → 0-based node ids

Helpers:

- ``getDomain(name) -> ElementArray``
- ``getSurface(name) -> SurfaceArray``
- ``getNodeset(name) -> NodeArray``
- ``bounds() -> (min, max)``

``Results``
^^^^^^^^^^^

- ``times() -> np.ndarray``: array of time values ``(T,)``
- ``__getitem__(name)``: return the view for that variable

Views
-----

Common controls
^^^^^^^^^^^^^^^

All views share:

- ``getTime(index)``: slice time vector
- ``comp(index)``: select component(s). Returns a **new** view with component filter applied

``NodeResultView`` (nodal)
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Data order: ``(T, N, C)``
- Indexing:

  - ``view[t] -> (N, C)``
  - ``view[t, nodes] -> (|nodes|, C)``
  - ``view[t, node, c] -> scalar``

- Helpers:

  - ``nodes(ids)``: narrow to arbitrary node ids
  - ``nodeset(name)``: narrow to a named nodeset

``ElemResultView`` (per-domain)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Data order per domain: ``(T, Ne, C)``
- ``domains()``: list available domain names
- ``domain(name)``: return a view narrowed to one domain
- ``at(t, domain=..., elems=...)``: extract arrays with explicit selectors
- ``values(domain)``: shorthand for ``at(slice(None), domain, slice(None))``

``FaceResultView`` (per-surface)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Data order per surface: ``(T, Nf, C)``
- ``surfaces()``: list surface names
- ``surface(name)``: return a view narrowed to one surface
- ``at(t, surface=..., faces=...)`` and ``values(surface)`` analogous to element view

Slicing examples
----------------

Time slicing
^^^^^^^^^^^^

.. code-block:: python

    U = xp.results["displacement"]      # NodeResultView
    tvec = U.getTime(slice(None))       # (T,)

    U_t0 = U[0]                         # (N, C)
    U_t0_9 = U[0:10]                    # (10, N, C)
    U_t_pick = U[[0, 5, 9]]             # (3, N, C)

Component selection
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    Uy_all_t = U.comp(1)[:]             # (T, N)
    Uyz_t0_nodes = U.comp([1, 2])[0, 0:10]  # (10, 2)

Nodes and nodesets
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    U_nodes_0_9_all_t = U[:, 0:10]             # (T, 10, C)
    U_nodes_pick_t0 = U[0, [0, 5, 42]]         # (3, C)

    U_top_all_t = U.nodeset("TopRing")[:]      # (T, set size, C)
    Uy_top_t5 = U.nodeset("TopRing").comp(1)[5]  # (set size,)

Elements by domain
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    S = xp.results["stress"]                   # ElemResultView, e.g. 6 comps
    S.domains()                                # ['Core', 'Shell', ...]

    S_core = S.domain("Core")                  # narrowed view
    S_core_t0 = S_core.at(0)                   # (Ne_core, C)

    S_core_first10_all_t = S_core.at(slice(None), elems=slice(0, 10))  # (T, 10, C)
    S_core_yy_t0 = S_core.comp(1).at(0)        # (Ne_core,)

Surfaces and faces
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    CF = xp.results["contact force"]           # FaceResultView
    CF.surfaces()                              # ['contactPin', ...]

    CF_pin = CF.surface("contactPin")
    CF_pin_t0 = CF_pin.at(0)                   # (Nf, C)
    CF_pin_faces_0_9_all_t = CF_pin.at(slice(None), faces=slice(0, 10))  # (T, 10, C)
    CF_pin_y_all_t = CF_pin.comp(1).at(slice(None))  # (T, Nf)

Chaining with mesh selections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    core_elems = xp.mesh.getDomain("Core")     # ElementArray
    core_nodes = core_elems.unique_nodes()     # np.ndarray of node ids
    U_core_nodes = U[:, core_nodes]            # (T, |unique|, C)

Minimal extraction patterns
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # nodal time history of a single node component
    Uy_node_42 = U.comp(1)[:, 42]              # (T,)

    # per-element field at a single time
    S_core_vm_t0 = S.domain("Core").comp(0).at(0)  # if comp 0 encodes VM -> (Ne_core,)

    # surface field, single component at t0
    CFy_pin_t0 = CF.surface("contactPin").comp(1).at(0)  # (Nf,)

Working with the Mesh
---------------------

.. code-block:: python

    m = xp.mesh
    print(m.nnodes, m.nelems)
    bb_min, bb_max = m.bounds()

    # Name-first access
