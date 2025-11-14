interFEBio package reference
============================

This page highlights the main subpackages that ship with ``interFEBio`` and
shows how they interact to produce FEBio-ready meshes, monitor optimization
runs, and expose results through PyVista and FastAPI.

Mesh and geometry utilities
---------------------------

Mesh construction and slicing helpers keep connectivity, surfaces, and node sets
separated so that downstream code can grab the subset it needs.

.. automodule:: interFEBio.Mesh.Mesh
   :members: NodeArray, ElementArray, SurfaceArray, Mesh
   :noindex:

Using ``NodeArray`` and ``ElementArray`` together lets you describe FEBio-ready
connectivity without force-fitting a single element type.

Logging bridge
--------------

All logging funnels through a singleton to avoid repeated configuration side
effects while keeping the interface short.

.. automodule:: interFEBio.Log
   :members:
   :noindex:

Optimization helpers
--------------------

Optimization modules provide interpolators, storage, and execution logic for the
calibration routines. The core cost combines experimental and simulation traces
as

.. math::

   J(\theta) = \sum_{i=1}^n \left( y_{\text{sim}}(x_i;\theta) - y_{\text{exp}}(x_i) \right)^2

so that ``theta`` minimizes the squared deviation of the computed run.

.. automodule:: interFEBio.Optimize.alignment
   :members: EvaluationGrid, Aligner
   :noindex:

.. automodule:: interFEBio.Optimize.engine
   :members:
   :noindex:

Examples
~~~~~~~~

Use ``examples/simple_biaxial_fit.py`` to see a complete optimization pipeline:

.. code-block:: python

   from interFEBio.Optimize.engine import FitEngine

   engine = FitEngine(...)
   engine.run()

Monitoring services
-------------------

The monitoring subpackage bundles WebSocket listeners, registries, and the FastAPI
app that exposes optimization metadata to browsers.

.. automodule:: interFEBio.monitoring.client
   :members: MonitorConfig, OptimizationMonitorClient
   :noindex:

.. automodule:: interFEBio.monitoring.events
   :members: EventEmitter, SocketEventEmitter, EventSocketListener
   :noindex:

.. automodule:: interFEBio.monitoring.registry
   :members: RunRegistry, ActiveRunDeletionError
   :noindex:

.. automodule:: interFEBio.monitoring.state
   :members: StorageInventory
   :noindex:

.. automodule:: interFEBio.monitoring.service
   :members: run_service, install_service, uninstall_service
   :noindex:

.. automodule:: interFEBio.monitoring.webapp
   :members: create_app, list_runs, run_detail, run_iterations, delete_run, delete_all_runs
   :noindex:

Visualization helpers
---------------------

PyVista bridges translate FEBio meshes into unstructured grids and attach nodal
and elemental datasets.

.. automodule:: interFEBio.Visualization.Plotter
   :members: PVBridge
   :noindex:

XPLT readers
-------------

The ``XPLT`` subpackage interprets FEBio ``.xplt`` files, reads binary blocks,
and exposes views that slice time, region, and component axes on demand.

.. automodule:: interFEBio.XPLT.BinaryReader
   :members:
   :noindex:

.. automodule:: interFEBio.XPLT.Enums
   :members:
   :noindex:

.. automodule:: interFEBio.XPLT.XPLT
   :members:
   :noindex:
