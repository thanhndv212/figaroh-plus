# Visualisation

## `MeshcatVisualizer(robot=None, model=None, collision_model=None, visual_model=None, url=None)`

Bases: `MeshcatVisualizer`

Extension of Pinocchio MeshcatVisualizer with additional features.

Initialize visualizer.

Parameters:

| Name              | Type | Description                  | Default |
| ----------------- | ---- | ---------------------------- | ------- |
| `robot`           |      | Robot object                 | `None`  |
| `model`           |      | Pinocchio model              | `None`  |
| `collision_model` |      | Collision model              | `None`  |
| `visual_model`    |      | Visual model                 | `None`  |
| `url`             |      | Server URL for visualization | `None`  |

Source code in `src/figaroh/visualisation/visualizer.py`

```
def __init__(
    self, robot=None, model=None, collision_model=None, visual_model=None, url=None
):
    """Initialize visualizer.

    Args:
        robot: Robot object
        model: Pinocchio model
        collision_model: Collision model
        visual_model: Visual model
        url: Server URL for visualization
    """
    if robot is not None:
        PMV.__init__(self, robot.model, robot.collision_model, robot.visual_model)
    elif model is not None:
        PMV.__init__(self, model, collision_model, visual_model)

    if url is not None:
        if url == "classical":
            url = "tcp://127.0.0.1:6000"
        logger.info("Wrapper tries to connect to server <%s>", url)
        server = meshcat.Visualizer(zmq_url=url)
    else:
        server = None

    if robot is not None or model is not None:
        self.initViewer(loadModel=True, viewer=server)
    else:
        self.viewer = server if server is not None else meshcat.Visualizer()
```

## `materialFromColor(color)`

Convert color to Material.

Parameters:

| Name    | Type | Description                                           | Default    |
| ------- | ---- | ----------------------------------------------------- | ---------- |
| `color` |      | Color specification (Material, string, list, or None) | *required* |

Returns:

| Type | Description                                |
| ---- | ------------------------------------------ |
|      | meshcat.geometry.Material: Material object |

Source code in `src/figaroh/visualisation/visualizer.py`

```
def materialFromColor(color):
    """Convert color to Material.

    Args:
        color: Color specification (Material, string, list, or None)

    Returns:
        meshcat.geometry.Material: Material object
    """
    if isinstance(color, meshcat.geometry.Material):
        return color
    elif isinstance(color, str):
        material = colors.colormap[color]
    elif isinstance(color, list):
        from .colors import rgb2int

        material = meshcat.geometry.Material()
        material.color = rgb2int(*[int(c * 255) for c in color[:3]])
        if len(color) == 3:
            material.transparent = False
        else:
            material.transparent = color[3] < 1
            material.opacity = float(color[3])
    elif color is None:
        import random

        material = random.sample(list(colors.colormap), 1)[0]
    else:
        material = colors.black
    return material
```
