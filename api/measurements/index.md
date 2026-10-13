# Measurements

## `Measurement(name, joint, frame, type, value)`

Class for handling different types of measurements.

Initialize measurement.

Parameters:

| Name    | Type | Description                                | Default    |
| ------- | ---- | ------------------------------------------ | ---------- |
| `name`  |      | Name of measurement                        | *required* |
| `joint` |      | Joint where measurement is expressed       | *required* |
| `frame` |      | Closest frame from measurement             | *required* |
| `type`  |      | Type of measurement (SE3, wrench, current) | *required* |
| `value` |      | 6D measurement value wrt joint placement   | *required* |

Source code in `src/figaroh/measurements/measurement.py`

```
def __init__(self, name, joint, frame, type, value):
    """Initialize measurement.

    Args:
        name: Name of measurement
        joint: Joint where measurement is expressed
        frame: Closest frame from measurement
        type: Type of measurement (SE3, wrench, current)
        value: 6D measurement value wrt joint placement
    """
    self.name = name
    self.joint = joint
    self.frame = frame
    self.type = type
    if self.type == "SE3":
        self.SE3_value = pin.SE3(
            pin.rpyToMatrix(np.array([value[3], value[4], value[5]])),
            np.array([value[0], value[1], value[2]]),
        )
    elif self.type == "wrench":
        self.wrench_value = pin.Force(value)
    elif self.type == "current":
        self.current_value = value
    else:
        raise TypeError("The type of your measurement is not valid")
```

### `add_SE3_measurement(model)`

Add SE3 measurement to model.

Parameters:

| Name    | Type | Description                           | Default    |
| ------- | ---- | ------------------------------------- | ---------- |
| `model` |      | Pinocchio model to add measurement to | *required* |

Returns:

| Name    | Type | Description            |
| ------- | ---- | ---------------------- |
| `tuple` |      | Updated model and data |

Source code in `src/figaroh/measurements/measurement.py`

```
def add_SE3_measurement(self, model):
    """Add SE3 measurement to model.

    Args:
        model: Pinocchio model to add measurement to

    Returns:
        tuple: Updated model and data
    """
    if self.type == "SE3":
        self.frame_index = model.addFrame(
            pin.Frame(
                self.name,
                model.getJointId(self.joint),
                model.getFrameId(self.frame),
                self.SE3_value,
                pin.OP_FRAME,
            ),
            "False",
        )
    data = model.createData()
    return model, data
```
