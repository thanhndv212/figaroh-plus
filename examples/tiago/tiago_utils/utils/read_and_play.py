from tiago_tools import load_robot
from pinocchio.visualize import GepettoVisualizer
import time
import sys

tiago = load_robot("data/urdf/tiago_48_hey5.urdf")


def play_motion(tiago, q):
    viz = GepettoVisualizer(
        model=tiago.model,
        collision_model=tiago.collision_model,
        visual_model=tiago.visual_model,
    )
    try:
        viz.initViewer()
    except ImportError as err:
        print(
            "Error while initializing the viewer. It seems you should install gepetto-viewer"
        )
        print(err)
        sys.exit(0)

    try:
        viz.loadViewerModel("pinocchio")
    except AttributeError as err:
        print(
            "Error while loading the viewer model. It seems you should start gepetto-viewer"
        )
        print(err)
        sys.exit(0)

    time.sleep(3)
    viz.display(tiago.q0)
    for i in range(q.shape[0]):
        viz.display(q[i])
        time.sleep(0.01)
