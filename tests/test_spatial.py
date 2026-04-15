import numpy as np
from openrobot_cognition.spatial.scene_graph import SceneGraph


def test_scene_graph_relation():
    sg = SceneGraph()
    sg.register_object("A", np.array([0.0, 0.0, 0.0]))
    sg.register_object("B", np.array([0.1, 0.0, 0.0]))
    rel = sg.query_spatial_relation("A", "B")
    assert rel == "right"

    sg.register_object("C", np.array([0.0, 0.0, 0.1]))
    rel2 = sg.query_spatial_relation("A", "C")
    assert rel2 == "above"
