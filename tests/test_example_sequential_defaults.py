import inspect

import pytest

from examples.epstein_civil_violence.model import EpsteinModel
from examples.negotiation.model import NegotiationModel
from examples.sugarscrap_g1mt.model import SugarScapeModel


@pytest.mark.parametrize(
    "model_class",
    [EpsteinModel, NegotiationModel, SugarScapeModel],
    ids=lambda model_class: model_class.__name__,
)
def test_normal_example_model_does_not_expose_parallel_stepping(model_class):
    parameters = inspect.signature(model_class.__init__).parameters

    assert "parallel_stepping" not in parameters, (
        f"{model_class.__name__} should use normal sequential Mesa stepping "
        "without exposing an inert parallel_stepping parameter"
    )
