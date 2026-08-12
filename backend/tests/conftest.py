import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routes import get_model_service


class StubModelService:
    """Test double so the API test-suite never needs the real trained artifact."""

    is_loaded = True
    class_names = ["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-Damage", "Snow-Covered"]

    def predict(self, image_bytes: bytes) -> dict:
        probabilities = {name: (0.6 if name == "Clean" else 0.08) for name in self.class_names}
        return {"predicted_class": "Clean", "confidence": 0.6, "probabilities": probabilities}


@pytest.fixture
def stub_service():
    return StubModelService()


@pytest.fixture
def client(stub_service):
    app.dependency_overrides[get_model_service] = lambda: stub_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
