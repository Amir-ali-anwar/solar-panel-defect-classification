import io

from PIL import Image


def make_image_bytes(fmt="JPEG"):
    image = Image.new("RGB", (32, 32), color=(120, 180, 90))
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    buf.seek(0)
    return buf


def test_predict_returns_class_and_confidence(client):
    file_bytes = make_image_bytes()
    response = client.post(
        "/api/predict",
        files={"file": ("panel.jpg", file_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == "Clean"
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"].keys()) == {
        "Bird-drop",
        "Clean",
        "Dusty",
        "Electrical-damage",
        "Physical-Damage",
        "Snow-Covered",
    }


def test_predict_rejects_unsupported_content_type(client):
    response = client.post(
        "/api/predict",
        files={"file": ("notes.txt", io.BytesIO(b"not an image"), "text/plain")},
    )
    assert response.status_code == 415


def test_predict_rejects_oversized_file(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "max_upload_mb", 0.0001)
    file_bytes = make_image_bytes()
    response = client.post(
        "/api/predict",
        files={"file": ("panel.jpg", file_bytes, "image/jpeg")},
    )
    assert response.status_code == 413
