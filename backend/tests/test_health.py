def test_health_reports_model_loaded(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_classes_returns_class_list(client, stub_service):
    response = client.get("/api/classes")
    assert response.status_code == 200
    assert response.json()["classes"] == stub_service.class_names
