"""
Тесты системных эндпоинтов
"""


class TestHealth:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, client):
        response = client.get("/health")
        data     = response.json()
        assert data["status"] == "ok"

    def test_health_has_timestamp(self, client):
        response = client.get("/health")
        data     = response.json()
        assert "timestamp" in data
        assert isinstance(data["timestamp"], float)


class TestReady:

    def test_ready_returns_200_when_loaded(self, client):
        response = client.get("/ready")
        assert response.status_code == 200

    def test_ready_returns_correct_structure(self, client):
        response = client.get("/ready")
        data     = response.json()

        assert data["status"] == "ready"
        assert "llm" in data
        assert "rag" in data

    def test_ready_returns_503_when_not_loaded(self, client, llm_mock):
        """503 если модель не загружена"""
        # Временно меняем возвращаемое значение мока
        llm_mock.is_loaded.return_value = False

        response = client.get("/ready")
        assert response.status_code == 503

        # Возвращаем обратно
        llm_mock.is_loaded.return_value = True


class TestRoot:

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_service_info(self, client):
        response = client.get("/")
        data     = response.json()

        assert "service" in data
        assert "version" in data
        assert "docs"    in data

    def test_root_docs_path(self, client):
        """docs указывает на /docs"""
        response = client.get("/")
        data     = response.json()
        assert data["docs"] == "/docs"


class TestStats:

    def test_stats_returns_200(self, client):
        response = client.get("/stats")
        assert response.status_code == 200