"""
Тесты эндпоинта /rag
"""


class TestRagSuccess:

    def test_rag_returns_200(self, client):
        """Базовый успешный запрос"""
        response = client.post("/rag", json={
            "query": "как получить аккредитацию?"
        })
        assert response.status_code == 200

    def test_rag_response_structure(self, client):
        """Ответ содержит все обязательные поля"""
        response = client.post("/rag", json={
            "query": "как получить аккредитацию?"
        })
        data = response.json()

        assert "request_id"   in data
        assert "answer"       in data
        assert "sources"      in data
        assert "sources_used" in data
        assert "latency_ms"   in data

    def test_rag_answer_is_not_empty(self, client):
        """Ответ содержит непустой текст"""
        response = client.post("/rag", json={
            "query": "как получить аккредитацию?"
        })
        data = response.json()

        assert data["answer"]
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_rag_sources_structure(self, client):
        """Источники содержат правильную структуру"""
        response = client.post("/rag", json={
            "query":           "тест",
            "include_sources": True,
        })
        data    = response.json()
        sources = data["sources"]

        assert isinstance(sources, list)

        for source in sources:
            assert "text"   in source
            assert "source" in source
            assert "score"  in source
            assert isinstance(source["score"], float)
            assert 0.0 <= source["score"] <= 1.0

    def test_rag_without_sources(self, client):
        """include_sources=False — список источников пустой"""
        response = client.post("/rag", json={
            "query":           "тест",
            "include_sources": False,
        })
        data = response.json()

        assert data["sources"] == []

    def test_rag_sources_used_count(self, client):
        """sources_used соответствует количеству найденных документов"""
        response = client.post("/rag", json={"query": "тест"})
        data     = response.json()

        # Мок возвращает 2 документа
        assert data["sources_used"] == 2

    def test_rag_latency_is_positive(self, client):
        """latency_ms >= 0"""
        response = client.post("/rag", json={"query": "тест"})
        data     = response.json()

        assert data["latency_ms"] >= 0

    def test_rag_custom_top_k(self, client, rag_mock):
        """top_k передаётся в сервис поиска"""
        rag_mock.search.reset_mock()

        client.post("/rag", json={
            "query": "тест",
            "top_k": 3,
        })

        rag_mock.search.assert_called_once()
        # Второй аргумент вызова search — top_k
        call_args = rag_mock.search.call_args
        assert call_args[0][1] == 3

    def test_rag_search_was_called(self, client, rag_mock):
        """RAG сервис действительно вызвал поиск"""
        rag_mock.search.reset_mock()

        client.post("/rag", json={"query": "тест поиска"})

        assert rag_mock.search.called

    def test_rag_llm_was_called(self, client, llm_mock):
        """LLM генерация вызвалась после поиска"""
        llm_mock.predict_rag.reset_mock()

        client.post("/rag", json={"query": "тест генерации"})

        assert llm_mock.predict_rag.called


class TestRagValidation:

    def test_empty_query_returns_400(self, client):
        """Пустой запрос → 400"""
        response = client.post("/rag", json={"query": "   "})
        assert response.status_code == 400

    def test_short_query_returns_400(self, client):
        """Слишком короткий запрос → 400"""
        response = client.post("/rag", json={"query": "а"})
        assert response.status_code == 400

    def test_missing_query_returns_400(self, client):
        """Нет поля query → 400"""
        response = client.post("/rag", json={"top_k": 3})
        assert response.status_code == 400

    def test_top_k_out_of_range_returns_400(self, client):
        """top_k > 20 → 400"""
        response = client.post("/rag", json={
            "query": "тест",
            "top_k": 100,
        })
        assert response.status_code == 400

    def test_temperature_out_of_range_returns_400(self, client):
        """temperature > 2.0 → 400"""
        response = client.post("/rag", json={
            "query":       "тест",
            "temperature": 3.0,
        })
        assert response.status_code == 400

    def test_error_response_structure(self, client):
        """Ответ с ошибкой содержит нужные поля"""
        response = client.post("/rag", json={"query": "а"})
        data     = response.json()

        assert "error"       in data
        assert "status_code" in data
        assert data["status_code"] == 400


class TestRagSearch:

    def test_search_returns_200(self, client):
        response = client.get("/rag/search?query=тест")
        assert response.status_code == 200

    def test_search_response_structure(self, client):
        response = client.get("/rag/search?query=аккредитация")
        data     = response.json()

        assert "query"   in data
        assert "results" in data
        assert "count"   in data
        assert isinstance(data["results"], list)
        assert data["count"] == len(data["results"])

    def test_search_empty_query_returns_400(self, client):
        """Пустой query → 400 (валидация в роутере)"""
        response = client.get("/rag/search?query=")
        assert response.status_code == 400

    def test_search_query_in_response(self, client):
        """Запрос возвращается в ответе"""
        response = client.get("/rag/search?query=аккредитация")
        data     = response.json()

        assert data["query"] == "аккредитация"


class TestRagStats:

    def test_stats_returns_200(self, client):
        response = client.get("/rag/stats")
        assert response.status_code == 200

    def test_stats_shows_loaded(self, client):
        response = client.get("/rag/stats")
        data     = response.json()
        assert data["status"] == "loaded"

    def test_stats_has_document_count(self, client):
        response = client.get("/rag/stats")
        data     = response.json()
        assert "document_count" in data
        assert data["document_count"] > 0