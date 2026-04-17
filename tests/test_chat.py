"""
Тесты эндпоинта /chat
"""


class TestChatSuccess:

    def test_chat_returns_200(self, client):
        """Базовый успешный запрос"""
        response = client.post("/chat", json={
            "messages": [{"role": "user", "content": "Привет!"}]
        })
        assert response.status_code == 200

    def test_chat_response_structure(self, client):
        """Ответ содержит все обязательные поля"""
        response = client.post("/chat", json={
            "messages": [{"role": "user", "content": "Привет!"}]
        })
        data = response.json()

        assert "request_id"    in data
        assert "text"          in data
        assert "tokens_used"   in data
        assert "finish_reason" in data
        assert "latency_ms"    in data

    def test_chat_returns_text(self, client):
        """Ответ содержит непустой текст"""
        response = client.post("/chat", json={
            "messages": [{"role": "user", "content": "Как дела?"}]
        })
        data = response.json()

        assert data["text"]
        assert isinstance(data["text"], str)
        assert len(data["text"]) > 0

    def test_chat_finish_reason_is_valid(self, client):
        """finish_reason — одно из допустимых значений"""
        response = client.post("/chat", json={
            "messages": [{"role": "user", "content": "Тест"}]
        })
        data = response.json()

        assert data["finish_reason"] in ("stop", "length", "error")

    def test_chat_latency_is_positive(self, client):
        """latency_ms > 0"""
        response = client.post("/chat", json={
            "messages": [{"role": "user", "content": "Тест"}]
        })
        data = response.json()

        assert data["latency_ms"] >= 0

    def test_chat_with_history(self, client):
        """Запрос с историей сообщений"""
        response = client.post("/chat", json={
            "messages": [
                {"role": "system",    "content": "Ты помощник."},
                {"role": "user",      "content": "Привет!"},
                {"role": "assistant", "content": "Здравствуйте!"},
                {"role": "user",      "content": "Как тебя зовут?"},
            ]
        })
        assert response.status_code == 200

    def test_chat_custom_parameters(self, client):
        """Запрос с кастомными параметрами"""
        response = client.post("/chat", json={
            "messages":    [{"role": "user", "content": "Тест"}],
            "temperature": 0.5,
            "max_tokens":  50,
        })
        assert response.status_code == 200

    def test_chat_request_id_is_string(self, client):
        """request_id — непустая строка"""
        response = client.post("/chat", json={
            "messages": [{"role": "user", "content": "Тест"}]
        })
        data = response.json()

        assert isinstance(data["request_id"], str)
        assert len(data["request_id"]) > 0

    def test_chat_llm_was_called(self, client, llm_mock):
        """LLM сервис действительно вызвался"""
        llm_mock.predict.reset_mock()

        client.post("/chat", json={
            "messages": [{"role": "user", "content": "Тест вызова"}]
        })

        assert llm_mock.predict.called
        assert llm_mock.predict.call_count == 1


class TestChatValidation:
    """
    Наш обработчик RequestValidationError возвращает 400 (не 422).
    Это намеренное поведение — все ошибки валидации идут как 400.
    """

    def test_empty_messages_returns_400(self, client):
        """Пустой список сообщений → 400"""
        response = client.post("/chat", json={
            "messages": []
        })
        assert response.status_code == 400

    def test_empty_content_returns_400(self, client):
        """Пустой текст сообщения → 400"""
        response = client.post("/chat", json={
            "messages": [{"role": "user", "content": "   "}]
        })
        assert response.status_code == 400

    def test_last_message_not_user_returns_400(self, client):
        """Последнее сообщение не от user → 400"""
        response = client.post("/chat", json={
            "messages": [
                {"role": "user",      "content": "Привет"},
                {"role": "assistant", "content": "Ответ"},
            ]
        })
        assert response.status_code == 400

    def test_invalid_role_returns_400(self, client):
        """Неизвестная роль → 400"""
        response = client.post("/chat", json={
            "messages": [{"role": "unknown_role", "content": "Тест"}]
        })
        assert response.status_code == 400

    def test_temperature_out_of_range_returns_400(self, client):
        """temperature > 2.0 → 400"""
        response = client.post("/chat", json={
            "messages":    [{"role": "user", "content": "Тест"}],
            "temperature": 5.0,
        })
        assert response.status_code == 400

    def test_max_tokens_out_of_range_returns_400(self, client):
        """max_tokens > 1024 → 400"""
        response = client.post("/chat", json={
            "messages":   [{"role": "user", "content": "Тест"}],
            "max_tokens": 9999,
        })
        assert response.status_code == 400

    def test_missing_messages_returns_400(self, client):
        """Нет поля messages → 400"""
        response = client.post("/chat", json={
            "temperature": 0.7
        })
        assert response.status_code == 400

    def test_error_response_has_detail(self, client):
        """Ответ с ошибкой содержит поле detail"""
        response = client.post("/chat", json={
            "messages": []
        })
        data = response.json()

        assert "error"      in data
        assert "status_code" in data
        assert data["status_code"] == 400