#!/usr/bin/env python3
"""
Скрипт индексации документов в векторную базу Chroma.

Использование:
    python scripts/index_documents.py
    python scripts/index_documents.py --path ./documents
    python scripts/index_documents.py --path ./documents --clear
    python scripts/index_documents.py --path ./documents --chunk-size 500
"""

import sys
import shutil
import argparse
from pathlib import Path

# Чтобы импорты из app/ работали при запуске из корня проекта
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.logger import setup_logging
from app.config import settings

logger = setup_logging()


# ==================== Загрузка документов ====================

def load_documents(directory: str) -> list:
    """Загрузка документов из директории (txt, md, pdf, docx)"""
    from langchain_community.document_loaders import (
        DirectoryLoader,
        TextLoader,
        PyPDFLoader,
        Docx2txtLoader,
    )

    dir_path = Path(directory)

    if not dir_path.exists():
        logger.error("Директория не существует: %s", directory)
        return []

    loaders_config = [
        ("**/*.txt",  TextLoader,     {"encoding": "utf-8"}),
        ("**/*.md",   TextLoader,     {"encoding": "utf-8"}),
        ("**/*.pdf",  PyPDFLoader,    {}),
        ("**/*.docx", Docx2txtLoader, {}),
    ]

    all_documents = []

    for glob_pattern, loader_cls, loader_kwargs in loaders_config:
        try:
            loader = DirectoryLoader(
                directory,
                glob=glob_pattern,
                loader_cls=loader_cls,
                loader_kwargs=loader_kwargs,
                show_progress=True,
                use_multithreading=True,
            )
            docs = loader.load()

            if docs:
                logger.info(
                    "  [%s] загружено документов: %d",
                    glob_pattern, len(docs),
                )
                all_documents.extend(docs)
            else:
                logger.debug("  [%s] файлов не найдено", glob_pattern)

        except Exception as e:
            logger.warning("  [%s] ошибка загрузки: %s", glob_pattern, e)

    return all_documents


# ==================== Разбиение на чанки ====================

def split_documents(
    documents:     list,
    chunk_size:    int,
    chunk_overlap: int,
) -> list:
    """Разбиение документов на чанки"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    logger.info(
        "Разбито на %d чанков (chunk_size=%d, overlap=%d)",
        len(chunks), chunk_size, chunk_overlap,
    )

    return chunks


# ==================== Создание векторной базы ====================

def create_vectorstore(chunks: list, db_path: str):
    """Создание и сохранение векторной базы Chroma"""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    from langchain_community.vectorstores import Chroma

    logger.info("Загрузка модели эмбеддингов: %s", settings.embedding_model)

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Создание векторной базы...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name=settings.collection_name,
    )

    return vectorstore


# ==================== Тестовый поиск ====================

def test_search(vectorstore, query: str = "тест", top_k: int = 3) -> None:
    """Проверка что база работает — делаем тестовый запрос"""
    logger.info("Тестовый поиск по запросу: '%s'", query)

    try:
        results = vectorstore.similarity_search_with_relevance_scores(query, k=top_k)

        if not results:
            logger.warning("Тестовый поиск не вернул результатов")
            return

        for i, (doc, score) in enumerate(results, 1):
            source  = Path(doc.metadata.get("source", "?")).name
            preview = doc.page_content[:120].replace("\n", " ").strip()
            logger.info(
                "  %d. [score=%.3f] %s\n     %s...",
                i, score, source, preview,
            )

    except Exception as e:
        logger.warning("Тестовый поиск завершился с ошибкой: %s", e)


# ==================== Статистика чанков ====================

def print_chunks_stats(chunks: list) -> None:
    """Вывод статистики по чанкам"""
    if not chunks:
        return

    lengths = [len(c.page_content) for c in chunks]
    avg_len = sum(lengths) / len(lengths)
    min_len = min(lengths)
    max_len = max(lengths)

    # Статистика по источникам
    sources: dict[str, int] = {}
    for chunk in chunks:
        source = Path(chunk.metadata.get("source", "unknown")).name
        sources[source] = sources.get(source, 0) + 1

    logger.info("Статистика чанков:")
    logger.info("  Всего чанков : %d",     len(chunks))
    logger.info("  Средний размер: %.0f символов", avg_len)
    logger.info("  Мин / Макс   : %d / %d символов", min_len, max_len)
    logger.info("  Источников   : %d файлов", len(sources))

    # Топ-5 файлов по количеству чанков
    top_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info("  Топ файлов по чанкам:")
    for name, count in top_sources:
        logger.info("    - %s: %d чанков", name, count)


# ==================== Пример чанка ====================

def print_sample_chunk(chunks: list) -> None:
    """Показать пример первого чанка"""
    if not chunks:
        return

    sample  = chunks[0]
    source  = Path(sample.metadata.get("source", "?")).name
    preview = sample.page_content[:300]

    logger.info("Пример чанка [%s]:", source)
    logger.info("-" * 50)
    logger.info(preview + ("..." if len(sample.page_content) > 300 else ""))
    logger.info("-" * 50)


# ==================== Main ====================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Индексация документов в векторную базу Chroma",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python scripts/index_documents.py
  python scripts/index_documents.py --path ./my_docs
  python scripts/index_documents.py --clear
  python scripts/index_documents.py --chunk-size 500 --chunk-overlap 100
  python scripts/index_documents.py --test-query "как получить аккредитацию"
        """,
    )

    parser.add_argument(
        "--path", "-p",
        default=settings.documents_path,
        help=f"Путь к документам (по умолчанию: {settings.documents_path})",
    )
    parser.add_argument(
        "--db-path", "-d",
        default=settings.chroma_db_path,
        help=f"Путь к базе Chroma (по умолчанию: {settings.chroma_db_path})",
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=settings.chunk_size,
        help=f"Размер чанка в символах (по умолчанию: {settings.chunk_size})",
    )
    parser.add_argument(
        "--chunk-overlap", "-o",
        type=int,
        default=settings.chunk_overlap,
        help=f"Перекрытие чанков (по умолчанию: {settings.chunk_overlap})",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Очистить существующую базу перед индексацией",
    )
    parser.add_argument(
        "--test-query",
        default="тест",
        help="Запрос для тестового поиска после индексации",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Пропустить тестовый поиск",
    )

    args = parser.parse_args()

    # ── Проверка директории с документами ───────────────────────
    docs_path = Path(args.path)
    if not docs_path.exists():
        logger.error("Директория не найдена: %s", args.path)
        logger.info("Создайте директорию и положите туда документы:")
        logger.info("  mkdir %s", args.path)
        sys.exit(1)

    # Ищем файлы поддерживаемых форматов
    supported = {".txt", ".md", ".pdf", ".docx"}
    files = [f for f in docs_path.rglob("*") if f.suffix.lower() in supported]

    if not files:
        logger.error(
            "В директории %s нет поддерживаемых файлов (%s)",
            args.path, ", ".join(supported),
        )
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Найдено файлов: %d", len(files))
    # Показываем первые 10
    for f in files[:10]:
        logger.info("  - %s", f.name)
    if len(files) > 10:
        logger.info("  ... и ещё %d файлов", len(files) - 10)

    # ── Очистка базы ─────────────────────────────────────────────
    db_path = Path(args.db_path)
    if args.clear and db_path.exists():
        logger.warning("Удаление существующей базы: %s", args.db_path)
        shutil.rmtree(db_path)
        logger.info("База удалена")

    # ── Загрузка документов ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Загрузка документов из: %s", args.path)

    documents = load_documents(args.path)

    if not documents:
        logger.error("Не удалось загрузить ни одного документа")
        sys.exit(1)

    logger.info("Загружено документов (страниц/секций): %d", len(documents))

    # ── Разбиение на чанки ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Разбиение на чанки...")

    chunks = split_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if not chunks:
        logger.error("Разбиение не дало чанков — проверьте документы")
        sys.exit(1)

    print_chunks_stats(chunks)
    print_sample_chunk(chunks)

    # ── Создание векторной базы ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("Создание векторной базы в: %s", args.db_path)

    try:
        vectorstore = create_vectorstore(chunks, args.db_path)
    except Exception:
        logger.exception("Ошибка создания векторной базы")
        sys.exit(1)

    # ── Итог ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    count = vectorstore._collection.count()
    logger.info("✅ Индексация завершена!")
    logger.info("   Документов в базе : %d", count)
    logger.info("   Путь к базе       : %s", Path(args.db_path).resolve())

    # ── Тестовый поиск ───────────────────────────────────────────
    if not args.no_test:
        logger.info("=" * 60)
        test_search(vectorstore, query=args.test_query)

    logger.info("=" * 60)
    logger.info("Готово! Запустите сервер: python main.py")


if __name__ == "__main__":
    main()