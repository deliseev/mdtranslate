"""Инкрементальный перевод зеркалируемого дерева документации."""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Protocol, Sequence

import tomllib

GIT_TIMEOUT_SECONDS = 60
HTTP_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class SourceConfig:
    """Откуда берётся оригинал и какие файлы вообще подлежат переводу."""

    ref: str
    include: list[str]
    exclude_paths: list[str]
    exclude_files: list[str]


@dataclass(frozen=True)
class TargetConfig:
    """Куда кладётся перевод и как оформляется pull request."""

    branch: str
    language: str
    branch_prefix: str
    pr_label: str
    pr_title: str
    pr_body_template: str
    review_comment_template: str = "**Оригинал:**\n\n{source}"
    max_review_comments: int = 300


@dataclass(frozen=True)
class StateConfig:
    """Пути к файлам, хранящим прогресс между прогонами."""

    sync_file: str
    pending_file: str


@dataclass(frozen=True)
class PromptConfig:
    """Шаблон промпта и параметры формирования запроса к модели."""

    template: str
    glossary: dict[str, str]
    context_mode: str
    window_blocks: int
    max_request_chars: int


@dataclass(frozen=True)
class MemoryConfig:
    """Память переводов: где лежит и насколько придирчива.

    Пустой `file` полностью выключает память — инструмент ведёт себя как раньше.
    """

    file: str = ""
    fuzzy_threshold: float = 0.8
    min_length_ratio: float = 0.4
    max_length_ratio: float = 2.6

    @property
    def enabled(self) -> bool:
        """Включена ли память."""
        return bool(self.file)


@dataclass(frozen=True)
class ProviderConfig:
    """Описание одного провайдера перевода."""

    name: str
    model: str
    api_key_env: str
    base_url: str = ""


@dataclass(frozen=True)
class Config:
    """Полная конфигурация инструмента."""

    source: SourceConfig
    target: TargetConfig
    state: StateConfig
    prompt: PromptConfig
    providers: list[ProviderConfig]
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    def replace(self, **changes: Any) -> "Config":
        """Возвращает копию конфига с изменёнными полями."""
        return dataclasses.replace(self, **changes)


def load_config(path: str) -> Config:
    """Читает TOML-конфиг и разбирает его в типизированные объекты."""
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)

    prompt = raw.get("prompt", {})
    return Config(
        source=SourceConfig(**raw["source"]),
        target=TargetConfig(**raw["target"]),
        state=StateConfig(**raw["state"]),
        prompt=PromptConfig(
            template=prompt["template"],
            glossary=prompt.get("glossary", {}),
            context_mode=prompt.get("context_mode", "full"),
            window_blocks=int(prompt.get("window_blocks", 20)),
            max_request_chars=int(prompt.get("max_request_chars", 200_000)),
        ),
        providers=[ProviderConfig(**p) for p in raw.get("providers", [])],
        memory=MemoryConfig(**raw.get("memory", {})),
    )


_COMMIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
_FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
_HEADING_RE = re.compile(r"^(#{1,6})\s")


@dataclass(frozen=True)
class Block:
    """Один блок markdown: текст плюс отделяющие его от следующего пустые строки."""

    text: str
    sep: str

    @property
    def is_code(self) -> bool:
        """Является ли блок огороженным блоком кода."""
        return bool(_FENCE_RE.match(self.text))

    def signature(self) -> tuple:
        """Не зависящий от языка отпечаток — по нему сопоставляются оригинал и перевод."""
        if self.is_code:
            return ("code", self.text)
        heading = _HEADING_RE.match(self.text)
        if heading:
            return ("h", len(heading.group(1)))
        stripped = self.text.lstrip()
        if stripped.startswith(">"):
            return ("quote",)
        if re.match(r"^\s*([-*+]|\d+\.)\s", self.text):
            return ("list", self.text.count("\n"))
        return ("para",)


@dataclass(frozen=True)
class Document:
    """Документ, разобранный на блоки, с сохранением исходного форматирования."""

    prefix: str
    blocks: tuple[Block, ...]

    def render(self) -> str:
        """Собирает документ обратно в текст."""
        return self.prefix + "".join(b.text + b.sep for b in self.blocks)


class MarkdownSplitter:
    """Режет markdown на блоки так, что join(split(x)) == x побайтово."""

    def split(self, text: str) -> Document:
        """Разбирает текст на блоки, не теряя ни одного разделителя."""
        lines = text.splitlines(keepends=True)
        index = 0

        prefix_parts: list[str] = []
        while index < len(lines) and lines[index].strip() == "":
            prefix_parts.append(lines[index])
            index += 1

        blocks: list[Block] = []
        while index < len(lines):
            body: list[str] = []
            fence: str | None = None
            while index < len(lines):
                line = lines[index]
                match = _FENCE_RE.match(line)
                if fence is None:
                    if match:
                        fence = match.group(1)[:3]
                        body.append(line)
                        index += 1
                        continue
                    if line.strip() == "":
                        break
                    body.append(line)
                    index += 1
                else:
                    body.append(line)
                    index += 1
                    if len(body) > 1 and re.match(r"^\s*" + re.escape(fence), line):
                        break

            separator_parts: list[str] = []
            while index < len(lines) and lines[index].strip() == "":
                separator_parts.append(lines[index])
                index += 1

            raw = "".join(body)
            separator = "".join(separator_parts)
            if raw.endswith("\n"):  # перевод строки принадлежит разделителю
                raw, separator = raw[:-1], "\n" + separator
            blocks.append(Block(text=raw, sep=separator))

        return Document(prefix="".join(prefix_parts), blocks=tuple(blocks))


@dataclass(frozen=True)
class Alignment:
    """Соответствие блоков оригинала блокам перевода."""

    mapping: dict[int, int]
    confident: bool


class BlockAligner:
    """Сопоставляет блоки оригинала и перевода, опираясь на структуру документа."""

    def __init__(self, min_ratio: float = 0.9) -> None:
        self.min_ratio = min_ratio

    def align(self, source: Sequence[Block], translated: Sequence[Block]) -> Alignment:
        """Строит соответствие блоков и оценивает, можно ли ему доверять."""
        if not source and not translated:
            return Alignment(mapping={}, confident=True)

        if len(source) == len(translated):
            code_ok = all(
                translated[i].is_code == block.is_code for i, block in enumerate(source)
            )
            if code_ok:
                return Alignment(
                    mapping={i: i for i in range(len(source))}, confident=True
                )

        left = [b.signature() for b in source]
        right = [b.signature() for b in translated]
        matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
        mapping: dict[int, int] = {}
        for block in matcher.get_matching_blocks():
            for offset in range(block.size):
                mapping[block.a + offset] = block.b + offset
        ratio = len(mapping) / max(len(source), 1)
        return Alignment(mapping=mapping, confident=ratio >= self.min_ratio)


class SegmentCodec:
    """Оборачивает переводимые куски в якоря, которые модель обязана вернуть."""

    def wrap(self, segment_id: int, text: str) -> str:
        """Обрамляет текст якорями с указанным номером."""
        return f"⟦S{segment_id}⟧\n{text}\n⟦/S{segment_id}⟧"

    def parse(self, response: str) -> dict[int, str]:
        """Достаёт из ответа модели переводы по номерам якорей."""
        found: dict[int, str] = {}
        for match in re.finditer(
            r"⟦S(\d+)⟧\n?(.*?)\n?⟦/S\1⟧", response, flags=re.DOTALL
        ):
            found[int(match.group(1))] = match.group(2)
        return found


@dataclass
class PlanItem:
    """Элемент будущего файла: либо готовый перевод, либо текст на перевод."""

    kind: str  # "keep" — оставить как есть, "translate" — отправить модели
    text: str
    sep: str
    source_index: int = -1
    # Похожая пара из памяти: модель чинит старый перевод вместо перевода с нуля,
    # поэтому ручная вычитка остальной части блока переживает правку оригинала.
    hint: "MemoryEntry | None" = None


@dataclass
class FilePlan:
    """План пересборки одного файла."""

    path: str
    prefix: str
    items: list[PlanItem]
    source_document: Document
    existing_translation: str

    @property
    def translatable(self) -> list[PlanItem]:
        """Элементы, которые нужно отправить на перевод."""
        return [i for i in self.items if i.kind == "translate"]

    def cost(self) -> int:
        """Оценка размера запроса для этого файла в символах."""
        return len(self.source_document.render()) + len(self.existing_translation)


class IncrementalMerger:
    """Строит план файла: что переводить заново, а что взять из старого перевода."""

    def __init__(self, aligner: BlockAligner) -> None:
        self.aligner = aligner

    def plan(
        self,
        path: str,
        base: Document,
        head: Document,
        translated: Document,
    ) -> FilePlan | None:
        """Возвращает план или None, если перевод не удалось надёжно сопоставить."""
        alignment = self.aligner.align(base.blocks, translated.blocks)
        if not alignment.confident:
            return None

        items: list[PlanItem] = []
        matcher = difflib.SequenceMatcher(
            a=[b.text for b in base.blocks],
            b=[b.text for b in head.blocks],
            autojunk=False,
        )
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for offset in range(i2 - i1):
                    base_index = i1 + offset
                    head_index = j1 + offset
                    target = alignment.mapping.get(base_index)
                    if target is not None and target < len(translated.blocks):
                        existing = translated.blocks[target]
                        items.append(
                            PlanItem(kind="keep", text=existing.text, sep=existing.sep)
                        )
                    else:
                        block = head.blocks[head_index]
                        items.append(
                            PlanItem(
                                kind="translate",
                                text=block.text,
                                sep=block.sep,
                                source_index=head_index,
                            )
                        )
            elif tag == "delete":
                continue
            else:  # replace / insert
                for head_index in range(j1, j2):
                    block = head.blocks[head_index]
                    if block.is_code:
                        # Код не переводится — переносим его как есть.
                        items.append(
                            PlanItem(kind="keep", text=block.text, sep=block.sep)
                        )
                    else:
                        items.append(
                            PlanItem(
                                kind="translate",
                                text=block.text,
                                sep=block.sep,
                                source_index=head_index,
                            )
                        )

        prefix = translated.prefix if translated.blocks else head.prefix
        return FilePlan(
            path=path,
            prefix=prefix,
            items=items,
            source_document=head,
            existing_translation=translated.render(),
        )


class BatchPlanner:
    """Жадно пакует работу по файлам в запросы под бюджет символов."""

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars

    def plan(self, items: Iterable[tuple[str, int]]) -> list[list[tuple[str, int]]]:
        """Группирует пары (путь, размер) в батчи, не превышающие бюджет."""
        batches: list[list[tuple[str, int]]] = []
        current: list[tuple[str, int]] = []
        used = 0
        for path, cost in items:
            if cost >= self.max_chars:
                batches.append([(path, cost)])
                continue
            if current and used + cost > self.max_chars:
                batches.append(current)
                current, used = [], 0
            current.append((path, cost))
            used += cost
        if current:
            batches.append(current)
        return batches


@dataclass(frozen=True)
class MemoryEntry:
    """Пара «оригинал — перевод» из памяти переводов."""

    source: str
    translation: str


class TranslationMemory:
    """База переводов, ключ — хеш исходного текста.

    Неверная пара тут хуже отсутствующей: она подставится молча и навсегда,
    поэтому запись строже чтения. Отношение длин перевода к оригиналу
    проверяется по границам, замеренным на реальном корпусе.
    """

    def __init__(
        self,
        entries: dict[str, MemoryEntry] | None = None,
        min_length_ratio: float = 0.4,
        max_length_ratio: float = 2.6,
    ) -> None:
        self._entries: dict[str, MemoryEntry] = dict(entries or {})
        self.min_length_ratio = min_length_ratio
        self.max_length_ratio = max_length_ratio

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def key(source: str) -> str:
        """Ключ записи — усечённый sha256 исходного текста."""
        return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

    def lookup(self, source: str) -> str | None:
        """Готовый перевод для точно такого же исходного текста."""
        entry = self._entries.get(self.key(source))
        return entry.translation if entry else None

    def add(self, source: str, translation: str) -> bool:
        """Запоминает пару. Возвращает False, если пара отвергнута как негодная."""
        if not source.strip() or not translation.strip():
            return False
        ratio = len(translation) / len(source)
        if not self.min_length_ratio <= ratio <= self.max_length_ratio:
            return False
        self._entries[self.key(source)] = MemoryEntry(source, translation)
        return True

    def nearest(self, source: str, cutoff: float) -> MemoryEntry | None:
        """Самая похожая запись, если сходство не ниже порога.

        Отбор по длине идёт первым: он отбрасывает подавляющее большинство
        кандидатов, не запуская дорогое посимвольное сравнение.
        """
        best: MemoryEntry | None = None
        best_ratio = cutoff
        for entry in self._entries.values():
            longest = max(len(entry.source), len(source))
            if not longest or abs(len(entry.source) - len(source)) / longest > 1 - cutoff:
                continue
            ratio = difflib.SequenceMatcher(None, source, entry.source).ratio()
            if ratio >= best_ratio:
                best, best_ratio = entry, ratio
        return best

    @classmethod
    def load(cls, fs: "FileSystem", path: str, **limits: float) -> "TranslationMemory":
        """Читает базу из JSONL; отсутствие файла — это просто пустая база."""
        memory = cls(**limits)
        if not fs.exists(path):
            return memory
        for line in fs.read(path).splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                memory._entries[row["h"]] = MemoryEntry(row["s"], row["t"])
            except (ValueError, KeyError):
                print(f"Пропускаю битую строку в {path}: {line[:60]!r}")
        return memory

    def save(self, fs: "FileSystem", path: str) -> None:
        """Пишет базу построчно и с сортировкой — чтобы диффы были минимальными."""
        lines = [
            json.dumps(
                {"h": key, "s": entry.source, "t": entry.translation},
                ensure_ascii=False,
            )
            for key, entry in sorted(self._entries.items())
        ]
        fs.write(path, "\n".join(lines) + ("\n" if lines else ""))


class QuotaExhausted(Exception):
    """Квота провайдера исчерпана; другой провайдер ещё может сработать."""


class ProviderError(Exception):
    """Любая другая ошибка на стороне провайдера."""


@dataclass(frozen=True)
class TranslationRequest:
    """Готовый запрос к модели: текст промпта и ожидаемые сегменты."""

    prompt: str
    segments: dict[int, str]


class TranslationProvider(Protocol):
    """Провайдер перевода."""

    def translate(self, request: TranslationRequest) -> str: ...


class GeminiProvider:
    """Провайдер поверх Google Gemini."""

    def __init__(self, config: ProviderConfig) -> None:
        from google import genai
        from google.genai import types

        api_key = _require_env(config.api_key_env)
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=HTTP_TIMEOUT_SECONDS * 1000),
        )
        self._model = config.model

    def translate(self, request: TranslationRequest) -> str:
        """Отправляет промпт в Gemini и возвращает сырой ответ модели."""
        from google.genai import errors

        try:
            response = self._client.models.generate_content(
                model=self._model, contents=request.prompt
            )
        except errors.ClientError as exc:
            if getattr(exc, "code", None) == 429 and (
                getattr(exc, "status", "") == "RESOURCE_EXHAUSTED"
            ):
                raise QuotaExhausted(str(exc)) from exc
            raise ProviderError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 — SDK бросает что угодно
            raise ProviderError(str(exc)) from exc
        return response.text or ""


class OpenAICompatibleProvider:
    """Работает с любым endpoint /chat/completions (OpenAI, OpenRouter, Ollama…)."""

    def __init__(self, config: ProviderConfig) -> None:
        self._url = (config.base_url or "https://api.openai.com/v1").rstrip("/")
        self._url += "/chat/completions"
        self._model = config.model
        self._key = _require_env(config.api_key_env)

    def translate(self, request: TranslationRequest) -> str:
        """Отправляет промпт по HTTP и возвращает содержимое ответа модели."""
        payload = json.dumps(
            {
                "model": self._model,
                "messages": [{"role": "user", "content": request.prompt}],
            }
        ).encode("utf-8")
        http_request = urllib.request.Request(
            self._url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key}",
            },
        )
        try:
            with urllib.request.urlopen(
                http_request, timeout=HTTP_TIMEOUT_SECONDS
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise QuotaExhausted(f"{exc.code} {exc.reason}") from exc
            raise ProviderError(f"{exc.code} {exc.reason}") from exc
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(str(exc)) from exc
        return body["choices"][0]["message"]["content"]


PROVIDERS: dict[str, Any] = {
    "gemini": GeminiProvider,
    "openai": OpenAICompatibleProvider,
}


def _require_env(name: str) -> str:
    """Читает обязательную переменную окружения."""
    value = os.environ.get(name)
    if not value:
        raise ProviderError(f"переменная окружения {name} не задана")
    return value


def build_translators(config: Config) -> list[Callable[[TranslationRequest], str]]:
    """Собирает цепочку провайдеров, пропуская тех, у кого нет ключа."""
    translators: list[Callable[[TranslationRequest], str]] = []
    for provider_config in config.providers:
        factory = PROVIDERS.get(provider_config.name)
        if factory is None:
            raise ProviderError(f"неизвестный провайдер: {provider_config.name}")
        if not os.environ.get(provider_config.api_key_env):
            print(
                f"Провайдер {provider_config.name} пропущен: "
                f"переменная {provider_config.api_key_env} не задана."
            )
            continue
        translators.append(factory(provider_config).translate)
    if not translators:
        raise ProviderError("не настроено ни одного пригодного провайдера")
    return translators


def chain_translate(
    translators: Sequence[Callable[[TranslationRequest], str]],
    request: TranslationRequest,
) -> str:
    """Переключается на следующего провайдера, когда у текущего кончилась квота."""
    last: QuotaExhausted | None = None
    for translate in translators:
        try:
            return translate(request)
        except QuotaExhausted as exc:
            last = exc
            print(f"Квота провайдера исчерпана, пробуем следующего: {exc}")
    raise last if last else ProviderError("нет доступных провайдеров")


class GitClient(Protocol):
    """Операции с git, нужные пайплайну."""

    def rev_parse(self, ref: str) -> str: ...
    def show(self, ref: str, path: str) -> str: ...
    def diff_name_status(
        self, base: str, head: str, patterns: Sequence[str]
    ) -> list[tuple[str, str]]: ...
    def create_branch(self, name: str) -> None: ...
    def commit(self, paths: Sequence[str], message: str) -> None: ...
    def push(self, branch: str) -> None: ...


class FileSystem(Protocol):
    """Доступ к файлам рабочего дерева."""

    def read(self, path: str) -> str: ...
    def write(self, path: str, text: str) -> None: ...
    def exists(self, path: str) -> bool: ...
    def remove(self, path: str) -> None: ...


@dataclass(frozen=True)
class ReviewComment:
    """Оригинал переведённого куска, привязанный к строке в диффе."""

    path: str
    line: int
    body: str


class PullRequestClient(Protocol):
    """Работа с pull request'ами."""

    def create_draft(
        self, branch: str, base: str, title: str, body: str, label: str
    ) -> int: ...
    def mark_ready(self, number: int) -> None: ...
    def update_body(self, number: int, body: str) -> None: ...
    def add_review(
        self, number: int, commit_id: str, comments: Sequence[ReviewComment]
    ) -> None: ...


class Clock(Protocol):
    """Источник времени для имён веток."""

    def stamp(self) -> str: ...


class SubprocessGit:
    """GitClient поверх настоящего git."""

    def _run(self, *args: str, quiet: bool = False) -> str:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            stderr=subprocess.DEVNULL if quiet else None,
        )

    def rev_parse(self, ref: str) -> str:
        """Разрешает ссылку в хеш коммита."""
        return self._run("rev-parse", ref).strip()

    def show(self, ref: str, path: str) -> str:
        """Читает содержимое файла на указанной ревизии."""
        try:
            # Отсутствие файла на ревизии — штатный случай (файл только добавлен),
            # поэтому ругань git в stderr здесь только зашумляет лог.
            return self._run("show", f"{ref}:{path}", quiet=True)
        except subprocess.CalledProcessError as exc:
            raise FileNotFoundError(f"{ref}:{path}") from exc

    def diff_name_status(
        self, base: str, head: str, patterns: Sequence[str]
    ) -> list[tuple[str, str]]:
        """Возвращает пары (статус, путь) для изменившихся файлов."""
        output = self._run(
            "diff", "--name-status", "--diff-filter=AMD", base, head, "--", *patterns
        )
        rows: list[tuple[str, str]] = []
        for line in output.splitlines():
            if "\t" in line:
                status, path = line.split("\t", 1)
                rows.append((status[:1], path))
        return rows

    def create_branch(self, name: str) -> None:
        """Создаёт ветку и переключается на неё."""
        self._run("checkout", "-b", name)

    def commit(self, paths: Sequence[str], message: str) -> None:
        """Индексирует указанные пути и коммитит, если в индексе что-то есть.

        Пустой коммит git отвергает кодом 1, и это уронило бы весь прогон:
        так бывает, когда содержимое файла после слияния совпало с прежним,
        а маркер состояния уже был записан предыдущим коммитом.
        """
        self._run("add", "--", *paths)
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            timeout=GIT_TIMEOUT_SECONDS,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        if staged == 0:  # 0 — различий нет, коммитить нечего
            print(f"Нечего коммитить, пропускаю: {message}")
            return
        self._run("commit", "-m", message)

    def push(self, branch: str) -> None:
        """Отправляет ветку в origin."""
        self._run("push", "origin", branch)


class RealFileSystem:
    """FileSystem поверх настоящего диска."""

    def read(self, path: str) -> str:
        """Читает файл целиком."""
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    def write(self, path: str, text: str) -> None:
        """Записывает файл, создавая недостающие каталоги."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)

    def exists(self, path: str) -> bool:
        """Существует ли файл."""
        return os.path.exists(path)

    def remove(self, path: str) -> None:
        """Удаляет файл, если он есть."""
        if os.path.exists(path):
            os.remove(path)


class NoPushGit:
    """Пропускает локальные операции, но не выпускает изменения наружу."""

    def __init__(self, inner: GitClient) -> None:
        self._inner = inner

    def rev_parse(self, ref: str) -> str:
        """Разрешает ссылку в хеш коммита."""
        return self._inner.rev_parse(ref)

    def show(self, ref: str, path: str) -> str:
        """Читает содержимое файла на указанной ревизии."""
        return self._inner.show(ref, path)

    def diff_name_status(
        self, base: str, head: str, patterns: Sequence[str]
    ) -> list[tuple[str, str]]:
        """Возвращает пары (статус, путь) для изменившихся файлов."""
        return self._inner.diff_name_status(base, head, patterns)

    def create_branch(self, name: str) -> None:
        """Создаёт ветку: операция локальная, наружу ничего не уходит."""
        self._inner.create_branch(name)

    def commit(self, paths: Sequence[str], message: str) -> None:
        """Коммитит: операция локальная, наружу ничего не уходит."""
        self._inner.commit(paths, message)

    def push(self, branch: str) -> None:
        """Сообщает о пуше, не выполняя его."""
        print(f"[--no-remote] не отправляю ветку {branch}")


class ReadOnlyGit(NoPushGit):
    """Дополнительно запрещает менять и локальный репозиторий.

    Нужен для полностью инертного прогона: без этого ветка «переводить нечего»
    доходит до публикации и создаёт настоящую ветку прямо в рабочем каталоге.
    """

    def create_branch(self, name: str) -> None:
        """Сообщает о создании ветки, не создавая её."""
        print(f"[--dry-run] создал бы ветку {name}")

    def commit(self, paths: Sequence[str], message: str) -> None:
        """Сообщает о коммите, не создавая его."""
        print(f"[--dry-run] закоммитил бы {len(paths)} файл(ов): {message}")


class ReadOnlyFileSystem:
    """Читает по-настоящему, но записи и удаления только печатает."""

    def __init__(self, inner: FileSystem) -> None:
        self._inner = inner

    def read(self, path: str) -> str:
        """Читает файл через вложенную ФС."""
        return self._inner.read(path)

    def write(self, path: str, text: str) -> None:
        """Сообщает о записи, не трогая диск."""
        print(f"[--dry-run] записал бы {path} ({len(text)} символов)")

    def exists(self, path: str) -> bool:
        """Существует ли файл."""
        return self._inner.exists(path)

    def remove(self, path: str) -> None:
        """Сообщает об удалении, не трогая диск."""
        print(f"[--dry-run] удалил бы {path}")


class GhPullRequests:
    """PullRequestClient поверх GitHub CLI."""

    def _run(self, *args: str) -> str:
        return subprocess.check_output(
            ["gh", *args], text=True, timeout=GIT_TIMEOUT_SECONDS
        )

    def create_draft(
        self, branch: str, base: str, title: str, body: str, label: str
    ) -> int:
        """Открывает черновой pull request и возвращает его номер."""
        output = self._run(
            "pr",
            "create",
            "--draft",
            "--base",
            base,
            "--head",
            branch,
            "--title",
            title,
            "--body",
            body,
            "--label",
            label,
        )
        match = re.search(r"/pull/(\d+)", output)
        return int(match.group(1)) if match else 0

    def mark_ready(self, number: int) -> None:
        """Переводит черновик в готовый к ревью."""
        self._run("pr", "ready", str(number))

    def add_review(
        self, number: int, commit_id: str, comments: Sequence[ReviewComment]
    ) -> None:
        """Публикует ревью с оригиналами, привязанными к строкам перевода."""
        if not comments:
            return
        payload = json.dumps(
            {
                "commit_id": commit_id,
                "event": "COMMENT",
                "comments": [
                    {"path": c.path, "line": c.line, "side": "RIGHT", "body": c.body}
                    for c in comments
                ],
            }
        )
        repo = os.environ.get("GITHUB_REPOSITORY", "")
        endpoint = f"repos/{repo}/pulls/{number}/reviews" if repo else None
        if endpoint is None:
            print("GITHUB_REPOSITORY не задана — ревью с оригиналами пропущено.")
            return
        try:
            subprocess.run(
                ["gh", "api", "--method", "POST", endpoint, "--input", "-"],
                input=payload,
                text=True,
                timeout=GIT_TIMEOUT_SECONDS,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        except Exception as exc:  # noqa: BLE001
            # Ревью — вспомогательная штука: перевод уже закоммичен, и терять
            # прогон из-за отвергнутого комментария нельзя.
            print(f"Не удалось опубликовать ревью с оригиналами: {exc}")

    def update_body(self, number: int, body: str) -> None:
        """Обновляет описание pull request'а."""
        self._run("pr", "edit", str(number), "--body", body)


class SystemClock:
    """Clock поверх системного времени."""

    def stamp(self) -> str:
        """Метка времени для имени ветки."""
        return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


@dataclass
class RunResult:
    """Итоги одного прогона."""

    translated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    quota_exhausted: bool = False

    @property
    def needs_attention(self) -> bool:
        """Требуется ли вмешательство человека.

        Исчерпанная квота — штатный режим: очередь разгребётся сама, когда
        лимит обновится. Красить такой прогон в красное значит сделать
        настоящую аварию неотличимой от обычного дня.
        """
        if self.skipped:
            return True
        return bool(self.pending) and not self.quota_exhausted


class FileFilter:
    """Отбирает файлы, подлежащие переводу, по правилам конфига."""

    def __init__(self, config: SourceConfig) -> None:
        self._config = config

    def accepts(self, path: str) -> bool:
        """Подлежит ли файл переводу."""
        parts = path.split("/")
        if any(segment in parts for segment in self._config.exclude_paths):
            return False
        name = os.path.basename(path).lower()
        if name in {n.lower() for n in self._config.exclude_files}:
            return False
        return any(fnmatch.fnmatch(path, pattern) for pattern in self._config.include)


class TranslationPipeline:
    """Оркестратор: планирует работу, переводит и публикует результат."""

    def __init__(
        self,
        config: Config,
        git: GitClient,
        fs: FileSystem,
        translate: Callable[[TranslationRequest], str],
        pull_requests: PullRequestClient,
        clock: Clock,
        splitter: MarkdownSplitter | None = None,
        merger: IncrementalMerger | None = None,
        codec: SegmentCodec | None = None,
    ) -> None:
        self.config = config
        self.git = git
        self.fs = fs
        self.translate = translate
        self.pull_requests = pull_requests
        self.clock = clock
        self.splitter = splitter or MarkdownSplitter()
        self.merger = merger or IncrementalMerger(BlockAligner())
        self.codec = codec or SegmentCodec()
        self.file_filter = FileFilter(config.source)
        self.memory = TranslationMemory.load(
            fs,
            config.memory.file,
            min_length_ratio=config.memory.min_length_ratio,
            max_length_ratio=config.memory.max_length_ratio,
        ) if config.memory.enabled else TranslationMemory()
        self._branch: str | None = None
        self._pr_number: int | None = None

    def _remember(self, source: str, translation: str) -> None:
        """Кладёт пару в память, если она включена."""
        if self.config.memory.enabled:
            self.memory.add(source, translation)

    def _apply_memory(self, plan: FilePlan) -> None:
        """Закрывает памятью то, что можно, остальному подбирает похожие пары.

        Точное совпадение снимает блок с перевода совсем. Похожее не
        подставляется молча — оно уходит модели как задание на минимальную
        правку, чтобы вычитанный человеком текст не переписывался заново.
        """
        if not self.config.memory.enabled:
            return
        reused = repairs = 0
        for item in plan.items:
            if item.kind != "translate":
                continue
            exact = self.memory.lookup(item.text)
            if exact is not None:
                item.kind, item.text = "keep", exact
                reused += 1
                continue
            near = self.memory.nearest(item.text, self.config.memory.fuzzy_threshold)
            if near is not None:
                item.hint = near
                repairs += 1
        if reused or repairs:
            print(
                f"{plan.path}: из памяти взято {reused}, на починку помечено {repairs}."
            )

    def _ingest(self, base: Document, translated: Document) -> None:
        """Забирает в память пары из уже переведённого файла.

        Только уверенное выравнивание: у прозы сигнатура всегда ("para",),
        поэтому при низкой уверенности совпадения внутри длинного прогона
        абзацев позиционные, и один пропущенный абзац сдвинул бы все
        последующие — в память попали бы неверные пары.
        """
        if not self.config.memory.enabled or not translated.blocks:
            return
        alignment = self.merger.aligner.align(base.blocks, translated.blocks)
        if not alignment.confident:
            return
        for source_index, target_index in alignment.mapping.items():
            block = base.blocks[source_index]
            if block.is_code or target_index >= len(translated.blocks):
                continue
            self._remember(block.text, translated.blocks[target_index].text)

    def _state_paths(self, head: str, pending: dict[str, str]) -> list[str]:
        """Файлы состояния для коммита, включая память, если она включена."""
        paths = self._write_state(head, pending)
        if self.config.memory.enabled:
            self.memory.save(self.fs, self.config.memory.file)
            paths.append(self.config.memory.file)
        return paths

    def _read_pending(self) -> dict[str, str]:
        """Отложенные файлы вида «путь -> коммит, которому отвечает их перевод».

        Базу нужно хранить рядом с путём: общий маркер к моменту повтора уже
        уходит вперёд, и если диффить от него, изменения файла окажутся позади
        базы — переводить будет «нечего», а старый текст молча останется.
        Записи старого формата и записи с испорченной базой получают пустое
        значение: обработать их можно только подставив заведомо неверную базу,
        поэтому они остаются в очереди и выносятся человеку.
        """
        path = self.config.state.pending_file
        if not self.fs.exists(path):
            return {}
        entries: dict[str, str] = {}
        for line in self.fs.read(path).splitlines():
            line = line.strip()
            if not line:
                continue
            base, _, name = line.partition("\t")
            if name and _COMMIT_SHA_RE.match(base):
                entries[name] = base
            else:
                entries[name or line] = ""
        return entries

    def _write_state(self, head: str, pending: dict[str, str]) -> list[str]:
        self.fs.write(self.config.state.sync_file, head)
        lines = [f"{base}\t{name}" for name, base in sorted(pending.items())]
        self.fs.write(
            self.config.state.pending_file,
            "\n".join(lines) + ("\n" if lines else ""),
        )
        return [self.config.state.sync_file, self.config.state.pending_file]

    @staticmethod
    def _defer(
        pending: dict[str, str],
        plans: Sequence[FilePlan],
        bases: dict[str, str],
    ) -> None:
        """Откладывает файлы, запоминая базу, от которой их надо диффить дальше."""
        for plan in plans:
            pending.setdefault(plan.path, bases[plan.path])

    def _show(self, ref: str, path: str) -> str:
        try:
            return self.git.show(ref, path)
        except FileNotFoundError:
            return ""

    def _build_plan(self, path: str, base_ref: str) -> FilePlan | None:
        head_text = self._show(self.config.source.ref, path)
        if not head_text:
            return None
        base_doc = self.splitter.split(self._show(base_ref, path))
        head_doc = self.splitter.split(head_text)
        existing = self.fs.read(path) if self.fs.exists(path) else ""
        translated_doc = self.splitter.split(existing)
        # Вчитываем то, что уже переведено, до планирования: так правки человека
        # становятся эталоном и переиспользуются в других файлах.
        self._ingest(base_doc, translated_doc)
        return self.merger.plan(path, base_doc, head_doc, translated_doc)

    def _window(self, anchors: Iterable[int], size: int) -> set[int] | None:
        """Индексы, попадающие в окно вокруг переводимых кусков.

        None означает «брать документ целиком» (context_mode = "full").
        """
        if self.config.prompt.context_mode != "window":
            return None
        radius = self.config.prompt.window_blocks
        keep: set[int] = set()
        for anchor in anchors:
            low = max(0, anchor - radius)
            keep.update(range(low, min(size, anchor + radius + 1)))
        return keep

    def _build_prompt(self, plans: Sequence[FilePlan], ids: dict[int, PlanItem]) -> str:
        reverse = {id(item): segment_id for segment_id, item in ids.items()}
        sections: list[str] = []
        for plan in plans:
            blocks = plan.source_document.blocks
            translate_by_index = {item.source_index: item for item in plan.translatable}
            source_window = self._window(translate_by_index, len(blocks))

            marked: list[str] = []
            elided = False
            for index, block in enumerate(blocks):
                if source_window is not None and index not in source_window:
                    if not elided:
                        marked.append("[…]\n\n")
                        elided = True
                    continue
                elided = False
                item = translate_by_index.get(index)
                if item is not None:
                    marked.append(self.codec.wrap(reverse[id(item)], block.text))
                else:
                    marked.append(block.text)
                marked.append(block.sep or "\n\n")

            sections.append(
                f"### FILE: {plan.path}\n"
                f"--- SOURCE (translate only the ⟦S…⟧ segments) ---\n"
                f"{''.join(marked)}\n"
                f"--- EXISTING TRANSLATION (match its terminology and style) ---\n"
                f"{self._translation_context(plan)}\n"
            )
        glossary = "\n".join(
            f"{k} -> {v}" for k, v in (self.config.prompt.glossary or {}).items()
        )
        # Задания на починку: у сегмента есть почти такой же прежний оригинал и
        # его перевод. Просим внести минимальную правку, а не переводить заново,
        # чтобы вычитанный человеком текст уцелел.
        repairs = "\n".join(
            f"Segment {segment_id}:\n"
            f"PREVIOUS SOURCE:\n{item.hint.source}\n"
            f"PREVIOUS TRANSLATION:\n{item.hint.translation}\n"
            for segment_id, item in sorted(ids.items())
            if item.hint is not None
        )
        return self.config.prompt.template.format(
            language=self.config.target.language,
            glossary=glossary,
            source="\n".join(sections),
            repairs=repairs or "(none)",
            existing_translation="",
        )

    def _translation_context(self, plan: FilePlan) -> str:
        """Существующий перевод как эталон терминологии — целиком или окном."""
        anchors = [i for i, item in enumerate(plan.items) if item.kind == "translate"]
        window = self._window(anchors, len(plan.items))
        if window is None:
            return plan.existing_translation

        parts: list[str] = []
        elided = False
        for index, item in enumerate(plan.items):
            if index not in window or item.kind == "translate":
                if not elided:
                    parts.append("[…]\n\n")
                    elided = True
                continue
            elided = False
            parts.append(item.text + (item.sep or "\n\n"))
        return "".join(parts)

    def _ensure_branch(self) -> str:
        if self._branch is None:
            self._branch = f"{self.config.target.branch_prefix}{self.clock.stamp()}"
            self.git.create_branch(self._branch)
        return self._branch

    def _publish(self, paths: Sequence[str], message: str) -> None:
        branch = self._ensure_branch()
        self.git.commit(paths, message)
        self.git.push(branch)
        if self._pr_number is None:
            self._pr_number = self.pull_requests.create_draft(
                branch=branch,
                base=self.config.target.branch,
                title=self.config.target.pr_title,
                body="Перевод в процессе…",
                label=self.config.target.pr_label,
            )

    def run(self) -> RunResult:
        """Выполняет полный прогон и возвращает его итоги."""
        result = RunResult()
        state_path = self.config.state.sync_file
        head = self.git.rev_parse(self.config.source.ref)

        if not self.fs.exists(state_path):
            self.fs.write(state_path, head)
            print(f"Создан стартовый маркер {state_path} на коммите {head}.")
            return result

        base_ref = self.fs.read(state_path).strip()
        if not _COMMIT_SHA_RE.match(base_ref):
            raise ValueError(f"{state_path} содержит некорректный коммит: {base_ref!r}")

        pending = self._read_pending()
        changed = self.git.diff_name_status(
            base_ref, self.config.source.ref, self.config.source.include
        )

        # Каждый кандидат несёт свою базу: для свежих изменений это общий маркер,
        # а для отложенных — коммит, которому отвечает их текущий перевод.
        candidates: list[tuple[str, str]] = []
        for status, path in changed:
            if not self.file_filter.accepts(path):
                continue
            if status == "D":
                self.fs.remove(path)
                pending.pop(path, None)
                result.removed.append(path)
            else:
                # Файл мог одновременно измениться и ждать в очереди. Тогда
                # верна база из очереди: она старше маркера, и перевод отвечает
                # именно ей. Взяв маркер, мы пропустили бы всё, что накопилось
                # до него, — то есть ровно то, ради чего файл и был отложен.
                queued = pending.get(path)
                if queued is not None and not queued:
                    continue  # база не указана — разберём ниже, вместе с человеком
                candidates.append((path, queued or base_ref))
        fresh = {path for path, _ in candidates}
        for path in sorted(set(pending) - fresh):
            # Правила исключений действуют и на очередь: иначе однажды
            # застрявший файл переводился бы вопреки изменившемуся конфигу.
            if not self.file_filter.accepts(path):
                print(f"{path} исключён конфигом — убираю из очереди.")
                pending.pop(path, None)
                continue
            if not pending[path]:
                print(
                    f"У {path} в {self.config.state.pending_file} нет корректного "
                    "базового коммита — оставляю в очереди, впишите его вручную."
                )
                result.skipped.append(path)
                continue
            candidates.append((path, pending[path]))

        # Пока файл не обработан, он числится отложенным. Маркер синхронизации
        # уходит на head уже на первом коммите, поэтому оборванный прогон иначе
        # оставил бы состояние «всё готово» для файлов, до которых не дошёл, —
        # и их изменения пропали бы навсегда.
        for path, file_base in candidates:
            pending.setdefault(path, file_base)

        plans: list[FilePlan] = []
        plan_bases: dict[str, str] = {}
        for path, file_base in candidates:
            plan = self._build_plan(path, file_base)
            if plan is None:
                # Из очереди не убираем: маркер уйдёт вперёд, и файл больше
                # ничем не всплывёт. Перепроверка стоит ноль обращений к API —
                # выравнивание считается локально, — поэтому файл сам вернётся
                # в работу в тот прогон, когда человек его починит.
                print(f"Пропускаю {path}: не удалось надёжно сопоставить перевод.")
                result.skipped.append(path)
                continue
            self._apply_memory(plan)
            if not plan.translatable:
                # Изменились только удаления или блоки кода — переводить нечего.
                # Публикуем сразу: иначе правка осталась бы только в рабочем
                # дереве, а маркер ушёл бы вперёд — и файл никогда не вернулся
                # бы в обработку.
                self.fs.write(
                    path, plan.prefix + "".join(i.text + i.sep for i in plan.items)
                )
                result.translated.append(path)
                pending.pop(path, None)
                self._publish(
                    [path, *self._state_paths(head, pending)],
                    f"docs: обновление {path} без перевода",
                )
                continue
            plan_bases[path] = file_base
            plans.append(plan)

        by_path = {plan.path: plan for plan in plans}
        batches = BatchPlanner(self.config.prompt.max_request_chars).plan(
            [(plan.path, plan.cost()) for plan in plans]
        )

        review: list[ReviewComment] = []
        quota_spent = False
        for batch in batches:
            batch_plans = [by_path[path] for path, _ in batch]
            if quota_spent:
                self._defer(pending, batch_plans, plan_bases)
                continue

            ids: dict[int, PlanItem] = {}
            next_id = 1
            for plan in batch_plans:
                for item in plan.translatable:
                    ids[next_id] = item
                    next_id += 1

            request = TranslationRequest(
                prompt=self._build_prompt(batch_plans, ids),
                segments={sid: item.text for sid, item in ids.items()},
            )
            try:
                response = self.translate(request)
            except QuotaExhausted as exc:
                print(f"Квота исчерпана, остальное откладываю: {exc}")
                quota_spent = True
                result.quota_exhausted = True
                self._defer(pending, batch_plans, plan_bases)
                continue
            except ProviderError as exc:
                print(f"Ошибка провайдера, батч отложен: {exc}")
                self._defer(pending, batch_plans, plan_bases)
                continue

            received = self.codec.parse(response)
            for plan in batch_plans:
                wanted = [sid for sid, item in ids.items() if item in plan.translatable]
                if any(not received.get(sid, "").strip() for sid in wanted):
                    print(
                        f"Неполный ответ по {plan.path}: файл оставлен без изменений."
                    )
                    self._defer(pending, [plan], plan_bases)
                    continue
                rendered = plan.prefix
                for item in plan.items:
                    if item.kind == "keep":
                        rendered += item.text + item.sep
                    else:
                        segment_id = next(s for s in wanted if ids[s] is item)
                        # Номер строки, на которой начнётся перевод: по нему
                        # оригинал прицепится к диффу, чтобы ревьюер видел, с
                        # чего переводили, не уходя в апстрим.
                        review.append(
                            ReviewComment(
                                path=plan.path,
                                line=rendered.count("\n") + 1,
                                body=self.config.target.review_comment_template.format(
                                    source=item.text
                                ),
                            )
                        )
                        self._remember(item.text, received[segment_id])
                        rendered += received[segment_id] + item.sep
                self.fs.write(plan.path, rendered)
                result.translated.append(plan.path)
                pending.pop(plan.path, None)
                self._publish(
                    [plan.path, *self._state_paths(head, pending)],
                    f"docs: перевод {plan.path}",
                )

        result.pending = sorted(pending)
        state_paths = self._state_paths(head, pending)
        if self._branch is not None:
            self._publish(state_paths, "docs: обновление состояния перевода")
            if self._pr_number:
                self.pull_requests.update_body(
                    self._pr_number, self._render_body(result)
                )
                # Ревью — после последнего коммита: комментарии привязываются к
                # строкам конечной версии файлов.
                self.pull_requests.add_review(
                    self._pr_number,
                    self.git.rev_parse("HEAD"),
                    review[: self.config.target.max_review_comments],
                )
                self.pull_requests.mark_ready(self._pr_number)
        return result

    def _render_body(self, result: RunResult) -> str:
        def listing(paths: Sequence[str]) -> str:
            return "\n".join(f"- {p}" for p in paths) or "—"

        # Непереводимые файлы остаются в очереди, чтобы не потеряться, но в теле
        # PR им место только в разделе про ручную работу — иначе один и тот же
        # файл читался бы как «ждёт квоты» и «ждёт человека» одновременно.
        skipped = set(result.skipped)
        return self.config.target.pr_body_template.format(
            translated=listing(result.translated),
            skipped=listing(result.skipped),
            pending=listing([p for p in result.pending if p not in skipped]),
        )


class NoRemotePullRequests:
    """PullRequestClient, который ничего не создаёт, а только печатает."""

    def create_draft(
        self, branch: str, base: str, title: str, body: str, label: str
    ) -> int:
        """Сообщает, какой PR был бы открыт."""
        print(f"[--no-remote] открыл бы черновой PR из {branch} в {base}")
        return 0

    def mark_ready(self, number: int) -> None:
        """Сообщает о переводе PR в готовый."""
        print(f"[--no-remote] пометил бы PR #{number} готовым")

    def update_body(self, number: int, body: str) -> None:
        """Сообщает об обновлении описания."""
        print(f"[--no-remote] обновил бы описание PR #{number}")

    def add_review(
        self, number: int, commit_id: str, comments: Sequence[ReviewComment]
    ) -> None:
        """Сообщает, сколько оригиналов было бы приложено к диффу."""
        print(f"[--no-remote] приложил бы к PR #{number} оригиналов: {len(comments)}")
        for c in comments[:3]:
            print(f"   {c.path}:{c.line} — {c.body.splitlines()[-1][:70]!r}")


def print_request(request: TranslationRequest) -> str:
    """Печатает промпт вместо обращения к модели и возвращает пустой ответ."""
    print("=" * 70)
    print(
        f"ЗАПРОС: {len(request.segments)} сегментов, "
        f"{len(request.prompt)} символов промпта"
    )
    print(f"Номера сегментов: {sorted(request.segments)}")
    print("-" * 70)
    print(request.prompt)
    print("=" * 70)
    return ""


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа: читает конфиг, запускает пайплайн, печатает итоги."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="translate-config.toml",
        help="путь к TOML-конфигу",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="не обращаться к модели: печатать промпт вместо запроса. "
        "Ключи API при этом не нужны",
    )
    parser.add_argument(
        "--no-remote",
        action="store_true",
        help="ничего не отправлять наружу: без push и без операций с pull request. "
        "Локальные ветки и коммиты создаются",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="полностью инертный прогон: включает --no-llm и --no-remote и вдобавок "
        "не трогает ни файлы, ни локальный репозиторий",
    )
    args = parser.parse_args(argv)

    no_llm = args.no_llm or args.dry_run
    no_remote = args.no_remote or args.dry_run

    config = load_config(args.config)
    file_system: FileSystem = RealFileSystem()
    git: GitClient = SubprocessGit()

    if no_llm:
        print("Запросы к модели отключены: печатаю промпты.")
        translate: Callable[[TranslationRequest], str] = print_request
    else:
        translators = build_translators(config)
        translate = lambda request: chain_translate(translators, request)  # noqa: E731

    if no_remote:
        print("Отправка наружу отключена: без push и без pull request.")
        pull_requests: PullRequestClient = NoRemotePullRequests()
        git = NoPushGit(git)
    else:
        pull_requests = GhPullRequests()

    if args.dry_run:
        # Без этого ветка «переводить нечего» доходит до публикации и создаёт
        # настоящую ветку с коммитом прямо в рабочем каталоге.
        print("Инертный прогон: файлы и локальный репозиторий не меняются.")
        file_system = ReadOnlyFileSystem(file_system)
        git = ReadOnlyGit(SubprocessGit())
    print()

    pipeline = TranslationPipeline(
        config=config,
        git=git,
        fs=file_system,
        translate=translate,
        pull_requests=pull_requests,
        clock=SystemClock(),
    )
    result = pipeline.run()

    print(
        f"\nПереведено: {len(result.translated)}, удалено: {len(result.removed)}, "
        f"пропущено: {len(result.skipped)}, отложено: {len(result.pending)}"
    )
    for path in result.skipped:
        print(f"  нужен ручной перевод (не сопоставилось): {path}")
    for path in result.pending:
        print(f"  отложено до следующего прогона: {path}")
    if result.quota_exhausted:
        print(
            "Квота провайдера исчерпана — это штатный режим: очередь "
            "разгребётся, когда лимит обновится."
        )

    if no_llm:
        # Без обращений к модели переводить нечем, поэтому «отложено» здесь
        # ничего не означает и сигналом о проблеме быть не может.
        return 0
    return 1 if result.needs_attention else 0


if __name__ == "__main__":
    sys.exit(main())
