"""Tests for translator.py.

Deliberately small: only behaviour that would actually hurt if it broke.
Hand-written fakes instead of unittest.mock — fakes survive refactoring of
call signatures, mock assertions do not.
"""

import contextlib
import io
import subprocess
import unittest
from types import SimpleNamespace

from mdtranslate import translator
from mdtranslate.translator import (
    BatchPlanner,
    BlockAligner,
    Config,
    MarkdownSplitter,
    NoPushGit,
    NoRemotePullRequests,
    MemoryConfig,
    ProviderConfig,
    PromptConfig,
    QuotaExhausted,
    RealFileSystem,
    ReadOnlyFileSystem,
    ReadOnlyGit,
    RunResult,
    SegmentCodec,
    SubprocessGit,
    SourceConfig,
    StateConfig,
    TargetConfig,
    TranslationPipeline,
    TranslationMemory,
    TranslationRequest,
    chain_translate,
    print_request,
)

# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeGit:
    """Serves file contents per (ref, path) and records write operations."""

    def __init__(self, head, trees, changes):
        self._head = head
        self.trees = trees  # {ref: {path: text}}
        self._changes = changes  # [(status, path)]
        self.commits = []
        self.pushed = []
        self.branch = None

    def rev_parse(self, ref):
        return self._head

    def show(self, ref, path):
        try:
            return self.trees[ref][path]
        except KeyError:
            raise FileNotFoundError(f"{ref}:{path}")

    def diff_name_status(self, base, head, patterns):
        return list(self._changes)

    def create_branch(self, name):
        self.branch = name

    def commit(self, paths, message):
        self.commits.append((tuple(paths), message))

    def push(self, branch):
        self.pushed.append(branch)


class FakeFS:
    def __init__(self, files=None):
        self.files = dict(files or {})

    def read(self, path):
        return self.files[path]

    def write(self, path, text):
        self.files[path] = text

    def exists(self, path):
        return path in self.files

    def remove(self, path):
        self.files.pop(path, None)


class FakeProvider:
    """Echoes each requested segment back as RU(<source>)."""

    def __init__(self, codec, fail_with=None, drop_segments=()):
        self.codec = codec
        self.calls = 0
        self.translated_sources = []
        self.prompts = []
        self._fail_with = fail_with
        self._drop = set(drop_segments)

    def translate(self, request):
        self.calls += 1
        self.prompts.append(request.prompt)
        if self._fail_with is not None:
            raise self._fail_with
        parts = []
        for seg_id, source in request.segments.items():
            if seg_id in self._drop:
                continue
            self.translated_sources.append(source)
            parts.append(self.codec.wrap(seg_id, f"RU({source})"))
        return "\n".join(parts)


class FakeSubprocess:
    """Stands in for the subprocess module so that git argv can be inspected."""

    DEVNULL = subprocess.DEVNULL
    CalledProcessError = subprocess.CalledProcessError

    def __init__(self, staged_returncode=1):
        self.argv = []
        self._staged = staged_returncode

    def check_output(self, args, **kwargs):
        self.argv.append(list(args))
        return ""

    def run(self, args, **kwargs):
        self.argv.append(list(args))
        return SimpleNamespace(returncode=self._staged)


class FakePRClient:
    def __init__(self):
        self.created = None
        self.ready = False
        self.body = None
        self.review = []

    def create_draft(self, branch, base, title, body, label):
        self.created = branch
        self.body = body
        return 42

    def mark_ready(self, number):
        self.ready = True

    def update_body(self, number, body):
        self.body = body

    def add_review(self, number, commit_id, comments):
        self.review = list(comments)


class FakeClock:
    def stamp(self):
        return "20260101-000000"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def make_config(**overrides):
    cfg = Config(
        source=SourceConfig(
            ref="origin/main",
            include=["*.md"],
            exclude_paths=[".github"],
            exclude_files=["readme.md"],
        ),
        target=TargetConfig(
            branch="ru",
            language="Russian",
            branch_prefix="translate/sync-",
            pr_label="translate",
            pr_title="auto",
            pr_body_template="{translated}|{skipped}|{pending}",
        ),
        state=StateConfig(
            sync_file=".github/sync.txt", pending_file=".github/pending.txt"
        ),
        prompt=PromptConfig(
            template="{language}\n{glossary}\n{source}\nREPAIRS:\n{repairs}",
            glossary={},
            context_mode="full",
            window_blocks=20,
            max_request_chars=100_000,
        ),
        providers=[ProviderConfig(name="gemini", model="m", api_key_env="K")],
    )
    return cfg if not overrides else cfg.replace(**overrides)


def build_pipeline(base_tree, head_tree, working, changes, provider=None, config=None):
    """Wires a pipeline over fakes. base_tree is the source at last sync."""
    cfg = config or make_config()
    codec = SegmentCodec()
    git = FakeGit(
        head="bbbbbbb2222222",
        trees={"origin/main": head_tree, "aaaaaaa1111111": base_tree},
        changes=changes,
    )
    fs = FakeFS({**working, ".github/sync.txt": "aaaaaaa1111111"})
    prov = provider if provider is not None else FakeProvider(codec)
    pipeline = TranslationPipeline(
        config=cfg,
        git=git,
        fs=fs,
        translate=lambda req: prov.translate(req),
        pull_requests=FakePRClient(),
        clock=FakeClock(),
    )
    return pipeline, git, fs, prov


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


class TestMarkdownSplitter(unittest.TestCase):
    def test_split_join_round_trip_is_byte_exact(self):
        splitter = MarkdownSplitter()
        samples = [
            "# Title\n\nSome text.\n\n```go\nfunc main() {}\n\nstill code\n```\n\nEnd.\n",
            "no trailing newline",
            "a\n\n\n\nb\n",
            "",
            "\n\nleading blanks\n",
        ]
        for text in samples:
            with self.subTest(text=text[:20]):
                doc = splitter.split(text)
                self.assertEqual(doc.render(), text)

    def test_fenced_code_stays_one_block(self):
        doc = MarkdownSplitter().split("intro\n\n```go\na\n\nb\n```\n\nouttro\n")
        self.assertEqual(len(doc.blocks), 3)
        self.assertIn("a\n\nb", doc.blocks[1].text)


class TestIncrementalMerge(unittest.TestCase):
    def test_manual_edit_in_unchanged_block_survives(self):
        base = "A para.\n\nB para.\n\nC para.\n"
        head = "A para.\n\nB para CHANGED.\n\nC para.\n"
        ru = "Ручной перевод A.\n\nПеревод B.\n\nПеревод C.\n"

        pipeline, _, fs, prov = build_pipeline(
            base_tree={"ch.md": base},
            head_tree={"ch.md": head},
            working={"ch.md": ru},
            changes=[("M", "ch.md")],
        )
        result = pipeline.run()

        self.assertEqual(result.translated, ["ch.md"])
        out = fs.files["ch.md"]
        self.assertIn("Ручной перевод A.", out)
        self.assertIn("Перевод C.", out)
        self.assertIn("RU(B para CHANGED.)", out)
        self.assertNotIn("Перевод B.", out)

    def test_review_attaches_the_source_to_the_translated_line(self):
        """К каждому переведённому блоку прикладывается его оригинал.

        Иначе в диффе виден только результат, и оценить качество перевода
        нельзя — сравнивать не с чем.
        """
        pipeline, _, fs, _ = build_pipeline(
            base_tree={"ch.md": "A para.\n\nB para.\n"},
            head_tree={"ch.md": "A para.\n\nB para.\n\nBrand new para.\n"},
            working={"ch.md": "Перевод A.\n\nПеревод B.\n"},
            changes=[("M", "ch.md")],
        )
        pipeline.run()

        review = pipeline.pull_requests.review
        self.assertEqual(len(review), 1, "один новый блок — один комментарий")
        comment = review[0]
        self.assertEqual(comment.path, "ch.md")
        self.assertIn("Brand new para.", comment.body, "в комментарии оригинал")

        # Комментарий должен указывать на строку, где лежит перевод этого блока.
        line = fs.files["ch.md"].split("\n")[comment.line - 1]
        self.assertIn("RU(Brand new para.)", line)

    def test_only_changed_blocks_are_sent_for_translation(self):
        base = "A.\n\nB.\n\nC.\n"
        head = "A.\n\nB2.\n\nC.\n"
        pipeline, _, _, prov = build_pipeline(
            base_tree={"ch.md": base},
            head_tree={"ch.md": head},
            working={"ch.md": "ra.\n\nrb.\n\nrc.\n"},
            changes=[("M", "ch.md")],
        )
        pipeline.run()
        self.assertEqual(prov.translated_sources, ["B2."])

    def test_block_deleted_upstream_disappears_from_translation(self):
        pipeline, git, fs, prov = build_pipeline(
            base_tree={"ch.md": "A.\n\nB.\n\nC.\n"},
            head_tree={"ch.md": "A.\n\nC.\n"},
            working={"ch.md": "ra.\n\nrb.\n\nrc.\n"},
            changes=[("M", "ch.md")],
        )
        pipeline.run()
        out = fs.files["ch.md"]
        self.assertNotIn("rb.", out)
        self.assertIn("ra.", out)
        self.assertIn("rc.", out)
        self.assertEqual(prov.calls, 0, "nothing new to translate")

        committed = {path for paths, _ in git.commits for path in paths}
        self.assertIn(
            "ch.md", committed, "a file with nothing to translate must still be committed"
        )

    def test_unalignable_file_stays_queued_until_a_human_repairs_it(self):
        """Файл не выбывает из очереди и сам возвращается в работу после починки.

        Маркер синхронизации уходит вперёд, поэтому выброшенный из очереди файл
        больше ничем не всплыл бы. Перепроверка бесплатна: выравнивание считается
        локально, без обращений к API.
        """
        base = "one.\n\ntwo.\n\nthree.\n\nfour.\n\nfive.\n"
        head = "one X.\n\ntwo.\n\nthree.\n\nfour.\n\nfive.\n"
        broken = "only.\n\ntwo.\n"  # структура разошлась — сопоставить нельзя

        pipeline, _, fs, prov = build_pipeline(
            base_tree={"ch.md": base},
            head_tree={"ch.md": head},
            working={"ch.md": broken},
            changes=[("M", "ch.md")],
        )
        result = pipeline.run()

        self.assertEqual(result.skipped, ["ch.md"])
        self.assertEqual(prov.calls, 0, "перепроверка не должна стоить запросов")
        self.assertIn("ch.md", fs.files[".github/pending.txt"], "остался в очереди")

        # Человек починил структуру — следующий прогон подхватывает файл сам.
        repaired = "r-one.\n\nr-two.\n\nr-three.\n\nr-four.\n\nr-five.\n"
        pipeline2, _, fs2, prov2 = build_pipeline(
            base_tree={"ch.md": base},
            head_tree={"ch.md": head},
            working={"ch.md": repaired},
            changes=[],  # свежих изменений нет, файл приходит только из очереди
        )
        fs2.files[".github/pending.txt"] = "aaaaaaa1111111\tch.md\n"
        result2 = pipeline2.run()

        self.assertEqual(result2.translated, ["ch.md"])
        self.assertIn("RU(one X.)", fs2.files["ch.md"])
        self.assertIn("r-five.", fs2.files["ch.md"], "ручной перевод сохранён")

    def test_unalignable_file_is_skipped_and_left_untouched(self):
        original_ru = "only.\n\ntwo.\n"
        pipeline, _, fs, _ = build_pipeline(
            base_tree={"ch.md": "one.\n\ntwo.\n\nthree.\n\nfour.\n\nfive.\n"},
            head_tree={"ch.md": "one X.\n\ntwo.\n\nthree.\n\nfour.\n\nfive.\n"},
            working={"ch.md": original_ru},
            changes=[("M", "ch.md")],
        )
        result = pipeline.run()
        self.assertIn("ch.md", result.skipped)
        self.assertEqual(fs.files["ch.md"], original_ru)


class TestResilience(unittest.TestCase):
    def test_malformed_response_only_fails_its_own_file(self):
        codec = SegmentCodec()
        # segment 1 belongs to a.md, segment 2 to b.md; drop the second one
        provider = FakeProvider(codec, drop_segments=(2,))
        pipeline, _, fs, _ = build_pipeline(
            base_tree={"a.md": "A.\n", "b.md": "B.\n"},
            head_tree={"a.md": "A2.\n", "b.md": "B2.\n"},
            working={"a.md": "ra.\n", "b.md": "rb.\n"},
            changes=[("M", "a.md"), ("M", "b.md")],
            provider=provider,
        )
        result = pipeline.run()
        self.assertEqual(result.translated, ["a.md"])
        self.assertEqual(result.pending, ["b.md"])
        self.assertEqual(fs.files["b.md"], "rb.\n", "untouched on bad response")

    def test_deferred_file_is_translated_after_the_marker_moved_on(self):
        """Отложенный файл диффится от своей базы, а не от общего маркера.

        Иначе после продвижения маркера его изменения оказываются позади базы,
        переводить «нечего», и устаревший текст молча остаётся навсегда.
        """
        pipeline, _, fs, prov = build_pipeline(
            base_tree={"ch.md": "A.\n\nB OLD.\n\nC.\n"},
            head_tree={"ch.md": "A.\n\nB NEW.\n\nC.\n"},
            working={"ch.md": "ra.\n\nrb.\n\nrc.\n"},
            changes=[],  # общий маркер уже догнал голову
        )
        fs.files[".github/pending.txt"] = "aaaaaaa1111111\tch.md\n"

        result = pipeline.run()

        self.assertEqual(result.translated, ["ch.md"])
        self.assertEqual(prov.calls, 1)
        self.assertIn("RU(B NEW.)", fs.files["ch.md"])
        self.assertIn("ra.", fs.files["ch.md"], "ручной перевод сохранён")

    def test_queued_file_keeps_its_own_base_when_it_also_changes_upstream(self):
        """У файла из очереди база старше маркера — она и должна победить.

        Иначе изменения, накопившиеся до маркера (ради которых файл и отложили),
        оказываются позади базы и теряются молча.
        """
        pipeline, git, fs, prov = build_pipeline(
            base_tree={"ch.md": "A.\n\nB OLD.\n"},  # чему отвечает перевод
            head_tree={"ch.md": "A.\n\nB NEW.\n\nC ДОБАВЛЕН ПОЗЖЕ.\n"},
            working={"ch.md": "ra.\n\nrb.\n"},
            changes=[("M", "ch.md")],  # файл ещё и изменился свежим диффом
        )
        # Маркер стоит МЕЖДУ базой очереди и головой: B успел измениться до него.
        git.trees["ccccccc3333333"] = {"ch.md": "A.\n\nB NEW.\n"}
        fs.files[".github/sync.txt"] = "ccccccc3333333"
        fs.files[".github/pending.txt"] = "aaaaaaa1111111\tch.md\n"

        pipeline.run()

        # Оба изменения должны попасть в перевод, а не только то, что позже маркера.
        self.assertIn("C ДОБАВЛЕН ПОЗЖЕ.", prov.translated_sources)
        self.assertIn(
            "B NEW.",
            prov.translated_sources,
            "изменение до маркера потеряно — взята не та база",
        )

    def test_quota_exhausted_falls_over_to_next_provider(self):
        codec = SegmentCodec()
        dead = FakeProvider(codec, fail_with=QuotaExhausted("daily limit"))
        alive = FakeProvider(codec)
        request = TranslationRequest(prompt="p", segments={1: "hello"})

        out = chain_translate([dead.translate, alive.translate], request)

        self.assertEqual(alive.calls, 1)
        self.assertIn("RU(hello)", out)

    def test_chain_reraises_when_every_provider_is_exhausted(self):
        codec = SegmentCodec()
        a = FakeProvider(codec, fail_with=QuotaExhausted("x"))
        b = FakeProvider(codec, fail_with=QuotaExhausted("y"))
        with self.assertRaises(QuotaExhausted):
            chain_translate(
                [a.translate, b.translate],
                TranslationRequest(prompt="p", segments={1: "s"}),
            )


class TestTranslationMemory(unittest.TestCase):
    def test_pair_with_wild_length_ratio_is_rejected(self):
        """Неверная пара хуже отсутствующей: она подставится молча и навсегда."""
        memory = TranslationMemory()
        self.assertTrue(memory.add("A normal sentence.", "Обычное предложение."))
        self.assertFalse(memory.add("short", "x" * 500), "перевод неправдоподобно длинный")
        self.assertFalse(memory.add("x" * 500, "short"), "перевод неправдоподобно короткий")
        self.assertFalse(memory.add("  ", "непустой"), "пустой оригинал")
        self.assertEqual(len(memory), 1)

    def test_round_trip_through_jsonl(self):
        memory = TranslationMemory()
        memory.add("Hello there.", "Здравствуйте.")
        fs = FakeFS()
        memory.save(fs, "tm.jsonl")
        restored = TranslationMemory.load(fs, "tm.jsonl")
        self.assertEqual(restored.lookup("Hello there."), "Здравствуйте.")
        self.assertEqual(len(restored), 1)

    def test_nearest_respects_the_threshold(self):
        memory = TranslationMemory()
        memory.add("The quick brown fox jumps over the dog.", "Быстрая лиса прыгает.")
        near = memory.nearest("The quick brown fox jumps over the cat.", 0.8)
        self.assertIsNotNone(near)
        self.assertEqual(near.translation, "Быстрая лиса прыгает.")
        self.assertIsNone(memory.nearest("Completely unrelated wording here.", 0.8))


class TestMemoryInPipeline(unittest.TestCase):
    def _config(self):
        return make_config().replace(memory=MemoryConfig(file=".github/tm.jsonl"))

    def test_exact_hit_costs_no_request(self):
        pipeline, _, fs, prov = build_pipeline(
            base_tree={"ch.md": "A.\n"},
            head_tree={"ch.md": "A.\n\nKnown sentence.\n"},
            working={"ch.md": "ra.\n"},
            changes=[("M", "ch.md")],
            config=self._config(),
        )
        pipeline.memory.add("Known sentence.", "Известное предложение.")

        result = pipeline.run()

        self.assertEqual(result.translated, ["ch.md"])
        self.assertEqual(prov.calls, 0, "перевод должен прийти из памяти")
        self.assertIn("Известное предложение.", fs.files["ch.md"])

    def test_similar_block_goes_to_the_model_with_its_previous_translation(self):
        """Похожее не подставляется молча — модель чинит старый перевод."""
        old = "The quick brown fox jumps over the lazy dog."
        new = "The quick brown fox jumps over the lazy cat."
        pipeline, _, _, prov = build_pipeline(
            base_tree={"ch.md": "A.\n"},
            head_tree={"ch.md": f"A.\n\n{new}\n"},
            working={"ch.md": "ra.\n"},
            changes=[("M", "ch.md")],
            config=self._config(),
        )
        pipeline.memory.add(old, "Быстрая лиса прыгает через ленивого пса.")

        pipeline.run()

        self.assertEqual(prov.calls, 1, "блок всё же переводится")
        prompt = prov.prompts[0]
        self.assertIn("Быстрая лиса прыгает через ленивого пса.", prompt)
        self.assertIn(old, prompt, "в промпте есть прежний оригинал")

    def test_translations_are_remembered_and_committed(self):
        pipeline, git, fs, _ = build_pipeline(
            base_tree={"ch.md": "A.\n"},
            head_tree={"ch.md": "A.\n\nFresh text.\n"},
            working={"ch.md": "ra.\n"},
            changes=[("M", "ch.md")],
            config=self._config(),
        )
        pipeline.run()

        self.assertEqual(pipeline.memory.lookup("Fresh text."), "RU(Fresh text.)")
        self.assertIn(".github/tm.jsonl", fs.files, "база записана на диск")
        committed = {p for paths, _ in git.commits for p in paths}
        self.assertIn(".github/tm.jsonl", committed, "база уходит в коммит")

    def test_nothing_is_ingested_from_a_poorly_aligned_file(self):
        """При низкой уверенности совпадения прозы позиционные — такие пары
        отравили бы базу навсегда."""
        pipeline, _, _, _ = build_pipeline(
            base_tree={"ch.md": "one.\n\ntwo.\n\nthree.\n\nfour.\n\nfive.\n"},
            head_tree={"ch.md": "one X.\n\ntwo.\n\nthree.\n\nfour.\n\nfive.\n"},
            working={"ch.md": "only.\n\ntwo.\n"},
            changes=[("M", "ch.md")],
            config=self._config(),
        )
        pipeline.run()
        self.assertEqual(len(pipeline.memory), 0)


class TestExitSignal(unittest.TestCase):
    """Исчерпанная квота — штатный режим, а не авария.

    Если красить такой прогон в красное, настоящая поломка перестаёт
    отличаться от обычного дня, когда лимит просто кончился.
    """

    def test_quota_deferral_alone_is_not_a_failure(self):
        result = RunResult(translated=["a.md"], pending=["b.md"], quota_exhausted=True)
        self.assertFalse(result.needs_attention)

    def test_deferral_without_quota_is_a_failure(self):
        self.assertTrue(RunResult(pending=["b.md"]).needs_attention)

    def test_file_needing_a_human_is_always_a_failure(self):
        result = RunResult(skipped=["c.md"], quota_exhausted=True)
        self.assertTrue(result.needs_attention)


class TestIsolationFlags(unittest.TestCase):
    """Разделённые режимы изоляции.

    Раньше единственный --dry-run не защищал git: ветка «переводить нечего»
    доходила до публикации и создавала настоящую ветку с пушем.
    """

    def _wired(self, git):
        return TranslationPipeline(
            config=make_config(),
            git=git,
            fs=ReadOnlyFileSystem(
                FakeFS(
                    {
                        "ch.md": "ra.\n\nrb.\n\nrc.\n",
                        ".github/sync.txt": "aaaaaaa1111111",
                    }
                )
            ),
            translate=print_request,
            pull_requests=NoRemotePullRequests(),
            clock=FakeClock(),
        )

    def _fake_git(self):
        return FakeGit(
            head="bbbbbbb2222222",
            # блок удалён в голове -> «переводить нечего» -> путь до публикации
            trees={
                "origin/main": {"ch.md": "A.\n\nB.\n"},
                "aaaaaaa1111111": {"ch.md": "A.\n\nB.\n\nC.\n"},
            },
            changes=[("M", "ch.md")],
        )

    def test_read_only_git_touches_nothing(self):
        git = self._fake_git()
        with contextlib.redirect_stdout(io.StringIO()):
            self._wired(ReadOnlyGit(git)).run()
        self.assertIsNone(git.branch, "ветка не создаётся")
        self.assertEqual(git.commits, [], "коммитов нет")
        self.assertEqual(git.pushed, [], "пушей нет")

    def test_no_push_git_keeps_work_local(self):
        git = self._fake_git()
        with contextlib.redirect_stdout(io.StringIO()):
            self._wired(NoPushGit(git)).run()
        self.assertIsNotNone(git.branch, "локальная ветка создаётся")
        self.assertTrue(git.commits, "локальные коммиты создаются")
        self.assertEqual(git.pushed, [], "но наружу ничего не уходит")


class TestRepoRoot(unittest.TestCase):
    """Пути из конфига относительны корню репозитория, а не текущему каталогу.

    Иначе инструмент можно было бы запускать только изнутри целевого
    репозитория, и локальная проверка на боевых данных была бы невозможна.
    """

    def test_relative_paths_are_resolved_against_the_root(self):
        fs = RealFileSystem("/tmp/some/repo")
        self.assertEqual(
            fs.resolve(".github/state.txt"), "/tmp/some/repo/.github/state.txt"
        )

    def test_absolute_paths_are_left_alone(self):
        fs = RealFileSystem("/tmp/some/repo")
        self.assertEqual(fs.resolve("/etc/hosts"), "/etc/hosts")

    def test_default_root_keeps_paths_relative_to_cwd(self):
        self.assertEqual(RealFileSystem().resolve("a/b.md"), "a/b.md")

    def test_git_commands_carry_the_root(self):
        self.assertEqual(SubprocessGit("/tmp/repo").root, "/tmp/repo")

    def _patched_git(self, staged_returncode=1):
        """SubprocessGit поверх фейкового subprocess: видно каждый argv."""
        fake = FakeSubprocess(staged_returncode)
        original = translator.subprocess
        translator.subprocess = fake
        self.addCleanup(setattr, translator, "subprocess", original)
        return SubprocessGit("/tmp/repo"), fake

    def test_every_git_command_carries_the_root(self):
        """Проверка индекса тоже. Без -C она смотрела в репозиторий текущего
        каталога, и с --repo каждый коммит молча превращался в «нечего
        коммитить»: работа оставалась незакоммиченной, а прогон рапортовал
        об успехе."""
        git, fake = self._patched_git(staged_returncode=1)  # в индексе есть что

        git.commit(["ch.md"], "docs: перевод ch.md")

        self.assertTrue(fake.argv)
        for argv in fake.argv:
            self.assertEqual(argv[:3], ["git", "-C", "/tmp/repo"], argv)
        self.assertIn(
            ["git", "-C", "/tmp/repo", "commit", "-m", "docs: перевод ch.md"],
            fake.argv,
            "коммит должен состояться",
        )

    def test_clean_index_is_not_committed(self):
        """Пустой коммит git отвергает кодом 1 — это уронило бы весь прогон."""
        git, fake = self._patched_git(staged_returncode=0)  # различий нет

        with contextlib.redirect_stdout(io.StringIO()):
            git.commit(["ch.md"], "docs: перевод ch.md")

        self.assertNotIn("commit", [argv[3] for argv in fake.argv])


class TestBatchPlanner(unittest.TestCase):
    def test_small_files_share_a_batch_and_large_one_travels_alone(self):
        planner = BatchPlanner(max_chars=100)
        items = [("a.md", 10), ("b.md", 10), ("big.md", 500), ("c.md", 10)]
        batches = planner.plan(items)
        grouped = [[path for path, _ in batch] for batch in batches]
        self.assertIn(["a.md", "b.md", "c.md"], grouped)
        self.assertIn(["big.md"], grouped)


if __name__ == "__main__":
    unittest.main()
