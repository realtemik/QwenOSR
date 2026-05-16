import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from faster_whisper import WhisperModel
from PySide6.QtCore import QEvent, QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QAction, QFont, QIcon, QKeySequence, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.json"
ICON_PATH = ROOT_DIR / "assets" / "app.ico"


def init_windows_ole_for_clipboard() -> None:
    """
    На Windows буфер обмена Qt работает через OLE/COM.
    Важно инициализировать OLE в главном GUI-потоке до импортов/вызовов
    библиотек, которые могут выставить другой COM apartment mode.
    """
    if os.name != "nt":
        return

    try:
        import ctypes

        # S_OK = 0, S_FALSE = 1. Оба варианта означают, что OLE доступен
        # для текущего потока. RPC_E_CHANGED_MODE здесь уже будет поздно лечить,
        # поэтому soundcard импортируется позже, не в главном потоке.
        ctypes.windll.ole32.OleInitialize(None)
    except Exception:
        # Приложение не должно падать из-за недоступной инициализации OLE.
        pass


@dataclass
class AppConfig:
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    whisper_model: str = "small"
    language: str = "ru"
    # В ручном режиме это не момент отправки, а размер маленького блока записи.
    # Отправка происходит только по кнопке "Отправить запись".
    chunk_seconds: int = 1
    sample_rate: int = 16000
    min_text_length: int = 8
    min_record_seconds: float = 1.0
    system_prompt: str = (
        "Ты локальный ассистент. Анализируй расшифровку системного звука: "
        "кратко объясняй суть, выделяй важные ошибки, команды, решения и следующие действия. "
        "Отвечай по-русски. Если текста мало или он бессмысленный, напиши: 'Недостаточно полезного текста'."
    )


def load_config() -> AppConfig:
    if not CONFIG_PATH.exists():
        example = ROOT_DIR / "config.example.json"
        if example.exists():
            CONFIG_PATH.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return AppConfig(**{**AppConfig().__dict__, **data})
    return AppConfig()


class AudioAnalyzerWorker(QObject):
    log = Signal(str)
    status = Signal(str)
    error = Signal(str)
    buffer_seconds = Signal(float)
    qwen_start = Signal()
    qwen_delta = Signal(str)
    qwen_end = Signal()
    finished = Signal()

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self._stop_event = threading.Event()
        self._send_event = threading.Event()
        self._clear_event = threading.Event()
        self._whisper = None
        self._buffer = []
        self._buffer_frames = 0

    @Slot()
    def run(self):
        try:
            self._run_loop()
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    def stop(self):
        self._stop_event.set()

    def request_send(self):
        self._send_event.set()

    def request_clear_recording(self):
        self._clear_event.set()

    def _load_whisper(self):
        if self._whisper is None:
            self.status.emit(f"Загрузка Whisper: {self.config.whisper_model}...")
            self._whisper = WhisperModel(
                self.config.whisper_model,
                device="cpu",
                compute_type="int8",
            )
            self.status.emit("Whisper загружен")
        return self._whisper

    def _run_loop(self):
        import soundcard as sc

        self._check_ollama()
        whisper = self._load_whisper()

        speaker = sc.default_speaker()
        if speaker is None:
            raise RuntimeError("Не найдено устройство вывода звука")

        self.status.emit(f"Идёт ручная запись системного звука: {speaker.name}")

        loopback_mic = sc.get_microphone(speaker.name, include_loopback=True)
        record_chunk_seconds = max(0.25, float(self.config.chunk_seconds))
        frames_per_chunk = int(self.config.sample_rate * record_chunk_seconds)

        with loopback_mic.recorder(
            samplerate=self.config.sample_rate,
            channels=1,
            blocksize=1024,
        ) as recorder:
            while not self._stop_event.is_set():
                audio = recorder.record(numframes=frames_per_chunk)
                audio = self._to_mono_float32(audio)

                if self._clear_event.is_set():
                    self._clear_event.clear()
                    self._clear_buffer()
                    self.status.emit("Накопленная запись очищена. Запись продолжается...")
                    continue

                if audio.size:
                    self._buffer.append(audio)
                    self._buffer_frames += int(audio.shape[0])
                    seconds = self._buffer_frames / float(self.config.sample_rate)
                    self.buffer_seconds.emit(seconds)
                    self.status.emit("Запись идёт.")

                if self._send_event.is_set():
                    self._send_event.clear()
                    self._process_current_buffer(whisper)
                    if not self._stop_event.is_set():
                        self.status.emit("Запись продолжается. Нажмите 'Отправить запись' для следующего анализа.")

    def _process_current_buffer(self, whisper: WhisperModel):
        audio = self._pop_buffer_audio()
        seconds = audio.shape[0] / float(self.config.sample_rate) if audio.size else 0.0
        self.buffer_seconds.emit(0.0)

        if seconds < self.config.min_record_seconds:
            self.status.emit("Слишком короткая запись для отправки")
            self.log.emit("<div style='background:#fff7ed;border:1px solid #fed7aa;border-radius:10px;padding:10px 12px;margin:6px 0;'><b>Запись не отправлена:</b> накоплено слишком мало аудио.</div>")
            return

        if self._is_silence(audio):
            self.status.emit("В записи тишина / нет полезного звука")
            self.log.emit(
                f"<div style='background:#fff7ed;border:1px solid #fed7aa;border-radius:10px;padding:10px 12px;margin:6px 0;'><b>Запись не отправлена:</b> в {seconds:.1f} сек. аудио почти тишина.</div>"
            )
            return

        self.log.emit(f"<div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:10px 12px;margin:6px 0;'><b>Отправляю запись:</b> {seconds:.1f} сек.</div>")
        text = self._transcribe_audio(whisper, audio)
        if len(text.strip()) < self.config.min_text_length:
            self.status.emit("Слишком короткий распознанный текст")
            self.log.emit("<div style='background:#fff7ed;border:1px solid #fed7aa;border-radius:10px;padding:10px 12px;margin:6px 0;'><b>Распознано слишком мало текста:</b> запись не отправлена в Qwen.</div>")
            return

        self.log.emit(
            f"<div style='background:#f8fafc;border:1px solid #dbe4f0;border-radius:10px;padding:10px 12px;margin:6px 0;'><b>Распознано:</b><br>{self._html_escape(text)}</div>"
        )
        self._ask_ollama_stream(text)

    def _pop_buffer_audio(self) -> np.ndarray:
        if not self._buffer:
            return np.array([], dtype=np.float32)
        audio = np.concatenate(self._buffer).astype(np.float32)
        self._clear_buffer()
        return audio

    def _clear_buffer(self):
        self._buffer = []
        self._buffer_frames = 0
        self.buffer_seconds.emit(0.0)

    def _check_ollama(self):
        url = f"{self.config.ollama_base_url.rstrip('/')}/api/tags"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                "Ollama не отвечает. Проверьте, что запущен `ollama serve` "
                f"и доступен {url}. Ошибка: {exc}"
            ) from exc

    @staticmethod
    def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32)

    @staticmethod
    def _is_silence(audio: np.ndarray, threshold: float = 0.004) -> bool:
        if audio.size == 0:
            return True
        rms = float(np.sqrt(np.mean(np.square(audio))))
        return rms < threshold

    def _transcribe_audio(self, whisper: WhisperModel, audio: np.ndarray) -> str:
        self.status.emit("Распознаю речь...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, audio, self.config.sample_rate)
            segments, _info = whisper.transcribe(
                tmp_path,
                language=self.config.language,
                vad_filter=True,
                beam_size=1,
            )
            return " ".join(seg.text.strip() for seg in segments).strip()
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _ask_ollama_stream(self, text: str) -> str:
        self.status.emit(f"Qwen отвечает потоком: {self.config.ollama_model}...")
        url = f"{self.config.ollama_base_url.rstrip('/')}/api/generate"
        prompt = (
            f"{self.config.system_prompt}\n\n"
            "Расшифровка системного звука:\n"
            f"{text}\n\n"
            "Ответ:"
        )
        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.2,
                "num_predict": 300,
            },
        }

        full_answer = []
        self.qwen_start.emit()

        try:
            with requests.post(url, json=payload, stream=True, timeout=(10, 180)) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines(decode_unicode=True):
                    if self._stop_event.is_set():
                        break
                    if not raw_line:
                        continue
                    try:
                        data = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    piece = data.get("response", "")
                    if piece:
                        full_answer.append(piece)
                        self.qwen_delta.emit(piece)

                    if data.get("done", False):
                        break
        finally:
            if not full_answer:
                self.qwen_delta.emit("Пустой ответ от модели")
            self.qwen_end.emit()

        self.status.emit("Ответ Qwen получен")
        return "".join(full_answer).strip()

    @staticmethod
    def _html_escape(value: str) -> str:
        return (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("\n", "<br>")
        )


class TextQwenWorker(QObject):
    status = Signal(str)
    error = Signal(str)
    qwen_start = Signal()
    qwen_delta = Signal(str)
    qwen_end = Signal()
    finished = Signal()

    def __init__(self, config: AppConfig, user_text: str):
        super().__init__()
        self.config = config
        self.user_text = user_text
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    @Slot()
    def run(self):
        try:
            self._ask_ollama_stream()
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    def _ask_ollama_stream(self):
        self.status.emit(f"Qwen отвечает: {self.config.ollama_model}...")
        url = f"{self.config.ollama_base_url.rstrip('/')}/api/generate"
        prompt = (
            f"{self.config.system_prompt}\n\n"
            "Сообщение пользователя:\n"
            f"{self.user_text}\n\n"
            "Ответ:"
        )
        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.2,
                "num_predict": 500,
            },
        }

        has_answer = False
        self.qwen_start.emit()
        try:
            with requests.post(url, json=payload, stream=True, timeout=(10, 180)) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines(decode_unicode=True):
                    if self._stop_event.is_set():
                        break
                    if not raw_line:
                        continue
                    try:
                        data = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    piece = data.get("response", "")
                    if piece:
                        has_answer = True
                        self.qwen_delta.emit(piece)

                    if data.get("done", False):
                        break
        finally:
            if not has_answer:
                self.qwen_delta.emit("Пустой ответ от модели")
            self.qwen_end.emit()
            self.status.emit("Готово")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.thread = None
        self.worker = None
        self.text_thread = None
        self.text_worker = None
        self._current_qwen_answer = ""
        self._last_qwen_answer = ""
        self._chat_blocks = []
        self._current_qwen_block_index = None
        self.sidebar_expanded = True
        self.sidebar_widget = None
        self.sidebar_layout = None
        self.sidebar_content_widgets = []
        self.sidebar_toggle_button = None

        self.setWindowTitle("QwenOSR1")
        self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.resize(1100, 760)
        self.setMinimumSize(900, 600)

        app = QApplication.instance()
        if app is not None:
            app.setFont(QFont("Segoe UI", 10))
        self._apply_styles()

        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        self.chat.setObjectName("chatView")
        self.chat.setFocusPolicy(Qt.StrongFocus)
        self.chat.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        self.chat.installEventFilter(self)

        self.input_box = QTextEdit()
        self.input_box.setObjectName("messageInput")
        self.input_box.setPlaceholderText("Напишите сообщение для Qwen...")
        self.input_box.setFixedHeight(78)
        self.input_box.installEventFilter(self)


        self.send_text_button = QPushButton("➤")
        self.send_text_button.setObjectName("sendTextButton")
        self.send_text_button.setToolTip("Отправить текстовое сообщение")
        self.send_text_button.setFixedSize(46, 46)

        self.start_button = QPushButton("🎙️")
        self.start_button.setToolTip("Начать запись системного звука")
        self.send_audio_button = QPushButton("⇑")
        self.send_audio_button.setToolTip("Отправить накопленную запись в Qwen")
        self.clear_recording_button = QPushButton("⌫")
        self.clear_recording_button.setToolTip("Очистить накопленную запись и продолжить запись")
        self.clear_chat_button = QPushButton("🗑")
        self.clear_chat_button.setToolTip("Очистить чат")

        for button in [
            self.start_button,
            self.send_audio_button,
            self.clear_recording_button,
            self.clear_chat_button,
        ]:
            button.setObjectName("iconButton")
            button.setFixedSize(38, 38)

        self.start_button.setFixedSize(52, 52)

        self.start_button.setObjectName("recordButton")
        self.send_audio_button.setObjectName("audioSendButton")
        self.clear_recording_button.setObjectName("clearRecordingButton")

        self.send_audio_button.setEnabled(False)
        self.clear_recording_button.setEnabled(False)

        self.status_value = QLabel("Готово")
        self.status_value.setObjectName("sideValue")
        self.status_value.setWordWrap(True)
        self.buffer_value = QLabel("0.0 сек.")
        self.buffer_value.setObjectName("sideValue")
        self.model_value = QLabel(self.config.ollama_model)
        self.model_value.setObjectName("sideMuted")
        self.whisper_value = QLabel(self.config.whisper_model)
        self.whisper_value.setObjectName("sideMuted")
        self.ollama_value = QLabel(self.config.ollama_base_url)
        self.ollama_value.setObjectName("sideMuted")
        self.ollama_value.setWordWrap(True)

        self.start_button.clicked.connect(self.toggle_recording)
        self.send_audio_button.clicked.connect(self.send_recording)
        self.clear_recording_button.clicked.connect(self.clear_recording)
        self.clear_chat_button.clicked.connect(self.clear_chat)
        self.send_text_button.clicked.connect(self.send_text_message)

        self.setCentralWidget(self._build_ui())
        self._setup_shortcuts()

    def _build_ui(self) -> QWidget:
        root = QWidget()
        main = QHBoxLayout(root)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        sidebar = self._make_sidebar()
        chat_area = self._make_chat_area()
        main.addWidget(sidebar)
        main.addWidget(chat_area, 1)
        return root

    def _make_sidebar(self) -> QFrame:
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(270)
        self.sidebar_widget = side

        layout = QVBoxLayout(side)
        self.sidebar_layout = layout
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        title_box = QVBoxLayout()
        title_box.setContentsMargins(0, 0, 0, 0)
        title_box.setSpacing(2)

        self.side_title = QLabel("Qwen Chat")
        self.side_title.setObjectName("sideTitle")
        self.side_subtitle = QLabel("Локально через Ollama")
        self.side_subtitle.setObjectName("sideMuted")

        title_box.addWidget(self.side_title)
        title_box.addWidget(self.side_subtitle)

        self.sidebar_toggle_button = QPushButton("‹")
        self.sidebar_toggle_button.setObjectName("sidebarToggleButton")
        self.sidebar_toggle_button.setToolTip("Свернуть боковую панель")
        self.sidebar_toggle_button.setFixedSize(34, 34)
        self.sidebar_toggle_button.clicked.connect(self.toggle_sidebar)

        header.addLayout(title_box, 1)
        header.addWidget(self.sidebar_toggle_button)

        layout.addLayout(header)
        layout.addSpacing(8)

        model_section = self._side_section("LLM", self.model_value)
        whisper_section = self._side_section("Whisper", self.whisper_value)
        ollama_section = self._side_section("Ollama", self.ollama_value)

        self.sidebar_content_widgets = [
            self.side_subtitle,
            model_section,
            whisper_section,
            ollama_section,
        ]

        for section in [model_section, whisper_section, ollama_section]:
            layout.addWidget(section)

        layout.addStretch(1)

        self.side_hint = QLabel("Кто прочитал тот лох")
        self.side_hint.setObjectName("sideHint")
        self.side_hint.setWordWrap(True)
        self.sidebar_content_widgets.append(self.side_hint)
        layout.addWidget(self.side_hint)
        return side

    def toggle_sidebar(self):
        if not self.sidebar_widget:
            return

        self.sidebar_expanded = not self.sidebar_expanded
        if self.sidebar_expanded:
            self.sidebar_widget.setFixedWidth(270)
            if self.sidebar_layout is not None:
                self.sidebar_layout.setContentsMargins(14, 14, 14, 14)
                self.sidebar_layout.setSpacing(12)
            self.side_title.setVisible(True)
            self.side_title.setText("Qwen Chat")
            self.sidebar_toggle_button.setText("‹")
            self.sidebar_toggle_button.setToolTip("Свернуть боковую панель")
            for widget in self.sidebar_content_widgets:
                widget.setVisible(True)
        else:
            # Панель остается видимой, но превращается в узкую навигационную колонку.
            # В свернутом состоянии скрываем весь текст, чтобы кнопка не выталкивала layout.
            self.sidebar_widget.setFixedWidth(56)
            if self.sidebar_layout is not None:
                self.sidebar_layout.setContentsMargins(10, 14, 10, 14)
                self.sidebar_layout.setSpacing(8)
            self.side_title.setVisible(False)
            self.sidebar_toggle_button.setText("☰")
            self.sidebar_toggle_button.setToolTip("Развернуть боковую панель")
            for widget in self.sidebar_content_widgets:
                widget.setVisible(False)

    def _side_section(self, title: str, value_widget: QLabel) -> QFrame:
        box = QFrame()
        box.setObjectName("sideSection")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        label = QLabel(title)
        label.setObjectName("sideLabel")
        layout.addWidget(label)
        layout.addWidget(value_widget)
        return box

    def _make_chat_area(self) -> QFrame:
        area = QFrame()
        area.setObjectName("chatArea")
        layout = QVBoxLayout(area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("chatHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(22, 12, 22, 12)
        header_layout.setSpacing(10)

        header_title = QLabel("Qwen")
        header_title.setObjectName("chatHeaderTitle")
        header_subtitle = QLabel("Локальный ассистент")
        header_subtitle.setObjectName("chatHeaderMuted")

        header_text = QVBoxLayout()
        header_text.setContentsMargins(0, 0, 0, 0)
        header_text.setSpacing(0)
        header_text.addWidget(header_title)
        header_text.addWidget(header_subtitle)

        header_layout.addLayout(header_text)
        header_layout.addSpacing(18)
        header_layout.addWidget(self._top_status_pill("Статус", self.status_value))
        header_layout.addWidget(self._top_status_pill("Аудио записано", self.buffer_value))
        header_layout.addStretch(1)
        layout.addWidget(header)

        body = QFrame()
        body.setObjectName("chatBody")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(18, 14, 18, 16)
        body_layout.setSpacing(12)
        body_layout.addWidget(self.chat, 1)
        body_layout.addWidget(self._make_composer())

        layout.addWidget(body, 1)
        return area

    def _top_status_pill(self, title: str, value_widget: QLabel) -> QFrame:
        pill = QFrame()
        pill.setObjectName("topStatusPill")
        pill_layout = QHBoxLayout(pill)
        pill_layout.setContentsMargins(12, 7, 12, 7)
        pill_layout.setSpacing(6)

        title_label = QLabel(f"{title}:")
        title_label.setObjectName("topStatusLabel")

        value_widget.setObjectName("topStatusValue")
        value_widget.setWordWrap(False)
        value_widget.setMinimumWidth(90 if title == "Статус" else 64)

        pill_layout.addWidget(title_label)
        pill_layout.addWidget(value_widget)
        return pill

    def _make_composer(self) -> QFrame:
        composer = QFrame()
        composer.setObjectName("composer")
        layout = QHBoxLayout(composer)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(self.input_box, 1)

        buttons_row = QHBoxLayout()
        buttons_row.setContentsMargins(0, 0, 0, 0)
        buttons_row.setSpacing(6)
        buttons_row.addWidget(self.start_button)
        buttons_row.addWidget(self.send_audio_button)
        buttons_row.addWidget(self.clear_recording_button)
        buttons_row.addWidget(self.clear_chat_button)
        buttons_row.addWidget(self.send_text_button)

        layout.addLayout(buttons_row)
        return composer

    def _apply_styles(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #ffffff;
                color: #111827;
            }
            QFrame#sidebar {
                background: #f7f7f8;
                border-right: 1px solid #e5e7eb;
            }
            QFrame#sidebar QLabel {
                background: transparent;
                color: #111827;
            }
            QLabel#sideTitle {
                background: transparent;
                color: #111827;
                font-size: 20px;
                font-weight: 700;
            }
            QLabel#sideLabel {
                background: transparent;
                color: #6b7280;
                font-size: 11px;
                font-weight: 700;
                text-transform: uppercase;
            }
            QLabel#sideValue {
                background: transparent;
                color: #111827;
                font-size: 13px;
                font-weight: 600;
            }
            QLabel#sideMuted, QLabel#sideHint {
                background: transparent;
                color: #6b7280;
                font-size: 12px;
            }
            QFrame#sideSection {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 14px;
            }
            QFrame#sideSection QLabel {
                background: transparent;
            }
            QFrame#chatArea, QFrame#chatBody {
                background: #ffffff;
            }
            QFrame#chatHeader {
                background: #ffffff;
                border-bottom: 1px solid #ececf1;
            }
            QLabel#chatHeaderTitle {
                background: transparent;
                color: #111827;
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#chatHeaderMuted {
                background: transparent;
                color: #6b7280;
                font-size: 12px;
            }
            QFrame#topStatusPill {
                background: #f7f7f8;
                border: 1px solid #e5e7eb;
                border-radius: 14px;
            }
            QLabel#topStatusLabel {
                background: transparent;
                color: #6b7280;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#topStatusValue {
                background: transparent;
                color: #111827;
                font-size: 12px;
                font-weight: 700;
            }
            QTextEdit#chatView {
                background: #ffffff;
                border: none;
                padding: 6px;
                font-size: 15px;
                line-height: 1.6;
                selection-background-color: #c7d2fe;
                selection-color: #111827;
            }
            QFrame#composer {
                background: #ffffff;
                border: 1px solid #d9d9e3;
                border-radius: 22px;
            }
            QTextEdit#messageInput {
                background: #ffffff;
                border: none;
                padding: 9px;
                font-size: 14px;
            }
            QPushButton {
                background: #f4f4f5;
                color: #111827;
                border: 1px solid #e4e4e7;
                border-radius: 12px;
                font-size: 16px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #ececf1;
            }
            QPushButton:disabled {
                background: #f7f7f8;
                color: #a1a1aa;
                border-color: #eeeeee;
            }
            QPushButton#sidebarToggleButton {
                background: #ffffff;
                color: #374151;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
                font-size: 18px;
            }
            QPushButton#sidebarToggleButton:hover {
                background: #f3f4f6;
            }
            QPushButton#sendTextButton {
                background: #111827;
                color: white;
                border-color: #111827;
                border-radius: 15px;
                font-size: 20px;
            }
            QPushButton#sendTextButton:hover {
                background: #000000;
            }
            QPushButton#recordButton {
                background: #fff1f2;
                color: #be123c;
                border-color: #fecdd3;
            }
            QPushButton#recordButton:hover {
                background: #ffe4e6;
            }
            QPushButton#audioSendButton {
                background: #dcfce7;
                color: #166534;
                border-color: #bbf7d0;
            }

            QPushButton#audioSendButton:hover {
                background: #bbf7d0;
            }

            QPushButton#audioSendButton:disabled {
                background: #f3f4f6;
                color: #9ca3af;
                border-color: #e5e7eb;
            }
            QPushButton#clearRecordingButton {
                background: #fef9c3;
                color: #854d0e;
                border-color: #fde68a;
            }

            QPushButton#clearRecordingButton:hover {
                background: #fef08a;
            }

            QPushButton#clearRecordingButton:disabled {
                background: #f3f4f6;
                color: #9ca3af;
                border-color: #e5e7eb;
            }
            """
        )

    def _setup_shortcuts(self):
        copy_action = QAction(self)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.triggered.connect(self.copy_selected_text)
        self.addAction(copy_action)

    def copy_selected_text(self):
        focused_widget = QApplication.focusWidget()

        if focused_widget == self.input_box:
            self.input_box.copy()
            return

        cursor = self.chat.textCursor()
        if cursor.hasSelection():
            selected_text = cursor.selectedText().replace("\u2029", "\n")
            clipboard = QApplication.clipboard()
            clipboard.clear()
            clipboard.setText(selected_text)
            self.status_value.setText("Выделенный текст скопирован")
            return

        self.status_value.setText("Нет выделенного текста для копирования")

    def eventFilter(self, watched, event):
        if event.type() == QEvent.KeyPress:
            key = event.key()
            modifiers = event.modifiers()

            if watched == self.input_box:
                if key in (Qt.Key_Return, Qt.Key_Enter):
                    if modifiers & Qt.ShiftModifier:
                        return False
                    self.send_text_message()
                    return True

            if watched == self.chat:
                if key == Qt.Key_C and modifiers & Qt.ControlModifier:
                    self.copy_selected_text()
                    return True

        return super().eventFilter(watched, event)

    def toggle_recording(self):
        if self.thread is None:
            self.start_listening()
        else:
            self.stop_listening()

    def start_listening(self):
        if self.thread is not None:
            return

        self.thread = QThread()
        self.worker = AudioAnalyzerWorker(self.config)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.qwen_start.connect(self.start_qwen_stream)
        self.worker.qwen_delta.connect(self.append_qwen_delta)
        self.worker.qwen_end.connect(self.end_qwen_stream)
        self.worker.status.connect(self.set_status)
        self.worker.buffer_seconds.connect(self.set_buffer_seconds)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._thread_finished)

        self.start_button.setText("■")
        self.start_button.setToolTip("Остановить запись")
        self.start_button.setEnabled(True)
        self.send_audio_button.setEnabled(True)
        self.clear_recording_button.setEnabled(True)
        self.status_value.setText("Запуск записи...")
        self.thread.start()

    def send_recording(self):
        if self.worker:
            self.worker.request_send()
            self.status_value.setText("Отправка аудио...")

    def clear_recording(self):
        if self.worker:
            self.worker.request_clear_recording()
            self.buffer_value.setText("0.0 сек.")
            self.status_value.setText("Запись очищена")

    def stop_listening(self):
        if self.worker:
            self.worker.stop()
            self.status_value.setText("Остановка записи...")
            self.start_button.setEnabled(False)
            self.send_audio_button.setEnabled(False)
            self.clear_recording_button.setEnabled(False)

    def _thread_finished(self):
        self.thread = None
        self.worker = None
        self.start_button.setText("🎙️")
        self.start_button.setToolTip("Начать запись системного звука")
        self.start_button.setEnabled(True)
        self.send_audio_button.setEnabled(False)
        self.clear_recording_button.setEnabled(False)
        self.buffer_value.setText("0.0 сек.")
        self.status_value.setText("Запись остановлена")

    def send_text_message(self):
        text = self.input_box.toPlainText().strip()
        if not text or self.text_thread is not None:
            return

        self.input_box.clear()
        self.append_user_message(text)

        self.text_thread = QThread()
        self.text_worker = TextQwenWorker(self.config, text)
        self.text_worker.moveToThread(self.text_thread)

        self.text_thread.started.connect(self.text_worker.run)
        self.text_worker.qwen_start.connect(self.start_qwen_stream)
        self.text_worker.qwen_delta.connect(self.append_qwen_delta)
        self.text_worker.qwen_end.connect(self.end_qwen_stream)
        self.text_worker.status.connect(self.set_status)
        self.text_worker.error.connect(self.show_error)
        self.text_worker.finished.connect(self.text_thread.quit)
        self.text_worker.finished.connect(self.text_worker.deleteLater)
        self.text_thread.finished.connect(self.text_thread.deleteLater)
        self.text_thread.finished.connect(self._text_thread_finished)

        self.send_text_button.setEnabled(False)
        self.status_value.setText("Отправка сообщения...")
        self.text_thread.start()

    def _text_thread_finished(self):
        self.text_thread = None
        self.text_worker = None
        self.send_text_button.setEnabled(True)

    def _scroll_chat_to_bottom(self):
        self.chat.verticalScrollBar().setValue(self.chat.verticalScrollBar().maximum())

    def _message_block_html(self, title: str, body: str, *, align: str = "left") -> str:
        safe_body = self._html_escape(body)

        if align == "right":
            # Для QTextEdit на Windows table-разметка устойчивее, чем CSS display:inline-block.
            # Так сообщение пользователя не уезжает вправо и не ломает ширину чата.
            return (
                "<table width='100%' cellspacing='0' cellpadding='0' style='margin:14px 0;'>"
                "<tr>"
                "<td align='right'>"
                "<table cellspacing='0' cellpadding='0' style='border-collapse:separate;'>"
                "<tr>"
                "<td style='background:#f4f4f5;border:1px solid #ededf0;"
                "border-radius:18px;padding:11px 15px;color:#111827;"
                "line-height:1.55;'>"
                f"{safe_body}"
                "</td>"
                "</tr>"
                "</table>"
                "</td>"
                "</tr>"
                "</table>"
            )
        return (
            "<table width='100%' cellspacing='0' cellpadding='0' style='margin:14px 0;'>"
            "<tr>"
            "</td>"
            "<td valign='top' style='padding-left:2px;'>"
            "<div style='color:#111827;line-height:1.1;'>"
            f"<div style='font-weight:700;margin-bottom:4px;'>{self._html_escape(title)}</div>"
            f"<div>{safe_body}</div>"
            "</div>"
            "</td>"
            "</tr>"
            "</table>"
        )

    def _render_chat(self):
        html = (
            "<html><body style='background:#ffffff;font-family:Segoe UI,Arial,sans-serif;"
            "font-size:15px;line-height:1.6;margin:0;padding:0;'>"
            + "".join(self._chat_blocks)
            + "</body></html>"
        )
        self.chat.setHtml(html)
        self._scroll_chat_to_bottom()

    def append_user_message(self, text: str):
        self._chat_blocks.append(self._message_block_html("Вы", text, align="right"))
        self._render_chat()

    @Slot()
    def start_qwen_stream(self):
        self._current_qwen_answer = ""
        self._current_qwen_block_index = len(self._chat_blocks)
        self._chat_blocks.append(self._message_block_html("Qwen", "", align="left"))
        self._render_chat()

    @Slot(str)
    def append_qwen_delta(self, message: str):
        message = self._clean_model_text(message)
        self._current_qwen_answer += message
        if self._current_qwen_block_index is None:
            self.start_qwen_stream()

        self._chat_blocks[self._current_qwen_block_index] = self._message_block_html(
            "Qwen",
            self._current_qwen_answer,
            align="left",
        )
        self._render_chat()

    @staticmethod
    def _clean_model_text(value: str) -> str:
        return (
            value.replace("\u2028", "\n")
            .replace("\u2029", "\n")
            .replace("\u00a0", " ")
        )

    @Slot()
    def end_qwen_stream(self):
        self._last_qwen_answer = self._current_qwen_answer.strip()
        self._current_qwen_block_index = None
        self._render_chat()

    def clear_chat(self):
        self.chat.clear()
        self._chat_blocks = []
        self._current_qwen_answer = ""
        self._last_qwen_answer = ""
        self._current_qwen_block_index = None
        self.status_value.setText("Чат очищен")

    @Slot(str)
    def append_log(self, message: str):
        self._chat_blocks.append(message)
        self._render_chat()

    @Slot(str)
    def set_status(self, message: str):
        self.status_value.setText(message)

    @Slot(float)
    def set_buffer_seconds(self, seconds: float):
        self.buffer_value.setText(f"{seconds:.1f} сек.")

    @Slot(str)
    def show_error(self, message: str):
        self._chat_blocks.append(
            f"<div style='background:#fef2f2;border:1px solid #fecaca;border-radius:22px;"
            f"padding:14px 16px;margin:8px 0;'><b>Ошибка:</b> {self._html_escape(message)}</div>"
        )
        self._render_chat()
        QMessageBox.critical(self, "Ошибка", message)

    @staticmethod
    def _html_escape(value: str) -> str:
        return (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("\n", "<br>")
        )

    def closeEvent(self, event):
        self.stop_listening()
        if self.text_worker:
            self.text_worker.stop()
        event.accept()


def main():
    init_windows_ole_for_clipboard()

    if os.name == "nt":
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "qwen.desktop.chat.local"
            )

    app = QApplication([])

    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))

    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
