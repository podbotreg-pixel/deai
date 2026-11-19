#!/usr/bin/env python3
"""Настольный Qt-клиент DI-NET (правило GUI: Linux/Windows — Qt).

Поверх Rust-ядра через C-ABI мост `libdinet_desktop.so` — тот же
PhoneCore, что у Android-телефона. Три вкладки: чат, рой, сеть.

Запуск:
    cargo build -p dinet-desktop
    python3 desktop/qt_chat.py
    python3 desktop/qt_chat.py --shot /tmp/opencode/qt.png   # автодемо + скриншот
"""

import ctypes
import os
import sys
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class Core:
    """ctypes-обёртка над dinet_desktop (все строки — UTF-8, frees через dinet_free)."""

    def __init__(self, lib_path: str):
        self.lib = ctypes.CDLL(lib_path)
        strp = ctypes.c_char_p
        self.lib.dinet_open.argtypes = [strp]
        self.lib.dinet_open.restype = ctypes.c_int
        self.lib.dinet_turn.argtypes = [strp]
        self.lib.dinet_teach.argtypes = [strp, strp]
        self.lib.dinet_add_peer.argtypes = [strp]
        self.lib.dinet_add_peer.restype = ctypes.c_int
        self.lib.dinet_net.argtypes = [ctypes.c_uint32]
        self.lib.dinet_abi_version.restype = ctypes.c_uint32
        self.lib.dinet_watermark_ok.restype = ctypes.c_int
        self.lib.dinet_free.argtypes = [ctypes.c_void_p]
        for name in (
            "dinet_turn", "dinet_teach", "dinet_stats", "dinet_swarm_stats",
            "dinet_peers", "dinet_probe", "dinet_bg", "dinet_net",
        ):
            getattr(self.lib, name).restype = ctypes.c_void_p

    def _s(self, fn, *args) -> str:
        p = fn(*args)
        if not p:
            return ""
        try:
            return ctypes.string_at(p).decode("utf-8", "replace")
        finally:
            self.lib.dinet_free(ctypes.c_void_p(p))

    def open(self, directory: str) -> bool:
        return bool(self.lib.dinet_open(directory.encode()))

    def turn(self, text: str) -> str:
        return self._s(self.lib.dinet_turn, text.encode())

    def teach(self, prompt: str, answer: str) -> str:
        return self._s(self.lib.dinet_teach, prompt.encode(), answer.encode())

    def stats(self) -> str:
        return self._s(self.lib.dinet_stats)

    def swarm_stats(self) -> str:
        return self._s(self.lib.dinet_swarm_stats)

    def peers(self) -> str:
        return self._s(self.lib.dinet_peers)

    def add_peer(self, addr: str) -> bool:
        return bool(self.lib.dinet_add_peer(addr.encode()))

    def probe(self) -> str:
        return self._s(self.lib.dinet_probe)

    def bg(self) -> str:
        return self._s(self.lib.dinet_bg)

    def net(self, eras: int = 4) -> str:
        return self._s(self.lib.dinet_net, eras)

    def watermark_ok(self) -> bool:
        return bool(self.lib.dinet_watermark_ok())

    def abi(self) -> int:
        return int(self.lib.dinet_abi_version())


class Worker(QObject):
    """Сетевые вызовы (probe/net — секунды) в отдельном потоке, UI не висит."""

    done = pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            self.done.emit(self._fn())
        except Exception as exc:  # pragma: no cover — GUI-обвязка
            self.done.emit(f"ошибка: {exc}")


class MainWindow(QMainWindow):
    def __init__(self, core: Core):
        super().__init__()
        self.core = core
        self.setWindowTitle("DI-NET — децентрализованный чат")
        self.resize(900, 700)
        self._threads = []

        tabs = QTabWidget()
        tabs.addTab(self._chat_tab(), "Чат")
        tabs.addTab(self._swarm_tab(), "Рой")
        tabs.addTab(self._net_tab(), "Сеть")
        self.tabs = tabs
        self.setCentralWidget(tabs)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(5000)
        self.refresh()
        self.log(f"ядро открыто; ABI {self.core.abi()}")

    def _chat_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.stats_label = QLabel("...")
        self.stats_label.setWordWrap(True)
        v.addWidget(QLabel("Лента:"))
        v.addWidget(self.log_view, 1)
        v.addWidget(self.stats_label)

        row = QHBoxLayout()
        self.prompt = QLineEdit()
        self.prompt.setPlaceholderText("ход или контекст (4 слова)")
        ask = QPushButton("Спросить")
        ask.clicked.connect(self.ask)
        row.addWidget(self.prompt)
        row.addWidget(ask)
        v.addLayout(row)

        teach = QHBoxLayout()
        self.pair_p = QLineEdit()
        self.pair_p.setPlaceholderText("контекст: ай би ви ги")
        self.pair_a = QLineEdit()
        self.pair_a.setPlaceholderText("ответ: ди")
        save = QPushButton("Записать пару")
        save.clicked.connect(self.teach)
        teach.addWidget(self.pair_p)
        teach.addWidget(self.pair_a)
        teach.addWidget(save)
        v.addLayout(teach)
        return w

    def _swarm_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.swarm_label = QLabel("...")
        self.swarm_label.setWordWrap(True)
        self.swarm_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.wm_label = QLabel("")
        v.addWidget(QLabel("Состояние роя:"))
        v.addWidget(self.swarm_label)
        v.addStretch(1)
        v.addWidget(self.wm_label)
        probe = QPushButton("Обновить состав (probe)")
        probe.clicked.connect(lambda: self.background(self.core.probe, "probe"))
        v.addWidget(probe)
        self.action_label = QLabel("")
        self.action_label.setWordWrap(True)
        v.addWidget(self.action_label)
        return w

    def _net_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.peer_input = QLineEdit()
        self.peer_input.setPlaceholderText("10.0.2.2:9300")
        add = QPushButton("Добавить пир")
        add.clicked.connect(self.add_peer)
        self.peer_list = QListWidget()
        v.addWidget(QLabel("Адреса смотрителей:"))
        v.addWidget(self.peer_input)
        v.addWidget(add)
        v.addWidget(self.peer_list)
        row = QHBoxLayout()
        net = QPushButton("Сетевой вклад")
        net.clicked.connect(lambda: self.background(lambda: self.core.net(4), "net"))
        bg = QPushButton("Фоновый вклад")
        bg.clicked.connect(lambda: self.background(self.core.bg, "bg"))
        row.addWidget(net)
        row.addWidget(bg)
        v.addLayout(row)
        return w

    def log(self, text: str):
        self.log_view.append(text)

    def ask(self):
        text = self.prompt.text().strip()
        if not text:
            return
        self.log(f"ты: {text}")
        self.log(f"рой: {self.core.turn(text)}")
        self.prompt.clear()
        self.refresh()

    def teach(self):
        p, a = self.pair_p.text().strip(), self.pair_a.text().strip()
        if not p or not a:
            return
        self.log(f"учитель: {p} → {a}")
        self.log(f"система: {self.core.teach(p, a)}")
        self.pair_p.clear()
        self.pair_a.clear()
        self.refresh()

    def add_peer(self):
        addr = self.peer_input.text().strip()
        if addr and self.core.add_peer(addr):
            self.peer_input.clear()
        self.refresh()

    def background(self, fn, tag: str):
        """Долгий вызов в QThread: сразу пишем «...», итог — сигналом."""
        self.action_label.setText(f"{tag}: выполняется…")
        thread = QThread(self)
        worker = Worker(fn)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.done.connect(self._on_background_done)
        worker.done.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        self._threads.append((thread, worker))
        thread.start()

    def _on_background_done(self, text: str):
        self.action_label.setText(text)
        self.log(f"система: {text}")
        self.refresh()

    def refresh(self):
        self.stats_label.setText(self.core.stats())
        self.swarm_label.setText(self.core.swarm_stats())
        self.wm_label.setText(
            "водяной знак: подтверждён" if self.core.watermark_ok() else "водяной знак: нет (ждёт эру)"
        )
        peers = [p for p in self.core.peers().splitlines() if p.strip()]
        self.peer_list.clear()
        self.peer_list.addItems(peers)


def default_lib() -> str:
    root = Path(__file__).resolve().parent.parent
    for rel in (
        "target/debug/libdinet_desktop.so",
        "target/release/libdinet_desktop.so",
        "target/debug/libdinet_desktop.dylib",
        "target/debug/dinet_desktop.dll",
    ):
        p = root / rel
        if p.exists():
            return str(p)
    raise SystemExit("нет libdinet_desktop — сначала: cargo build -p dinet-desktop")


def main():
    args = sys.argv[1:]
    shot = None
    if "--shot" in args:
        shot = args[args.index("--shot") + 1]
    lib = args[args.index("--lib") + 1] if "--lib" in args else default_lib()
    directory = args[args.index("--dir") + 1] if "--dir" in args else str(
        Path(os.environ.get("XDG_DATA_HOME", "/tmp")) / "dinet-qt"
    )
    app = QApplication(args)
    core = Core(lib)
    if not core.open(directory):
        raise SystemExit(f"не удалось открыть ядро в {directory}")
    window = MainWindow(core)
    window.show()

    if shot:
        # автодемо: пара ходов + 4 пары → эра → знак; затем скриншот
        def demo():
            window.log(f"ты: привет, что помнишь?")
            window.log(f"рой: {core.turn('привет, что помнишь?')}")
            for _ in range(4):
                core.teach("ай би ви ги", "ди")
            window.log(f"ты: ай би ви ги")
            window.log(f"рой: {core.turn('ай би ви ги')}")
            window.refresh()
            app.processEvents()
            base = Path(shot)
            window.grab().save(str(base))
            window.tabs.setCurrentIndex(1)
            app.processEvents()
            window.grab().save(str(base.with_name(base.stem + "_swarm" + base.suffix)))
            window.tabs.setCurrentIndex(2)
            window.peer_input.setText("10.0.2.2:9300")
            app.processEvents()
            window.grab().save(str(base.with_name(base.stem + "_net" + base.suffix)))
            app.quit()

        QTimer.singleShot(1200, demo)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
