"""gazeGait pull GUI — pick Quest / Neon ADB devices, list Quest JSON + Neon exports."""
from __future__ import annotations

import sys
from typing import Optional

from PySide6.QtCore import Qt, QSettings, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QCheckBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from adb_util import AdbDevice, fetch_device_name, find_adb, list_devices
from device_files import (
    BOUTS,
    INTERACTIONS,
    NeonExport,
    QuestJsonFolder,
    assign_neons_to_interactions,
    cursor_stream_label,
    filter_quest_by_sub,
    fmt_duration,
    list_neon_exports,
    list_quest_json_folders,
    quest_folder_interaction_windows,
    quest_names_for_folder,
)
from pull_jobs import (
    PullPlan,
    make_plans_from_gui,
    run_pulls,
)


class NumericItem(QTableWidgetItem):
    def __lt__(self, other):
        try:
            a = float(self.data(Qt.UserRole))
        except (TypeError, ValueError):
            a = -1.0
        try:
            b = float(other.data(Qt.UserRole))
        except (TypeError, ValueError):
            b = -1.0
        return a < b

ROLES = ("", "Quest", "Neon")
COL_SERIAL, COL_STATE, COL_LINK, COL_MODEL, COL_NAME, COL_ROLE = range(6)
HEADERS = ("serial", "state", "link", "model", "name", "role")


class RefreshWorker(QThread):
    done = Signal(object, object)  # list[AdbDevice] | None, error str | None

    def __init__(self, adb: str, fetch_names: bool, parent=None):
        super().__init__(parent)
        self._adb = adb
        self._fetch_names = fetch_names

    def run(self):
        try:
            devices = list_devices(self._adb)
            if self._fetch_names:
                for d in devices:
                    if d.state != "device":
                        continue
                    try:
                        d.friendly_name = fetch_device_name(self._adb, d.serial)
                    except Exception as e:
                        d.friendly_name = f"(name failed: {e})"
            self.done.emit(devices, None)
        except Exception as e:
            self.done.emit(None, str(e))


class QuestListWorker(QThread):
    done = Signal(object, object)

    def __init__(self, adb: str, serial: str, parent=None):
        super().__init__(parent)
        self._adb = adb
        self._serial = serial

    def run(self):
        try:
            rows = list_quest_json_folders(self._adb, self._serial)
            self.done.emit(rows, None)
        except Exception as e:
            self.done.emit(None, str(e))


class NeonListWorker(QThread):
    done = Signal(object, object, object)  # rows, root_or_msg, err

    def __init__(self, adb: str, serial: str, parent=None):
        super().__init__(parent)
        self._adb = adb
        self._serial = serial

    def run(self):
        try:
            rows, root = list_neon_exports(self._adb, self._serial)
            self.done.emit(rows, root, None)
        except Exception as e:
            self.done.emit(None, None, str(e))


class QuestMatchWorker(QThread):
    done = Signal(object, object)  # dict[str, TimeWindow] | None, err

    def __init__(self, adb: str, serial: str, folder: QuestJsonFolder, parent=None):
        super().__init__(parent)
        self._adb = adb
        self._serial = serial
        self._folder = folder

    def run(self):
        try:
            windows = quest_folder_interaction_windows(
                self._adb, self._serial, self._folder
            )
            self.done.emit(windows, None)
        except Exception as e:
            self.done.emit(None, str(e))


class PullWorker(QThread):
    progress = Signal(str)
    done = Signal(object, object)

    def __init__(
        self,
        adb: str,
        quest_serial: str,
        neon_serial: Optional[str],
        plans: list[PullPlan],
        overwrite: bool,
        parent=None,
    ):
        super().__init__(parent)
        self._adb = adb
        self._quest_serial = quest_serial
        self._neon_serial = neon_serial
        self._plans = plans
        self._overwrite = overwrite

    def run(self):
        try:
            rels = run_pulls(
                self._adb,
                self._quest_serial,
                self._neon_serial,
                self._plans,
                overwrite=self._overwrite,
                on_line=self.progress.emit,
            )
            self.done.emit(rels, None)
        except Exception as e:
            self.done.emit(None, str(e))


class PullGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("gazeGait pull — Quest JSON / Neon export")
        self.resize(1280, 760)
        self._settings = QSettings("gazeGait", "00_pulling")
        self._adb = ""
        self._devices: list[AdbDevice] = []
        raw_roles = self._settings.value("roles", {}) or {}
        if not isinstance(raw_roles, dict):
            raw_roles = {}
        self._roles = {
            str(k): str(v) for k, v in raw_roles.items() if str(v) in ("Quest", "Neon")
        }
        self._worker: Optional[RefreshWorker] = None
        self._quest_worker: Optional[QuestListWorker] = None
        self._neon_worker: Optional[NeonListWorker] = None
        self._quest_rows: list[QuestJsonFolder] = []
        self._neon_rows: list[NeonExport] = []
        self._neon_root: str = ""
        self._match_worker: Optional[QuestMatchWorker] = None
        self._pull_worker: Optional[PullWorker] = None
        self._neon_filtered: bool = False
        self._neon_to_interaction: dict[str, str] = {}
        self._neon_bout: dict[str, str] = {}

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        top = QGroupBox("ADB devices")
        top_l = QVBoxLayout(top)
        row = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        self.chk_names = QCheckBox("Show device names")
        self.chk_names.setChecked(bool(self._settings.value("show_names", False, type=bool)))
        self.chk_names.setToolTip(
            "Query Android device_name / bluetooth_name (slower). "
            "Model from `adb devices -l` is always shown."
        )
        self.adb_label = QLabel("adb: —")
        self.adb_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row.addWidget(self.btn_refresh)
        row.addWidget(self.chk_names)
        row.addStretch(1)
        row.addWidget(self.adb_label)
        top_l.addLayout(row)

        self.table = QTableWidget(0, len(HEADERS))
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(COL_SERIAL, QHeaderView.Stretch)
        hdr.setSectionResizeMode(COL_STATE, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(COL_LINK, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(COL_MODEL, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(COL_NAME, QHeaderView.Stretch)
        hdr.setSectionResizeMode(COL_ROLE, QHeaderView.ResizeToContents)
        top_l.addWidget(self.table)

        self.assign_label = QLabel("Quest: —    Neon: —")
        self.assign_label.setWordWrap(True)
        hint = QLabel(
            "Mark Quest vs Neon, list, set bout + interaction on each Neon row, then Pull."
        )
        hint.setStyleSheet("color: gray;")
        top_l.addWidget(self.assign_label)
        top_l.addWidget(hint)
        root.addWidget(top)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_quest_panel())
        split.addWidget(self._build_neon_panel())
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        root.addWidget(split, stretch=1)

        pull_box = QGroupBox("Pull to data/participants")
        pull_l = QVBoxLayout(pull_box)
        pull_row = QHBoxLayout()
        self.btn_pull = QPushButton("Pull selected")
        self.btn_pull.setToolTip(
            "Pulls only the selected Quest JSON row(s). Uses bout + interaction from each "
            "selected Neon row. Quest → …/<bout>/<interaction>/00_raw/Quest/  "
            "Neon → …/00_raw/Motorola/."
        )
        self.dest_label = QLabel("Dest: select a Quest folder")
        self.dest_label.setWordWrap(True)
        self.dest_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        pull_row.addWidget(self.btn_pull)
        pull_row.addWidget(self.dest_label, stretch=1)
        pull_l.addLayout(pull_row)
        self.pull_status = QLabel("")
        self.pull_status.setWordWrap(True)
        pull_l.addWidget(self.pull_status)
        root.addWidget(pull_box)

        self.btn_refresh.clicked.connect(self.refresh)
        self.chk_names.toggled.connect(self._on_names_toggled)
        self.btn_pull.clicked.connect(self.pull_selected)

        self._resolve_adb()
        self.refresh()

    def _resolve_adb(self) -> None:
        try:
            self._adb = find_adb()
            self.adb_label.setText(f"adb: {self._adb}")
        except Exception as e:
            self._adb = ""
            self.adb_label.setText("adb: not found")
            QMessageBox.warning(self, "adb", str(e))

    def _on_names_toggled(self, checked: bool) -> None:
        self._settings.setValue("show_names", bool(checked))
        if checked:
            self.refresh()
        else:
            self._fill_table(self._devices)

    def refresh(self) -> None:
        if not self._adb:
            self._resolve_adb()
        if not self._adb:
            return
        if self._worker and self._worker.isRunning():
            return
        self.btn_refresh.setEnabled(False)
        self.btn_refresh.setText("Refreshing…")
        self._worker = RefreshWorker(self._adb, self.chk_names.isChecked(), self)
        self._worker.done.connect(self._on_refresh_done)
        self._worker.start()

    def _on_refresh_done(self, devices, err) -> None:
        self.btn_refresh.setEnabled(True)
        self.btn_refresh.setText("Refresh")
        if err:
            QMessageBox.warning(self, "adb devices", str(err))
            return
        self._devices = list(devices or [])
        for d in self._devices:
            if d.serial not in self._roles and d.guessed_role == "quest":
                if "Quest" not in self._roles.values():
                    self._roles[d.serial] = "Quest"
        self._fill_table(self._devices)
        self._save_roles()

    def _fill_table(self, devices: list[AdbDevice]) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        show_names = self.chk_names.isChecked()
        self.table.setColumnHidden(COL_NAME, not show_names)
        for d in devices:
            r = self.table.rowCount()
            self.table.insertRow(r)
            for col, text in (
                (COL_SERIAL, d.serial),
                (COL_STATE, d.state),
                (COL_LINK, d.link),
                (COL_MODEL, d.model),
                (COL_NAME, d.friendly_name if show_names else ""),
            ):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if col == COL_SERIAL:
                    item.setData(Qt.UserRole, d.serial)
                self.table.setItem(r, col, item)
            combo = QComboBox()
            combo.addItems(["—", "Quest", "Neon"])
            role = self._roles.get(d.serial, "")
            combo.setCurrentIndex(ROLES.index(role) if role in ROLES else 0)
            combo.currentTextChanged.connect(
                lambda text, serial=d.serial: self._on_role_changed(serial, text)
            )
            self.table.setCellWidget(r, COL_ROLE, combo)
        self.table.blockSignals(False)
        self._refresh_assign_label()

    def _on_role_changed(self, serial: str, text: str) -> None:
        role = "" if text in ("—", "") else text
        if role in ("Quest", "Neon"):
            for other, existing in list(self._roles.items()):
                if existing == role and other != serial:
                    self._roles.pop(other, None)
        if role:
            self._roles[serial] = role
        else:
            self._roles.pop(serial, None)
        self._save_roles()
        self._sync_role_combos()
        self._refresh_assign_label()

    def _sync_role_combos(self) -> None:
        for r in range(self.table.rowCount()):
            item = self.table.item(r, COL_SERIAL)
            combo = self.table.cellWidget(r, COL_ROLE)
            if item is None or combo is None:
                continue
            want = self._roles.get(item.text(), "")
            combo.blockSignals(True)
            combo.setCurrentIndex(ROLES.index(want) if want in ROLES else 0)
            combo.blockSignals(False)

    def _save_roles(self) -> None:
        self._settings.setValue("roles", dict(self._roles))

    def _refresh_assign_label(self) -> None:
        quest = next((s for s, r in self._roles.items() if r == "Quest"), "—")
        neon = next((s for s, r in self._roles.items() if r == "Neon"), "—")
        n = len(self._devices)
        self.assign_label.setText(
            f"{n} device(s)    Quest: {quest}    Neon: {neon}"
        )

    def assigned_quest(self) -> Optional[str]:
        return next((s for s, r in self._roles.items() if r == "Quest"), None)

    def assigned_neon(self) -> Optional[str]:
        return next((s for s, r in self._roles.items() if r == "Neon"), None)

    def _build_quest_panel(self) -> QWidget:
        box = QGroupBox("Quest JSON")
        lay = QVBoxLayout(box)
        row = QHBoxLayout()
        self.quest_search = QLineEdit()
        self.quest_search.setPlaceholderText("participant (sub), e.g. 1")
        self.quest_search.setClearButtonEnabled(True)
        self.chk_quest_latest = QCheckBox("Latest per cursor/stream only")
        self.chk_quest_latest.setChecked(
            bool(self._settings.value("quest_latest_only", True, type=bool))
        )
        self.chk_quest_latest.setToolTip(
            "When checked, hide older takes and show only the newest JSON for each "
            "_cursorX_streamY_ pair (9 per bout). Uncheck to list every JSON on the Quest."
        )
        self.btn_quest_list = QPushButton("List Quest")
        row.addWidget(QLabel("sub:"))
        row.addWidget(self.quest_search, stretch=1)
        row.addWidget(self.chk_quest_latest)
        row.addWidget(self.btn_quest_list)
        lay.addLayout(row)
        self.quest_table = QTableWidget(0, 6)
        self.quest_table.setHorizontalHeaderLabels(
            ("sub", "subsub", "bout", "cursor/stream", "apk", "json")
        )
        self.quest_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.quest_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.quest_table.verticalHeader().setVisible(False)
        self.quest_table.setSortingEnabled(True)
        qh = self.quest_table.horizontalHeader()
        qh.setSectionResizeMode(QHeaderView.ResizeToContents)
        qh.setSectionResizeMode(5, QHeaderView.Stretch)
        self.quest_table.setToolTip(
            "Ctrl/Shift-click JSON rows to choose what to pull. "
            "With “Latest per cursor/stream only”, older takes are hidden unless you uncheck it."
        )
        lay.addWidget(self.quest_table)
        self.quest_status = QLabel("Assign Quest, type sub, List.")
        self.quest_status.setWordWrap(True)
        lay.addWidget(self.quest_status)
        self.btn_quest_list.clicked.connect(self.list_quest)
        self.quest_search.textChanged.connect(self._apply_quest_filter)
        self.quest_search.returnPressed.connect(self.list_quest)
        self.chk_quest_latest.toggled.connect(self._on_quest_latest_toggled)
        self.quest_table.itemSelectionChanged.connect(self._update_dest_label)
        return box

    def _build_neon_panel(self) -> QWidget:
        box = QGroupBox("Neon Export")
        lay = QVBoxLayout(box)
        row = QHBoxLayout()
        self.btn_neon_list = QPushButton("List Neon exports")
        self.btn_neon_match = QPushButton("Match selected Quest")
        self.btn_neon_all = QPushButton("Show all")
        self.btn_neon_match.setToolTip(
            "Guess interaction from Quest overlap. You can still change bout / interaction "
            "in the Neon table before Pull."
        )
        row.addWidget(self.btn_neon_list)
        row.addWidget(self.btn_neon_match)
        row.addWidget(self.btn_neon_all)
        row.addStretch(1)
        lay.addLayout(row)
        self.neon_table = QTableWidget(0, 7)
        self.neon_table.setHorizontalHeaderLabels(
            ("folder", "bout", "interaction", "dest", "started", "duration", "note")
        )
        self.neon_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.neon_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.neon_table.verticalHeader().setVisible(False)
        self.neon_table.setSortingEnabled(False)
        nh = self.neon_table.horizontalHeader()
        nh.setSectionResizeMode(0, QHeaderView.Stretch)
        nh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        nh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        nh.setSectionResizeMode(3, QHeaderView.Stretch)
        nh.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        nh.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        nh.setSectionResizeMode(6, QHeaderView.Stretch)
        lay.addWidget(self.neon_table)
        self.neon_status = QLabel(
            "Assign Neon phone, then List. Set bout + interaction on each row, then Pull. "
            "Companion → export to Documents/Neon Export first."
        )
        self.neon_status.setWordWrap(True)
        lay.addWidget(self.neon_status)
        self.btn_neon_list.clicked.connect(self.list_neon)
        self.btn_neon_match.clicked.connect(self.match_neon_to_quest)
        self.btn_neon_all.clicked.connect(self._show_all_neons)
        self.neon_table.itemSelectionChanged.connect(self._update_dest_label)
        return box

    def list_quest(self) -> None:
        serial = self.assigned_quest()
        if not self._adb or not serial:
            QMessageBox.information(self, "Quest", "Assign a Quest device first.")
            return
        if self._quest_worker and self._quest_worker.isRunning():
            return
        self.btn_quest_list.setEnabled(False)
        self.quest_status.setText(f"Listing on {serial}…")
        self._quest_worker = QuestListWorker(self._adb, serial, self)
        self._quest_worker.done.connect(self._on_quest_listed)
        self._quest_worker.start()

    def _on_quest_listed(self, rows, err) -> None:
        self.btn_quest_list.setEnabled(True)
        if err:
            self.quest_status.setText(f"Quest list failed: {err}")
            QMessageBox.warning(self, "Quest JSON", str(err))
            return
        self._quest_rows = list(rows or [])
        self._apply_quest_filter()

    def _on_quest_latest_toggled(self, checked: bool) -> None:
        self._settings.setValue("quest_latest_only", bool(checked))
        self._apply_quest_filter()

    def _quest_latest_only(self) -> bool:
        return self.chk_quest_latest.isChecked()

    def _apply_quest_filter(self) -> None:
        shown = filter_quest_by_sub(self._quest_rows, self.quest_search.text())
        latest_only = self._quest_latest_only()
        self.quest_table.setSortingEnabled(False)
        self.quest_table.setRowCount(0)
        n_json = 0
        n_skipped = 0
        for rec in shown:
            names = quest_names_for_folder(rec, latest_only)
            if latest_only:
                n_skipped += rec.json_skipped
            for name in names:
                r = self.quest_table.rowCount()
                self.quest_table.insertRow(r)
                n_json += 1
                pair = cursor_stream_label(name)
                vals = (
                    str(rec.sub),
                    str(rec.subsub),
                    rec.speed,
                    pair,
                    rec.package_label,
                    name,
                )
                for c, text in enumerate(vals):
                    if c in (0, 1):
                        item = NumericItem(text)
                        item.setData(Qt.UserRole, rec.sub if c == 0 else rec.subsub)
                    else:
                        item = QTableWidgetItem(text)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    if c == 5:
                        item.setData(Qt.UserRole, rec.remote_dir)
                        item.setToolTip(f"{rec.remote_dir}/{name}")
                    self.quest_table.setItem(r, c, item)
        self.quest_table.setSortingEnabled(True)
        q = self.quest_search.text().strip()
        hid = ""
        if latest_only and n_skipped:
            hid = f"; hid {n_skipped} older"
        elif not latest_only:
            hid = "; all JSON on Quest"
        if not self._quest_rows:
            self.quest_status.setText("No Quest JSON folders found (Main/Practice/Pro).")
        elif q:
            self.quest_status.setText(
                f"sub {q}: {n_json} json"
                f" ({'latest per pair' if latest_only else 'all files'}{hid})"
            )
        else:
            self.quest_status.setText(
                f"{n_json} json on Quest — type sub to filter"
                f" ({'latest per pair' if latest_only else 'all files'}{hid})"
            )
        self._update_dest_label()

    def list_neon(self) -> None:
        serial = self.assigned_neon()
        if not self._adb or not serial:
            QMessageBox.information(self, "Neon", "Assign a Neon phone first.")
            return
        if self._neon_worker and self._neon_worker.isRunning():
            return
        self.btn_neon_list.setEnabled(False)
        self.neon_status.setText(f"Listing Neon exports on {serial}…")
        self._neon_worker = NeonListWorker(self._adb, serial, self)
        self._neon_worker.done.connect(self._on_neon_listed)
        self._neon_worker.start()

    def _on_neon_listed(self, rows, root, err) -> None:
        self.btn_neon_list.setEnabled(True)
        if err:
            self.neon_status.setText(f"Neon list failed: {err}")
            QMessageBox.warning(self, "Neon Export", str(err))
            return
        self._neon_rows = list(rows or [])
        self._neon_root = str(root or "")
        self._neon_to_interaction = {}
        self._neon_bout = {}
        self._fill_neon_table(self._neon_rows)
        if not self._neon_rows:
            self.neon_status.setText(str(root or "No exports found."))
        else:
            durs = [x.duration_s for x in self._neon_rows if x.duration_s is not None]
            hint = f"  longest {fmt_duration(max(durs))}" if durs else ""
            self.neon_status.setText(
                f"{len(self._neon_rows)} export(s) in {root}{hint}  "
                "(newest first; Match selected Quest to hide extra record presses)"
            )

    def _fill_neon_table(
        self,
        rows: list[NeonExport],
        notes: Optional[dict[str, str]] = None,
        *,
        select_all: bool = False,
    ) -> None:
        notes = notes or {}
        self._neon_filtered = rows is not self._neon_rows and len(rows) != len(self._neon_rows)
        folder = self._selected_quest_folder()
        quest_bout = folder.speed if folder is not None else ""
        self.neon_table.setSortingEnabled(False)
        self.neon_table.setRowCount(0)
        for rec in rows:
            r = self.neon_table.rowCount()
            self.neon_table.insertRow(r)
            dur = fmt_duration(rec.duration_s)
            bout = self._neon_bout.get(rec.remote_dir) or quest_bout
            inter = self._neon_to_interaction.get(rec.remote_dir, "")
            note = notes.get(rec.remote_dir) or rec.error or (rec.recording_id[:8] if rec.recording_id else "")
            name_item = QTableWidgetItem(rec.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            name_item.setData(Qt.UserRole, rec.remote_dir)
            name_item.setToolTip(rec.remote_dir)
            self.neon_table.setItem(r, 0, name_item)
            self.neon_table.setCellWidget(
                r, 1, self._make_assign_combo(BOUTS, bout, rec.remote_dir, "bout")
            )
            self.neon_table.setCellWidget(
                r, 2, self._make_assign_combo(INTERACTIONS, inter, rec.remote_dir, "interaction")
            )
            dest_item = QTableWidgetItem("")
            dest_item.setFlags(dest_item.flags() & ~Qt.ItemIsEditable)
            self.neon_table.setItem(r, 3, dest_item)
            started = QTableWidgetItem(rec.started_local)
            started.setFlags(started.flags() & ~Qt.ItemIsEditable)
            self.neon_table.setItem(r, 4, started)
            dur_item = NumericItem(dur)
            sec = rec.duration_s if rec.duration_s is not None else -1.0
            dur_item.setData(Qt.UserRole, float(sec))
            dur_item.setToolTip(f"{rec.duration_s:.1f} s" if rec.duration_s is not None else "")
            dur_item.setFlags(dur_item.flags() & ~Qt.ItemIsEditable)
            self.neon_table.setItem(r, 5, dur_item)
            note_item = QTableWidgetItem(note)
            note_item.setFlags(note_item.flags() & ~Qt.ItemIsEditable)
            self.neon_table.setItem(r, 6, note_item)
        self._refresh_neon_dest_cells()
        if select_all and self.neon_table.rowCount() > 0:
            self.neon_table.selectAll()
        self._update_dest_label()

    def _make_assign_combo(
        self, values: tuple[str, ...], current: str, remote: str, kind: str
    ) -> QComboBox:
        cb = QComboBox()
        cb.addItem("")
        cb.addItems(list(values))
        cb.blockSignals(True)
        if current:
            idx = cb.findText(current)
            if idx >= 0:
                cb.setCurrentIndex(idx)
        cb.blockSignals(False)
        cb.currentTextChanged.connect(
            lambda text, rem=remote, k=kind: self._on_neon_combo(rem, k, text)
        )
        return cb

    def _on_neon_combo(self, remote: str, kind: str, text: str) -> None:
        store = self._neon_bout if kind == "bout" else self._neon_to_interaction
        if text:
            store[remote] = text
        else:
            store.pop(remote, None)
        self._refresh_neon_dest_cells()
        self._sync_dest_label()

    def _neon_row_bout(self, row: int) -> str:
        w = self.neon_table.cellWidget(row, 1)
        return w.currentText().strip() if isinstance(w, QComboBox) else ""

    def _neon_row_interaction(self, row: int) -> str:
        w = self.neon_table.cellWidget(row, 2)
        return w.currentText().strip() if isinstance(w, QComboBox) else ""

    def _selected_table_rows(self) -> list[int]:
        sel = self.neon_table.selectionModel().selectedRows() if self.neon_table.selectionModel() else []
        rows = [i.row() for i in sel]
        if not rows and self.neon_table.rowCount() == 1:
            rows = [0]
        return rows

    def _neon_at_row(self, row: int) -> Optional[NeonExport]:
        item = self.neon_table.item(row, 0)
        if item is None:
            return None
        remote = item.data(Qt.UserRole)
        for rec in self._neon_rows:
            if rec.remote_dir == remote:
                return rec
        return None

    def _assignments_from_rows(self, rows: list[int]) -> list[tuple[NeonExport, str, str]]:
        out: list[tuple[NeonExport, str, str]] = []
        seen: set[str] = set()
        for row in rows:
            rec = self._neon_at_row(row)
            if rec is None or rec.remote_dir in seen:
                continue
            seen.add(rec.remote_dir)
            out.append((rec, self._neon_row_bout(row), self._neon_row_interaction(row)))
        return out

    def _complete_neon_assignments(self) -> list[tuple[NeonExport, str, str]]:
        rows = list(range(self.neon_table.rowCount()))
        return [
            (rec, bout, inter)
            for rec, bout, inter in self._assignments_from_rows(rows)
            if bout and inter
        ]

    def _selected_neon_assignments(self) -> list[tuple[NeonExport, str, str]]:
        selected = self._assignments_from_rows(self._selected_table_rows())
        if selected:
            return selected
        return self._complete_neon_assignments()

    def _update_dest_label(self) -> None:
        self._refresh_neon_dest_cells()
        self._sync_dest_label()

    def _sync_dest_label(self) -> None:
        folder = self._selected_quest_folder()
        selections = self._selected_quest_jsons()
        assignments = self._selected_neon_assignments()
        if folder is None:
            self.dest_label.setText(
                "Dest: select Quest JSON row(s) (Ctrl/Shift for multiple)"
            )
            return
        n_quest = len(selections) if selections else len(
            quest_names_for_folder(folder, self._quest_latest_only())
        )
        if not assignments:
            self.dest_label.setText(
                f"Dest: participant{folder.sub}/<bout>/<interaction>/00_raw/  "
                f"Quest {n_quest} json selected — set bout + interaction on Neon rows"
            )
            return
        bits = []
        for rec, bout, inter in assignments:
            bits.append(f"{bout or '?'}/{inter or '?'} ← {rec.name}")
        self.dest_label.setText("Dest: " + "  |  ".join(bits))

    def pull_selected(self) -> None:
        selections = self._selected_quest_jsons()
        if not selections:
            QMessageBox.information(
                self,
                "Pull",
                "Select one or more Quest JSON rows (Ctrl/Shift-click), then Pull.",
            )
            return
        remotes = {rec.remote_dir for rec, _ in selections}
        if len(remotes) > 1:
            QMessageBox.information(
                self,
                "Pull",
                "Select JSON from one Quest folder only (same sub-subsub bout).",
            )
            return
        folder = selections[0][0]
        selected_names = [name for _, name in selections]
        if not selected_names:
            QMessageBox.information(self, "Pull", f"No JSON in {folder.folder}.")
            return
        quest_serial = self.assigned_quest()
        if not self._adb or not quest_serial:
            QMessageBox.information(self, "Pull", "Assign a Quest device first.")
            return
        assignments = self._selected_neon_assignments()
        missing = [rec.name for rec, bout, inter in assignments if not bout or not inter]
        if missing:
            QMessageBox.information(
                self,
                "Pull",
                "Set bout and interaction on each selected Neon row:\n  "
                + "\n  ".join(missing),
            )
            return
        neon_serial = self.assigned_neon() if assignments else None
        if assignments and not neon_serial:
            QMessageBox.information(self, "Pull", "Assign a Neon phone first.")
            return
        try:
            plans = make_plans_from_gui(
                folder,
                assignments,
                selected_names,
                only_selected_quest=True,
            )
        except Exception as e:
            QMessageBox.warning(self, "Pull", str(e))
            return
        if not plans:
            QMessageBox.information(self, "Pull", "Nothing to pull.")
            return

        existing = []
        for plan in plans:
            existing.extend(
                plan.quest_dst / n for n in plan.quest_names if (plan.quest_dst / n).exists()
            )
            if plan.neon is not None and (plan.moto_dst / plan.neon.name).exists():
                existing.append(plan.moto_dst / plan.neon.name)
        overwrite = False
        if existing:
            r = QMessageBox.question(
                self,
                "Overwrite?",
                "Already on disk:\n  "
                + "\n  ".join(p.name for p in existing[:8])
                + ("\n  …" if len(existing) > 8 else "")
                + "\n\nOverwrite?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if r != QMessageBox.Yes:
                return
            overwrite = True

        lines = []
        for plan in plans:
            neon_n = plan.neon.name if plan.neon else "(no neon)"
            lines.append(
                f"  {plan.speed}/{plan.interaction} ← {neon_n}  "
                f"({len(plan.quest_names)} json)"
            )
        confirm = QMessageBox.question(
            self,
            "Pull",
            f"Pull {len(plans)} into participant{folder.sub}/\n\n" + "\n".join(lines),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if confirm != QMessageBox.Yes:
            return
        if self._pull_worker and self._pull_worker.isRunning():
            return
        self.btn_pull.setEnabled(False)
        self.pull_status.setText("Pulling…")
        self._pull_worker = PullWorker(
            self._adb, quest_serial, neon_serial, plans, overwrite, self
        )
        self._pull_worker.progress.connect(self._on_pull_progress)
        self._pull_worker.done.connect(self._on_pull_done)
        self._pull_worker.start()

    def _on_pull_progress(self, line: str) -> None:
        self.pull_status.setText(line[:240])

    def _on_pull_done(self, rel, err) -> None:
        self.btn_pull.setEnabled(True)
        if err:
            self.pull_status.setText(f"Pull failed: {err}")
            QMessageBox.warning(self, "Pull", str(err))
            return
        rels = rel if isinstance(rel, list) else [rel]
        text = "\n".join(f"{r}/00_raw/" for r in rels)
        self.pull_status.setText(f"Pulled {len(rels)} bout(s)")
        QMessageBox.information(self, "Pull", f"Done.\n{text}")

    def _refresh_neon_dest_cells(self) -> None:
        folder = self._selected_quest_folder()
        sub = folder.sub if folder is not None else None
        self.neon_table.blockSignals(True)
        try:
            for r in range(self.neon_table.rowCount()):
                item0 = self.neon_table.item(r, 0)
                if item0 is None:
                    continue
                bout = self._neon_row_bout(r)
                inter = self._neon_row_interaction(r)
                dest = (
                    f"participant{sub}/{bout}/{inter}/00_raw/Motorola/"
                    if sub is not None and bout and inter
                    else ""
                )
                item = self.neon_table.item(r, 3)
                if item is None:
                    item = QTableWidgetItem()
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.neon_table.setItem(r, 3, item)
                item.setText(dest)
                item.setToolTip((dest + item0.text()) if dest else "")
        finally:
            self.neon_table.blockSignals(False)

    def _selected_quest_jsons(self) -> list[tuple[QuestJsonFolder, str]]:
        sel = (
            self.quest_table.selectionModel().selectedRows()
            if self.quest_table.selectionModel()
            else []
        )
        out: list[tuple[QuestJsonFolder, str]] = []
        seen: set[tuple[str, str]] = set()
        for idx in sel:
            item = self.quest_table.item(idx.row(), 5)
            if item is None:
                continue
            remote = item.data(Qt.UserRole)
            name = item.text().strip()
            if not remote or not name:
                continue
            key = (str(remote), name)
            if key in seen:
                continue
            seen.add(key)
            for rec in self._quest_rows:
                if rec.remote_dir == remote:
                    out.append((rec, name))
                    break
        return out

    def _selected_quest_folder(self) -> Optional[QuestJsonFolder]:
        pairs = self._selected_quest_jsons()
        if pairs:
            return pairs[0][0]
        return None

    def _show_all_neons(self) -> None:
        self._fill_neon_table(self._neon_rows)
        n = len(self._neon_rows)
        self.neon_status.setText(
            f"{n} export(s)" + (f" in {self._neon_root}" if self._neon_root else "")
        )

    def match_neon_to_quest(self) -> None:
        folder = self._selected_quest_folder()
        if folder is None:
            QMessageBox.information(
                self,
                "Match Quest",
                "Select a Quest JSON row first (left table), then Match selected Quest.",
            )
            return
        if not folder.json_names and not folder.all_json_names:
            QMessageBox.information(self, "Match Quest", f"No JSON in {folder.folder}.")
            return
        if not self._neon_rows:
            QMessageBox.information(
                self, "Match Quest", "List Neon exports first (right table)."
            )
            return
        serial = self.assigned_quest()
        if not self._adb or not serial:
            QMessageBox.information(self, "Match Quest", "Assign a Quest device first.")
            return
        if self._match_worker and self._match_worker.isRunning():
            return
        self.btn_neon_match.setEnabled(False)
        self.neon_status.setText(
            f"Reading Head/Hand/Eye timelines in {folder.folder}…"
        )
        self._match_worker = QuestMatchWorker(self._adb, serial, folder, self)
        self._match_worker.done.connect(self._on_quest_window)
        self._match_worker.start()

    def _on_quest_window(self, windows, err) -> None:
        self.btn_neon_match.setEnabled(True)
        if err:
            self.neon_status.setText(f"Quest timeline failed: {err}")
            QMessageBox.warning(self, "Match Quest", str(err))
            return
        windows = dict(windows or {})
        assigned = assign_neons_to_interactions(self._neon_rows, windows)
        quest_folder = self._selected_quest_folder()
        quest_bout = quest_folder.speed if quest_folder is not None else ""
        self._neon_to_interaction = {m.rec.remote_dir: inter for inter, m in assigned}
        if quest_bout:
            for _inter, m in assigned:
                self._neon_bout[m.rec.remote_dir] = quest_bout
        if not assigned:
            self._fill_neon_table(self._neon_rows)
            self.neon_status.setText(
                f"No Neon overlap with {len(windows)} Quest interaction window(s). "
                "Showing all — set bout + interaction on the 3 takes, then Pull."
            )
            return
        notes = {}
        rows = []
        bits = []
        for inter, m in assigned:
            rows.append(m.rec)
            notes[m.rec.remote_dir] = (
                f"{inter}  overlap {fmt_duration(m.overlap_s)}"
                + (" contained" if m.contained else "")
            )
            bits.append(f"{inter}={fmt_duration(m.overlap_s)}")
        self._fill_neon_table(rows, notes=notes, select_all=True)
        self.neon_status.setText(
            f"Matched {len(assigned)} Neon(s) for this subsub (expect 3): "
            + ", ".join(bits)
        )


def main() -> int:
    app = QApplication(sys.argv)
    win = PullGui()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
