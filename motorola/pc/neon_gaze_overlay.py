#!/usr/bin/env python3
"""Live Neon gaze cursor on this PC monitor (grid calibration).

- Normal maximized window with toolbar (Calibrate / Reset / Close).
- During calibration: dark background + grid dots; after cal: transparent + one center dot.
- Red gaze cursor when mapping is active.
- Calibrate: look at each green dot, press Space to advance (no timer).
- H hide / show the top toolbar only (gaze canvas stays visible).

Prerequisites:
  cd motorola/pc && uv sync

Usage:
  uv run python neon_gaze_overlay.py
  uv run python neon_gaze_overlay.py --ip 192.168.1.42
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QKeySequence, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QShortcut,
    QVBoxLayout,
    QWidget,
)

from neon_probe import discover_ip

@dataclass
class GazeSample:
    x: float
    y: float
    worn: bool
    ts: float


@dataclass
class OverlayState:
    cols: int
    rows: int
    margin_frac: float = 0.12
    cal_min_samples: int = 3
    cursor_radius: int = 18
    trail_len: int = 25

    H_map: Optional[np.ndarray] = None
    trail: list[tuple[float, float]] = field(default_factory=list)

    cal_mode: bool = False
    cal_idx: int = 0
    cal_gaze_samples: list[tuple[float, float]] = field(default_factory=list)
    cal_gaze_observed: list[tuple[float, float]] = field(default_factory=list)
    status: str = "Calibrate: look at green dot, press Space"

    def targets(self, w: int, h: int) -> list[tuple[float, float]]:
        return grid_points(self.cols, self.rows, w, h, self.margin_frac)


def get_scene_calibration(device) -> Optional[tuple[np.ndarray, np.ndarray]]:
    fn = getattr(device, "get_calibration", None)
    if not callable(fn):
        return None
    try:
        cal = fn()
    except Exception as e:
        print(f"[cal] get_calibration() failed: {e}", file=sys.stderr)
        return None
    matrix = getattr(cal, "scene_camera_matrix", None)
    if matrix is None:
        matrix = getattr(cal, "camera_matrix", None)
    dist = getattr(cal, "scene_distortion_coefficients", None)
    if dist is None:
        dist = getattr(cal, "dist_coefs", None)
    if dist is None:
        dist = getattr(cal, "distortion_coefficients", None)
    if matrix is None or dist is None:
        return None
    return np.array(matrix, dtype=np.float64).reshape(3, 3), np.array(dist, dtype=np.float64).reshape(-1)


def undistort_point(x: float, y: float, K: np.ndarray, D: np.ndarray) -> tuple[float, float]:
    pts = np.array([[[x, y]]], dtype=np.float32)
    out = cv2.undistortPoints(pts, K, D, P=K)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


class NeonGazeReader(threading.Thread):
    def __init__(self, ip: str, port: int) -> None:
        super().__init__(daemon=True)
        self.ip = ip
        self.port = port
        self._stop = threading.Event()
        self.latest: Optional[GazeSample] = None
        self.calibration: Optional[tuple[np.ndarray, np.ndarray]] = None
        self.error: Optional[str] = None

    def run(self) -> None:
        try:
            from pupil_labs.realtime_api.simple import Device
        except ImportError:
            self.error = "pupil-labs-realtime-api not installed — run: cd motorola/pc && uv sync"
            return

        device = Device(address=self.ip, port=self.port)
        self.calibration = get_scene_calibration(device)
        if self.calibration is None:
            print("[neon] No scene calibration — undistort disabled.", file=sys.stderr)

        try:
            while not self._stop.is_set():
                receive = device.receive_gaze_datum
                try:
                    gaze = receive(timeout_seconds=0.2)
                except TypeError:
                    gaze = receive()
                if gaze is None:
                    continue
                self.latest = GazeSample(
                    x=float(gaze.x),
                    y=float(gaze.y),
                    worn=bool(gaze.worn),
                    ts=float(gaze.timestamp_unix_seconds),
                )
        except Exception as e:
            self.error = str(e)
        finally:
            device.close()

    def stop(self) -> None:
        self._stop.set()


def grid_points(cols: int, rows: int, w: int, h: int, margin_frac: float = 0.12) -> list[tuple[float, float]]:
    mx = w * margin_frac
    my = h * margin_frac
    sx = w - 2 * mx
    sy = h - 2 * my
    pts: list[tuple[float, float]] = []
    for r in range(rows):
        for c in range(cols):
            x = w / 2 if cols == 1 else mx + sx * c / (cols - 1)
            y = h / 2 if rows == 1 else my + sy * r / (rows - 1)
            pts.append((x, y))
    return pts


def gaze_to_monitor(
    gx: float,
    gy: float,
    H: np.ndarray,
    K: Optional[np.ndarray],
    D: Optional[np.ndarray],
) -> tuple[float, float]:
    if K is not None and D is not None:
        gx, gy = undistort_point(gx, gy, K, D)
    pt = np.array([[[gx, gy]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, H)
    return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])


def fit_gaze_homography(
    gaze_pts: list[tuple[float, float]],
    screen_pts: list[tuple[float, float]],
    K: Optional[np.ndarray],
    D: Optional[np.ndarray],
) -> tuple[Optional[np.ndarray], float]:
    if len(gaze_pts) < 4:
        return None, float("inf")
    src = []
    for gx, gy in gaze_pts:
        if K is not None and D is not None:
            gx, gy = undistort_point(gx, gy, K, D)
        src.append([gx, gy])
    src_arr = np.array(src, dtype=np.float32)
    dst_arr = np.array(screen_pts, dtype=np.float32)
    H, _ = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 5.0)
    if H is None:
        return None, float("inf")
    errs = []
    for (gx, gy), (tx, ty) in zip(src, screen_pts):
        mx, my = gaze_to_monitor(gx, gy, H, None, None)
        errs.append((mx - tx) ** 2 + (my - ty) ** 2)
    return H, float(np.sqrt(np.mean(errs)))


def _apply_windows_layered(hwnd: int) -> None:
    if sys.platform != "win32":
        return
    import ctypes

    GWL_EXSTYLE = -20
    WS_EX_LAYERED = 0x00080000
    user32 = ctypes.windll.user32
    style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED)


class GazeCanvas(QWidget):
    """Calibration: dark background + grid. After cal: transparent + center dot only."""

    BG = QColor(24, 24, 24)
    CENTER_DOT_RADIUS = 8

    def __init__(self, host: "GazeOverlayWindow") -> None:
        super().__init__(host)
        self._host = host
        self._overlay_mode = False
        self._mapped: Optional[tuple[float, float]] = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._apply_solid_background()

    def _apply_solid_background(self) -> None:
        self._overlay_mode = False
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), self.BG)
        self.setPalette(pal)

    def set_overlay_mode(self, enabled: bool) -> None:
        """After calibration: see-through except center target dot + gaze cursor."""
        self._overlay_mode = enabled
        if enabled:
            self.setAttribute(Qt.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WA_NoSystemBackground, True)
            self.setAutoFillBackground(False)
            self.setStyleSheet("background: transparent;")
        else:
            self.setStyleSheet("")
            self._apply_solid_background()
        self.update()

    def set_mapped_gaze(self, point: Optional[tuple[float, float]]) -> None:
        self._mapped = point

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._host.on_space_pressed()
            event.accept()
            return
        super().keyPressEvent(event)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        st = self._host.state
        w, h = self.width(), self.height()
        show_grid = st.cal_mode or st.H_map is None

        if self._overlay_mode and not st.cal_mode:
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.eraseRect(self.rect())
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        else:
            painter.fillRect(self.rect(), self.BG)

        if show_grid:
            for x, y in st.targets(w, h):
                painter.setPen(QPen(QColor(100, 100, 100), 1))
                painter.setBrush(QColor(240, 240, 240))
                painter.drawEllipse(QPoint(int(x), int(y)), 8, 8)

        if self._overlay_mode and not st.cal_mode and st.H_map is not None:
            cx, cy = w / 2.0, h / 2.0
            painter.setPen(QPen(QColor(100, 100, 100), 1))
            painter.setBrush(QColor(240, 240, 240))
            painter.drawEllipse(
                QPoint(int(cx), int(cy)), self.CENTER_DOT_RADIUS, self.CENTER_DOT_RADIUS
            )

        for i, (tx, ty) in enumerate(st.trail[:-1]):
            alpha = (i + 1) / max(len(st.trail), 1)
            r = max(2, int(st.cursor_radius * 0.35 * alpha))
            painter.setBrush(QColor(255, int(160 * alpha), int(240 * alpha)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(int(tx), int(ty)), r, r)

        if self._mapped is not None:
            mx, my = self._mapped
            painter.setPen(QPen(QColor(255, 60, 60), 2))
            painter.setBrush(QColor(255, 50, 50))
            painter.drawEllipse(
                QPoint(int(mx), int(my)), st.cursor_radius, st.cursor_radius
            )

        if st.cal_mode and st.cal_idx < len(st.targets(w, h)):
            tx, ty = st.targets(w, h)[st.cal_idx]
            painter.setPen(QPen(QColor(50, 220, 90), 3))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPoint(int(tx), int(ty)), 26, 26)
            painter.setBrush(QColor(50, 220, 90))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(int(tx), int(ty)), 8, 8)

        if not self._host._chrome.isVisible():
            painter.setPen(QColor(140, 140, 140))
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(8, 18, "H = show toolbar")
        elif st.cal_mode and st.cal_idx < len(st.targets(w, h)):
            n = len(st.cal_gaze_samples)
            need = st.cal_min_samples
            painter.setPen(QColor(160, 160, 160))
            painter.setFont(QFont("Segoe UI", 10))
            painter.drawText(
                8,
                18,
                f"Space = next dot  ({n}/{need} samples)" if n < need else "Space = next dot",
            )


class GazeOverlayWindow(QWidget):
    def __init__(
        self,
        reader: NeonGazeReader,
        state: OverlayState,
        *,
        start_maximized: bool,
    ) -> None:
        super().__init__()
        self.reader = reader
        self.state = state
        self.K: Optional[np.ndarray]
        self.D: Optional[np.ndarray]
        if reader.calibration:
            self.K, self.D = reader.calibration
        else:
            self.K, self.D = None, None

        self.setWindowTitle("Neon gaze overlay")
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.setMinimumSize(640, 480)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet("GazeOverlayWindow { background-color: #181818; }")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._chrome = QWidget()
        self._chrome.setObjectName("chrome")
        self._chrome.setStyleSheet(
            "#chrome { background-color: #2d2d2d; }"
            "QLabel { color: #e8e8e8; }"
            "QPushButton {"
            "  background: #444; color: #fff; border: 1px solid #666;"
            "  padding: 6px 16px; border-radius: 4px; min-width: 72px;"
            "}"
            "QPushButton:hover { background: #555; }"
            "QPushButton#closeBtn:hover { background: #c0392b; }"
        )
        bar = QHBoxLayout(self._chrome)
        bar.setContentsMargins(12, 8, 12, 8)

        self._status = QLabel()
        self._status.setFont(QFont("Segoe UI", 10))
        bar.addWidget(self._status, 1)

        cal_btn = QPushButton("Calibrate")
        cal_btn.clicked.connect(self.start_calibration)
        bar.addWidget(cal_btn)

        self._next_btn = QPushButton("Next dot (Space)")
        self._next_btn.clicked.connect(self.on_space_pressed)
        self._next_btn.setEnabled(False)
        bar.addWidget(self._next_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset_mapping)
        bar.addWidget(reset_btn)

        close_btn = QPushButton("Close")
        close_btn.setObjectName("closeBtn")
        close_btn.clicked.connect(self.close)
        bar.addWidget(close_btn)

        self._canvas = GazeCanvas(self)
        root.addWidget(self._chrome)
        root.addWidget(self._canvas, 1)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

        hide_shortcut = QShortcut(QKeySequence(Qt.Key_H), self)
        hide_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        hide_shortcut.activated.connect(self.toggle_toolbar)

        self._space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self._space_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._space_shortcut.activated.connect(self.on_space_pressed)

        self._refresh_status()
        if start_maximized:
            self.showMaximized()
        else:
            self.resize(1280, 720)
        self.show()
        QTimer.singleShot(0, self._bring_to_front)

    def _bring_to_front(self) -> None:
        self.raise_()
        self.activateWindow()

    def _refresh_status(self) -> None:
        st = self.state
        gaze = self.reader.latest
        worn = gaze.worn if gaze else False
        bar_vis = "toolbar on" if self._chrome.isVisible() else "toolbar off (H)"
        extra = ""
        if st.cal_mode:
            extra = f"  samples: {len(st.cal_gaze_samples)}/{st.cal_min_samples}"
        self._status.setText(
            f"{st.status}  |  map: {'OK' if st.H_map is not None else '—'}  "
            f"worn: {worn}  grid: {st.cols}×{st.rows}{extra}  |  H: {bar_vis}"
        )

    def toggle_toolbar(self) -> None:
        self._chrome.setVisible(not self._chrome.isVisible())
        if self._chrome.isVisible():
            self._refresh_status()
        self._canvas.update()

    def _enter_post_cal_overlay(self) -> None:
        self._canvas.set_overlay_mode(True)
        self.setStyleSheet("GazeOverlayWindow { background-color: transparent; }")
        _apply_windows_layered(int(self.winId()))

    def start_calibration(self) -> None:
        st = self.state
        self._canvas.set_overlay_mode(False)
        self.setStyleSheet("GazeOverlayWindow { background-color: #181818; }")
        st.cal_mode = True
        st.cal_idx = 0
        st.cal_gaze_samples = []
        st.cal_gaze_observed = []
        st.trail = []
        self._next_btn.setEnabled(True)
        n = len(st.targets(max(self._canvas.width(), 1), max(self._canvas.height(), 1)))
        if self.reader.latest is None:
            st.status = f"Dot 1/{n} — waiting for gaze… look at green, press Space"
        else:
            st.status = f"Dot 1/{n} — look at green, press Space"
        self._canvas.setFocus(Qt.OtherFocusReason)
        self.activateWindow()
        self._refresh_status()
        self._canvas.update()

    def reset_mapping(self) -> None:
        st = self.state
        st.H_map = None
        st.trail = []
        st.cal_mode = False
        st.status = "Mapping cleared — click Calibrate"
        self._next_btn.setEnabled(False)
        self._canvas.set_overlay_mode(False)
        self.setStyleSheet("GazeOverlayWindow { background-color: #181818; }")
        self._refresh_status()
        self._canvas.update()

    def on_space_pressed(self) -> None:
        if not self.state.cal_mode:
            self.state.status = "Press Calibrate first (or C)"
            self._refresh_status()
            return
        self.advance_calibration_dot()

    def advance_calibration_dot(self) -> None:
        st = self.state
        w, h = self._canvas.width(), self._canvas.height()
        if w < 10 or h < 10:
            return
        targets = st.targets(w, h)
        if not st.cal_mode or st.cal_idx >= len(targets):
            return

        n = len(st.cal_gaze_samples)
        if n < st.cal_min_samples:
            st.status = (
                f"Need {st.cal_min_samples}+ gaze samples on this dot (have {n}) — keep looking"
            )
            self._refresh_status()
            self._canvas.update()
            return

        med = np.median(np.array(st.cal_gaze_samples), axis=0)
        st.cal_gaze_observed.append((float(med[0]), float(med[1])))
        st.cal_idx += 1
        st.cal_gaze_samples = []

        if st.cal_idx >= len(targets):
            H_new, rms = fit_gaze_homography(st.cal_gaze_observed, targets, self.K, self.D)
            st.cal_mode = False
            self._next_btn.setEnabled(False)
            if H_new is None:
                st.status = "Calibration failed — try again"
            else:
                st.H_map = H_new
                st.status = f"Calibration OK ({rms:.1f} px error)"
                self._enter_post_cal_overlay()
        else:
            st.status = f"Dot {st.cal_idx + 1}/{len(targets)} — look at green, press Space"

        self._refresh_status()
        self._canvas.update()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self.on_space_pressed()
            event.accept()
            return
        key = event.key()
        if key in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
        elif key in (Qt.Key_C,):
            self.start_calibration()
        elif key in (Qt.Key_R,):
            self.reset_mapping()
        elif key in (Qt.Key_H,):
            self.toggle_toolbar()
        else:
            super().keyPressEvent(event)

    def _tick(self) -> None:
        if self.reader.error:
            self.state.status = f"Stream error: {self.reader.error}"
            self._refresh_status()
            self._timer.stop()
            return

        st = self.state
        gaze = self.reader.latest
        w, h = self._canvas.width(), self._canvas.height()
        if w < 10 or h < 10:
            return

        mapped: Optional[tuple[float, float]] = None
        if st.H_map is not None and gaze is not None and gaze.worn:
            try:
                mx, my = gaze_to_monitor(gaze.x, gaze.y, st.H_map, self.K, self.D)
                mapped = (mx, my)
                st.trail.append(mapped)
                st.trail = st.trail[-max(1, st.trail_len) :]
            except Exception:
                mapped = None
                st.trail = []
        else:
            st.trail = []

        self._canvas.set_mapped_gaze(mapped)

        if st.cal_mode and gaze is not None:
            st.cal_gaze_samples.append((gaze.x, gaze.y))
            if len(st.cal_gaze_samples) > 500:
                st.cal_gaze_samples = st.cal_gaze_samples[-300:]

        self._refresh_status()
        self._canvas.update()

    def closeEvent(self, event) -> None:  # noqa: N802
        self.reader.stop()
        super().closeEvent(event)


def main() -> int:
    parser = argparse.ArgumentParser(description="Neon gaze overlay on PC monitor")
    parser.add_argument("--ip", help="Companion phone IP (default: mDNS discover)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--discover-seconds", type=float, default=10.0)
    parser.add_argument("--cal-cols", type=int, default=4)
    parser.add_argument("--cal-rows", type=int, default=3)
    parser.add_argument(
        "--cal-min-samples",
        type=int,
        default=3,
        help="Gaze samples required per dot before Space accepts it (default 3)",
    )
    parser.add_argument("--cursor-radius", type=int, default=18)
    parser.add_argument("--trail", type=int, default=25)
    parser.add_argument("--windowed", action="store_true", help="Resizable window (not maximized)")
    args = parser.parse_args()

    ip = args.ip or discover_ip(args.discover_seconds)
    if not ip:
        print(
            "No Companion found. Open Neon → Streaming for IP, then:\n"
            "  uv run python neon_gaze_overlay.py --ip <phone-ip>",
            file=sys.stderr,
        )
        return 1

    reader = NeonGazeReader(ip, args.port)
    reader.start()
    time.sleep(0.8)
    if reader.error:
        print(f"Error: {reader.error}", file=sys.stderr)
        return 1

    state = OverlayState(
        cols=max(1, min(16, args.cal_cols)),
        rows=max(1, min(16, args.cal_rows)),
        cal_min_samples=max(1, args.cal_min_samples),
        cursor_radius=max(4, min(48, args.cursor_radius)),
        trail_len=max(1, args.trail),
    )

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setApplicationName("Neon gaze overlay")

    win = GazeOverlayWindow(reader, state, start_maximized=not args.windowed)
    code = app.exec_()
    reader.stop()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
