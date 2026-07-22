from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tkinter as tk
from tkinter import filedialog, ttk


@dataclass
class AppState:
    image_path: str | None = None
    scene_path: str | None = None
    proc: subprocess.Popen[str] | None = None
    preview_ref: object | None = None


class ImageApp(tk.Tk):
    def __init__(self, *, init_robot: str = "franka_fr3", init_cmd: str = "把红色方块移到左边", init_image: str | None = None, init_scene: str | None = None, init_no_viewer: bool = False, autorun: bool = False) -> None:
        super().__init__()
        self.state = AppState(image_path=init_image, scene_path=init_scene)

        self.title("VGArm 图片应用（MVP）")
        self.geometry("980x620")

        self._robot_var = tk.StringVar(value=init_robot)
        self._cmd_var = tk.StringVar(value=init_cmd)
        self._no_viewer_var = tk.BooleanVar(value=init_no_viewer)
        self._status_var = tk.StringVar(value="就绪")
        self._image_var = tk.StringVar(value=init_image or "未选择图片")

        self._build_ui()
        self._refresh_robot_choices()
        if init_image:
            self._render_preview(init_image)
        if autorun and (init_image or init_scene):
            self.after(300, self._run_sim)

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="选择图片", command=self._choose_image).pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self._image_var, width=90).pack(side=tk.LEFT, padx=10)

        mid = ttk.Frame(self, padding=10)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = ttk.Frame(mid)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(left, width=640, height=480, bg="#202020", highlightthickness=1, highlightbackground="#404040")
        self._canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        right = ttk.Frame(mid, width=300)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(right, text="机器人").pack(anchor=tk.W, pady=(0, 4))
        self._robot_box = ttk.Combobox(right, textvariable=self._robot_var, state="readonly")
        self._robot_box.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(right, text="指令").pack(anchor=tk.W, pady=(0, 4))
        ttk.Entry(right, textvariable=self._cmd_var).pack(fill=tk.X, pady=(0, 12))

        ttk.Checkbutton(right, text="无窗口运行（--no-viewer）", variable=self._no_viewer_var).pack(anchor=tk.W, pady=(0, 12))

        ttk.Button(right, text="运行仿真", command=self._run_sim).pack(fill=tk.X, pady=(0, 8))
        ttk.Button(right, text="停止运行", command=self._stop_sim).pack(fill=tk.X)

        bottom = ttk.Frame(self, padding=10)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(bottom, textvariable=self._status_var).pack(side=tk.LEFT)

    def _refresh_robot_choices(self) -> None:
        try:
            from vgarm.mjc import available_robots

            robots = sorted(available_robots().keys())
        except Exception:
            robots = ["franka_fr3"]
        self._robot_box["values"] = robots
        if self._robot_var.get() not in robots and robots:
            self._robot_var.set(robots[0])

    def _choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择环境图片",
            filetypes=[
                ("Image", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.state.image_path = path
        self._image_var.set(path)
        self._status_var.set("已选择图片")
        self._render_preview(path)

    def _render_preview(self, path: str) -> None:
        self._canvas.delete("all")
        try:
            from PIL import Image, ImageTk

            img = Image.open(path)
            canvas_w = max(int(self._canvas.winfo_width()), 640)
            canvas_h = max(int(self._canvas.winfo_height()), 480)
            img.thumbnail((canvas_w, canvas_h))
            photo = ImageTk.PhotoImage(img)
            self.state.preview_ref = photo
            self._canvas.create_image(canvas_w // 2, canvas_h // 2, image=photo, anchor=tk.CENTER)
            return
        except Exception:
            self.state.preview_ref = None
            self._canvas.create_text(
                320,
                240,
                text="无法预览图片（建议安装 Pillow：pip install pillow）",
                fill="#d0d0d0",
                font=("Segoe UI", 12),
                anchor=tk.CENTER,
            )

    def _run_sim(self) -> None:
        if self.state.proc is not None and self.state.proc.poll() is None:
            self._status_var.set("已有仿真在运行，请先停止")
            return
        if not (self.state.image_path or self.state.scene_path):
            self._status_var.set("请选择图片或提供场景JSON")
            return

        robot = self._robot_var.get()
        cmd = self._cmd_var.get().strip()
        if not cmd:
            self._status_var.set("请输入指令")
            return

        args = [sys.executable, "-m", "vgarm.cli", "--robot", robot, "--cmd", cmd]
        if self.state.scene_path:
            args += ["--scene", str(Path(self.state.scene_path).resolve())]
        elif self.state.image_path:
            args += ["--image", self.state.image_path]
        if self._no_viewer_var.get():
            args.append("--no-viewer")

        try:
            self.state.proc = subprocess.Popen(args, cwd=str(Path(__file__).resolve().parents[1]), text=True)
            self._status_var.set("仿真已启动")
            self.after(500, self._poll_proc)
        except Exception as e:
            self._status_var.set(f"启动失败：{type(e).__name__}")

    def _poll_proc(self) -> None:
        p = self.state.proc
        if p is None:
            return
        code = p.poll()
        if code is None:
            self.after(500, self._poll_proc)
            return
        self._status_var.set(f"仿真已结束（exit={code}）")
        self.state.proc = None

    def _stop_sim(self) -> None:
        p = self.state.proc
        if p is None or p.poll() is not None:
            self._status_var.set("当前无运行中的仿真")
            self.state.proc = None
            return
        try:
            p.terminate()
            self._status_var.set("已发送终止信号")
        except Exception as e:
            self._status_var.set(f"停止失败：{type(e).__name__}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vgarm-image-app")
    ap.add_argument("--robot", default="franka_fr3")
    ap.add_argument("--cmd", default="把红色方块移到左边")
    ap.add_argument("--image", default=None)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--autorun", action="store_true")
    args = ap.parse_args(argv)

    app = ImageApp(
        init_robot=args.robot,
        init_cmd=args.cmd,
        init_image=args.image,
        init_scene=args.scene,
        init_no_viewer=args.no_viewer,
        autorun=args.autorun,
    )
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
