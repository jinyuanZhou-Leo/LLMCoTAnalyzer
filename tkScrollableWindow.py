import ttkbootstrap as ttk
import tkinter as tk
from tkinter import messagebox
from ttkbootstrap.constants import *


class tkScrollableWindow(ttk.Window):
    def __init__(
        self,
        title="ttkbootstrap",
        themename="darkly",
        iconphoto="",
        size=None,
        position=None,
        minsize=None,
        maxsize=None,
        resizable=None,
        hdpi=True,
        scaling=None,
        transient=None,
        overrideredirect=False,
        alpha=1,
        **kwargs,
    ):
        super().__init__(
            title,
            themename,
            iconphoto,
            size,
            position,
            minsize,
            maxsize,
            resizable,
            hdpi,
            scaling,
            transient,
            overrideredirect,
            alpha,
            **kwargs,
        )
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.pack(side="left", fill="both", expand=True, ipadx=10, ipady=10, padx=10, pady=10)
        self.scrollbar.pack(side="right", fill="y")

        style = ttk.Style()
        style.configure(".", font="Helvetica")
        self.protocol("WM_DELETE_WINDOW", self.on_close)  # 新增：绑定窗口关闭事件

    def _on_mousewheel(self, event):
        if self.canvas.winfo_height() < self.canvas.bbox("all")[3]:
            direction = -1 if event.delta > 0 else 1
            self.canvas.yview_scroll(direction, "units")

    def _update_scrollregion(self):
        self.scrollable_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_close(self):
        # 窗口关闭时显式销毁界面
        self.destroy()


if __name__ == "__main__":
    root = tkScrollableWindow(themename="darkly")
    root.mainloop()
