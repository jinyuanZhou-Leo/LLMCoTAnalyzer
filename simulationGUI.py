import ttkbootstrap as ttk
import tkinter as tk
import json
from loguru import logger
from ttkbootstrap.tableview import Tableview
from threading import Thread
from simulation import Simulation
from ttkbootstrap.constants import *


class SimulationGUI(ttk.Window):
    def __init__(
        self,
        model_list: list[str] = None,
        question_list: list[str] = None,
    ):
        super().__init__(themename="darkly")
        self.model_list = model_list or []
        self.question_list = question_list or []
        self.title("LLM CoT Simulation")
        self.geometry("800x800")

        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.pack(
            side="left", fill="both", expand=True, ipadx=10, ipady=10, padx=10, pady=10
        )
        self.scrollbar.pack(side="right", fill="y")

        style = ttk.Style()
        style.configure(".", font="Helvetica")
        self.simulation_cofig = {}
        self.create_widgets()
        logger.add(self.update_log, level="DEBUG", colorize=False)

    def create_widgets(self):
        self._create_model_config_widgets()
        self._create_question_config_widgets()
        self._create_other_config_widgets()
        self._create_log_widget()
        self.start_btn = ttk.Button(
            self.scrollable_frame,
            text="▶ Start Simulation",
            style="Start.TButton",
            command=self.start_simulation,
        )
        self.start_btn.pack(fill=tk.X, pady=10)
        self.after_idle(self._update_scrollregion)

    def _create_question_config_widgets(self):
        question_config_frame = ttk.LabelFrame(
            self.scrollable_frame, text="Question Configuration", padding=10
        )
        question_config_frame.pack(fill=tk.X, pady=10)

        cols = "Question"
        self.question_table = Tableview(
            master=question_config_frame,
            paginated=False,  # 不分页
            searchable=False,  # 不启用搜索
            bootstyle=PRIMARY,
            coldata=[{"text": cols, "stretch": True}],  # 列定义
            rowdata=[],  # 初始行数据为空
            autoalign=True,
            autofit=True,
        )
        self.question_table.pack(fill=tk.X, pady=5)

        add_question_frame = ttk.Frame(question_config_frame)
        add_question_frame.pack(fill=tk.X, pady=5)
        self.add_question_entry = ttk.Entry(add_question_frame, width=50)
        self.add_question_entry.pack(side=tk.LEFT, padx=5)

        btn_frame = ttk.Frame(question_config_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(
            btn_frame,
            text="+ Add",
            command=lambda: self._add_question(  # 使用lambda延迟执行
                self.add_question_entry.get(),
            ),
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            btn_frame, text="- Remove", command=self._remove_question, bootstyle="danger"
        ).pack(side=tk.LEFT, padx=2)

        # pre-load
        self._load_default_table_content(self.question_table, self.question_list)

    def _create_model_config_widgets(self):
        model_config_frame = ttk.LabelFrame(
            self.scrollable_frame, text="Model Configuration", padding=10
        )
        model_config_frame.pack(fill=tk.X, pady=10)

        cols = ("Model", "API URL", "API Key")
        self.model_table = Tableview(
            master=model_config_frame,
            paginated=False,  # 不分页
            searchable=False,  # 不启用搜索
            bootstyle=PRIMARY,
            coldata=[{"text": col, "stretch": True} for col in cols],  # 列定义
            rowdata=[],  # 初始行数据为空
            autoalign=True,
            autofit=True,
        )
        self.model_table.pack(fill=tk.X, pady=5)

        add_model_entries_frame = ttk.Frame(model_config_frame)
        add_model_entries_frame.pack(fill=tk.X, pady=5)
        self.add_model_entry_model_name = ttk.Entry(add_model_entries_frame, width=25)
        self.add_model_entry_api_url = ttk.Entry(add_model_entries_frame, width=25)
        self.add_model_entry_api_key = ttk.Entry(add_model_entries_frame, width=25)
        self.add_model_entry_model_name.pack(side=tk.LEFT, padx=5)
        self.add_model_entry_api_url.pack(side=tk.LEFT, padx=5)
        self.add_model_entry_api_key.pack(side=tk.LEFT, padx=5)

        btn_frame = ttk.Frame(model_config_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(
            btn_frame,
            text="+ Add",
            command=lambda: self._add_model(  # 使用lambda延迟执行
                self.add_model_entry_model_name.get(),
                self.add_model_entry_api_url.get(),
                self.add_model_entry_api_key.get(),
            ),
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="- Remove", command=self._remove_model, bootstyle="danger").pack(
            side=tk.LEFT, padx=2
        )

        # pre-load
        self._load_default_table_content(self.model_table, self.model_list)

    def _create_other_config_widgets(self):
        other_config_frame = ttk.LabelFrame(
            self.scrollable_frame, text="Other Configuration", padding=10
        )
        other_config_frame.pack(fill=tk.X, pady=10)

        ttk.Label(other_config_frame, text="Repetition (Per Question)").pack(anchor=tk.W)
        self.repetition_spin = ttk.Spinbox(other_config_frame, from_=2, to=20)
        self.repetition_spin.set(2)
        self.repetition_spin.pack(anchor=tk.W, pady=5)

        self.ask_when_unsure_var = tk.BooleanVar(value=False)
        ask_when_unsure_checkbutton = ttk.Checkbutton(
            other_config_frame,
            text="Ask me when unsure",
            variable=self.ask_when_unsure_var,
        )
        ask_when_unsure_checkbutton.pack(anchor=tk.W, pady=5)

    def start_simulation(self):
        raise NotImplementedError()

    def _load_default_table_content(self, table, content):
        for c in content:
            if isinstance(c, str):
                table.insert_row(values=(c,))
            elif isinstance(c, (tuple, list)):
                table.insert_row(values=tuple(c))
        table.load_table_data()
        logger.debug(f"Default content for {table} loaded successfully.")

    def _add_model(self, model_name, api_url, api_key):
        if all([model_name, api_url, api_key]):
            if any(row.values[0] == model_name for row in self.model_table.rows):
                logger.error(f"Model {model_name} already exists")
            self.model_table.insert_row(values=(model_name, api_url, api_key))
            self.model_table.load_table_data()  # 刷新表格
            self.add_model_entry_model_name.delete(0, tk.END)
            self.add_model_entry_api_url.delete(0, tk.END)
            self.add_model_entry_api_key.delete(0, tk.END)
            logger.info("Model added successfully.")
        else:
            logger.error("Fail to add model: [modelname], [api-url], or [api-key] cannot be empty.")
        self._update_scrollregion()

    def _add_question(self, question):
        if question:
            self.question_table.insert_row(values=(question,))
            self.question_table.load_table_data()  # 刷新表格
            self.add_question_entry.delete(0, tk.END)
            logger.info("Question added successfully.")
        else:
            logger.error("Fail to add question: [question_contents] cannot be empty.")
        self._update_scrollregion()

    def _remove_model(self):
        if selected_ids := self.model_table.view.selection():
            # 获取所有行数据
            all_rows = self.model_table.get_rows()
            # 通过ID反向查找行索引
            for item_id in reversed(selected_ids):
                row_index = self.model_table.view.index(item_id)
                if 0 <= row_index < len(all_rows):
                    self.model_table.delete_row(row_index)

            self.model_table.load_table_data()  # 单次刷新
            logger.info(f"Removed {len(selected_ids)} models")
        else:
            logger.warning("No model selected for removal")
        self._update_scrollregion()

    def _remove_question(self):
        if selected_ids := self.question_table.view.selection():
            all_rows = self.question_table.get_rows()
            # 通过ID反向查找行索引
            for item_id in reversed(selected_ids):
                row_index = self.question_table.view.index(item_id)
                if 0 <= row_index < len(all_rows):
                    self.question_table.delete_row(row_index)

            self.question_table.load_table_data()  # 单次刷新
            logger.info(f"Removed {len(selected_ids)} question")
        else:
            logger.warning("No question selected for removal")
        self._update_scrollregion()

    def update_log(self, message):
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)

    def _update_scrollregion(self):
        # 强制更新几何管理器，确保所有控件都已正确布局
        self.scrollable_frame.update_idletasks()
        # 更新滚动区域
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _create_log_widget(self):
        self.log_text = tk.Text(self.scrollable_frame, height=15)
        self.log_text.pack(fill=BOTH, expand=True)

    def save_config(self):
        models = [self.model_table.item(item)["values"] for item in self.model_table.get_children()]
        with open("config.json", "w") as f:
            json.dump(models, f)

    def _on_mousewheel(self, event):
        if self.canvas.winfo_height() < self.canvas.bbox("all")[3]:
            direction = -1 if event.delta > 0 else 1
            self.canvas.yview_scroll(direction, "units")


if __name__ == "__main__":
    model_list = [
        ["qwen3-0.6b", "https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"],
        ["qwen3-1.7b", "https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"],
        ["qwen3-4b", "https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"],
        ["qwen3-8b", "https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"],
    ]
    question_list = ["What is the integral of x^2?", "What is the derivative of (lnx)^2?"]
    app = SimulationGUI(model_list=model_list, question_list=question_list)
    app.mainloop()
