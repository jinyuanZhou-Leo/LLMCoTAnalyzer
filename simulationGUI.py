import ttkbootstrap as ttk
import tkinter as tk
from tkinter import messagebox
import json
import sys
from loguru import logger
from threading import Thread
from simulation import *
from ttkbootstrap.constants import *
from tkScrollableWindow import tkScrollableWindow
from tkTableView import ModernTableView


logger.remove()
logger.add(sys.stdout, level="INFO", colorize=True)
logger.add("simulationGUI.log", level="TRACE", rotation="300KB")

class SimulationGUI(tkScrollableWindow):

    def __init__(
        self,
        model_list: list[dict] = None,
        question_list: list[str] = None,
        repetition: int = 2,
        system_prompt: str = "You are a helpful assistant.",
        ask_when_unsure: bool = False,
    ):
        super().__init__(themename="darkly", iconphoto="./assets/icon.png")
        self.system_prompt = system_prompt
        self.ask_when_unsure = ask_when_unsure
        self.repetition = repetition
        self.model_list = model_list or []
        self.model_list = [
            (
                model["name"],
                model["size"],
                model["api_url"],
                model["api_key"],
            )
            for model in self.model_list
        ]
        self.question_list = question_list or []

        self.title("LLM CoT Simulation")
        self.geometry("1100x800")
        self.simulation_cofig = {}
        self.simulation_thread = None
        self.create_widgets()
        logger.add(self.update_log, level="INFO", colorize=False, format="{level} - {message}")

    def create_widgets(self):
        self._create_model_config_widgets()
        self._create_question_config_widgets()
        self._create_other_config_widgets()
        self._create_log_widget()
        self._create_simulation_control_panel()
        self.after_idle(self._update_scrollregion)

    def _create_simulation_control_panel(self):
        self.start_btn = ttk.Button(
            self.scrollable_frame,
            text="▶ Start Simulation",
            command=self.start_simulation,
        )
        self.start_btn.pack(fill=tk.X, pady=10)
        # self.terminate_btn = ttk.Button(
        #     self.scrollable_frame,
        #     text="Terminate Simulation",
        #     command=self.terminate_simulation,
        # )
        # self.terminate_btn.pack(fill=tk.X, pady=10)

    def _create_question_config_widgets(self):
        question_config_frame = ttk.LabelFrame(self.scrollable_frame, text="Question Configuration", padding=10)
        question_config_frame.pack(fill=tk.X, pady=10)

        self.question_table = ModernTableView(
            master=question_config_frame,
            bootstyle=PRIMARY,
            coldata=[{"text": "Question", "stretch": True}],  # 列定义
            rowdata=self.question_list,
            autofit=True,
        )
        self.question_table.pack(fill=tk.X, pady=5)

        add_question_frame = ttk.Frame(question_config_frame)
        add_question_frame.pack(fill=tk.X, pady=5)
        self.add_question_entry = ttk.Entry(add_question_frame, width=50)
        self.add_question_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        btn_frame = ttk.Frame(question_config_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(
            btn_frame,
            text="+ Add",
            command=lambda: self.question_table.insert_row(  # 使用lambda延迟执行
                values=(self.add_question_entry.get(),),
                validator=lambda x: x != "",
            ),
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            btn_frame, text="- Remove", command=self.question_table.remove_selected_rows, bootstyle="danger"
        ).pack(side=tk.LEFT, padx=2)

    def _create_model_config_widgets(self):
        model_config_frame = ttk.LabelFrame(self.scrollable_frame, text="Model Configuration", padding=10)
        model_config_frame.pack(fill=tk.X, pady=10)

        cols = (
            "Model",
            "Size (Billion Params)",
            "API URL",
            "API Key",
        )
        self.model_table = ModernTableView(
            master=model_config_frame,
            bootstyle=PRIMARY,
            coldata=[{"text": col, "stretch": True} for col in cols],  # 列定义
            rowdata=self.model_list,  # 初始行数据为空
            autofit=True,
        )
        self.model_table.pack(fill=tk.X, pady=5)

        add_model_entries_frame = ttk.Frame(model_config_frame)
        add_model_entries_frame.pack(fill=tk.X, pady=5)
        self.add_model_entry_model_name = ttk.Entry(add_model_entries_frame, width=25)
        self.add_model_entry_model_size = ttk.Entry(add_model_entries_frame, width=25)
        self.add_model_entry_api_url = ttk.Entry(add_model_entries_frame, width=25)
        self.add_model_entry_api_key = ttk.Entry(add_model_entries_frame, width=25)
        self.add_model_entry_model_name.pack(side=tk.LEFT, padx=5, expand=True)
        self.add_model_entry_model_size.pack(side=tk.LEFT, padx=5, expand=True)
        self.add_model_entry_api_url.pack(side=tk.LEFT, padx=5, expand=True)
        self.add_model_entry_api_key.pack(side=tk.LEFT, padx=5, expand=True)

        btn_frame = ttk.Frame(model_config_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(
            btn_frame,
            text="+ Add",
            command=lambda: self.model_table.insert_row(
                values=(
                    self.add_model_entry_model_name.get(),
                    self.add_model_entry_model_size.get(),
                    self.add_model_entry_api_url.get(),
                    self.add_model_entry_api_key.get(),
                ),
                validator=lambda x: all(x),
            ),
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="- Remove", command=self.model_table.remove_selected_rows, bootstyle="danger").pack(
            side=tk.LEFT, padx=2
        )

    def _create_other_config_widgets(self):
        other_config_frame = ttk.LabelFrame(self.scrollable_frame, text="Other Configuration", padding=10)
        other_config_frame.pack(fill=tk.X, pady=10)

        ttk.Label(other_config_frame, text="Repetition (Per Question)").pack(anchor=tk.W)
        self.repetition_spin = ttk.Spinbox(other_config_frame, from_=2, to=20)
        self.repetition_spin.set(self.repetition)
        self.repetition_spin.pack(anchor=tk.W, pady=5)

        ttk.Label(other_config_frame, text="System Prompt:").pack(anchor=tk.W)
        self.system_prompt_entry = ttk.Entry(other_config_frame, width=100)
        self.system_prompt_entry.pack(anchor=tk.W, pady=5)
        self.system_prompt_entry.insert(0, self.system_prompt)

        self.ask_when_unsure_var = tk.BooleanVar(value=self.ask_when_unsure)
        ask_when_unsure_checkbutton = ttk.Checkbutton(
            other_config_frame,
            text="Ask me when unsure",
            variable=self.ask_when_unsure_var,
        )
        ask_when_unsure_checkbutton.pack(anchor=tk.W, pady=10)

    def __get_simulation_config(self):
        model_list: list[dict] = []
        for row in self.model_table.get_rows():
            model_list.append(
                {
                    "name": row.values[0],
                    "api_url": row.values[2],
                    "api_key": row.values[3],
                    "size": row.values[1],
                }
            )

        return {
            "model_list": model_list,  # 直接遍历rows属性
            "repetition": int(self.repetition_spin.get()),
            "question_list": [row.values[0] for row in self.question_table.get_rows()],
            "system_prompt": self.system_prompt_entry.get(),
            "embedding_method": "SupervisedClassification",
            "ask_when_unsure": self.ask_when_unsure_var.get(),
        }

    def start_simulation(self):
        self.start_btn.config(state=tk.DISABLED)
        # 修改：使用守护线程并保存引用
        self.simulation_thread = Thread(target=self._run, daemon=True)
        self.simulation_thread.start()

    # TODO:
    def terminate_simulation(self):
        if self.simulation_thread and self.simulation_thread.is_alive():
            if messagebox.askyesno("Confirm", "Are you sure you want to terminate the simulation?"):
                self.simulation.terminate_simulation()
        else:
            messagebox.showwarning("Warning", "No simulation is running.")

    def _run(self):
        try:
            self.simulation = Simulation(**self.__get_simulation_config())
            self.simulation.start_simulation()
        except ForceTerminateException:
            messagebox.showinfo("Info", "Simulation terminated.")
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            messagebox.showerror("Error", f"Simulation failed: {e}")
        finally:
            self.start_btn.config(state=tk.NORMAL)

    def update_log(self, message):
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)

    def _create_log_widget(self):
        self.log_text = tk.Text(self.scrollable_frame, height=15)
        self.log_text.pack(fill=BOTH, expand=True)


if __name__ == "__main__":
    with open("simulation_config.json", "r", encoding="utf-8") as f:
        config: dict = json.load(f)

    app = SimulationGUI(
        model_list=config["model_list"],
        question_list=[(question,) for question in config["question_list"]],
        system_prompt=config["system_prompt"],
        ask_when_unsure=config["ask_when_unsure"],
        repetition=int(config["repetition"]),
    )
    app.mainloop()
