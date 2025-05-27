import ttkbootstrap as ttk
import tkinter as tk
from tkinter import messagebox
from ttkbootstrap.tableview import TableRow, Tableview
from typing import Callable, Any
from loguru import logger


class ModernTableView(Tableview):
    def __init__(
        self,
        master=None,
        bootstyle=...,
        coldata=...,
        rowdata=...,
        paginated=False,
        searchable=False,
        yscrollbar=False,
        autofit=False,
        autoalign=True,
        stripecolor=None,
        pagesize=10,
        height=10,
        delimiter=",",
    ):
        super().__init__(
            master,
            bootstyle,
            coldata,
            rowdata,
            paginated,
            searchable,
            yscrollbar,
            autofit,
            autoalign,
            stripecolor,
            pagesize,
            height,
            delimiter,
        )

    def insert_rows(self, index, rowdata, validator: Callable[..., bool] = lambda x: True) -> None:
        if validator(rowdata):
            logger.info(f"Inserted row data: {rowdata}")
            return super().insert_rows(index, rowdata)
        else:
            logger.error(f"Invalid row data: {rowdata}")

    def insert_row(self, index: str = "end", values=[], validator: Callable[..., bool] = lambda x: True) -> TableRow:
        if validator(values):
            return super().insert_row(index, values)
        else:
            logger.error(f"Invalid row data: {values}")

    def remove_selected_rows(self):
        selected_iids = self.view.selection()
        if selected_iids:
            # 删除选中的行
            self.delete_rows(iids=selected_iids)


if __name__ == "__main__":
    logger.warning("This is a demo script, this module is not meant to be run directly.")
