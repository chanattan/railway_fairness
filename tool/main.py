# main.py
import tkinter as tk
from tkinter import ttk
from map_tab import MapTab
import json
from tkinter import filedialog

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Railway Network - Tabbed Interface")
        self.geometry("1200x800")

        self.tab_control = ttk.Notebook(self)
        self.tab_control.pack(fill="both", expand=True)

        self.menu = tk.Menu(self)
        file_menu = tk.Menu(self.menu, tearoff=0)
        file_menu.add_command(label="New Map Tab", command=self.add_tab)
        file_menu.add_command(label="Exit", command=self.quit)
        file_menu.add_command(label="Save Config", command=self.save_config)
        file_menu.add_command(label="Load Config", command=self.load_config)
        self.menu.add_cascade(label="File", menu=file_menu)
        self.config(menu=self.menu)

        self.tab_control.bind("<<NotebookTabChanged>>", self.on_tab_change)

        self.tab_count = 0
        self.TAB_ROWS = 8
        self.history_data = []
        self.tabs = []
        self.add_tab()

    def add_tab(self):
        self.tab_count += 1
        new_tab = MapTab(self.tab_control, self, tab_name=f"Map {self.tab_count}")
        self.tab_control.add(new_tab.frame, text=new_tab.tab_name)
        # copy content
        if len(self.tabs) > 0:
            current_index = self.tab_control.index(self.tab_control.select())
            tab = self.tabs[current_index]
            new_tab.load_map_data(tab.get_map_data())
            new_tab.city_ids = set(tab.city_ids)
        self.tabs.append(new_tab)
        self.tab_control.select(new_tab.frame)
    
    def on_tab_change(self, _):
        current_index = self.tab_control.index(self.tab_control.select())
        self.tabs[current_index].refresh_history_table()
        self.tabs[current_index].display_formula(None)

    def save_config(self):
        data = {
            "tabs": [],
            "history_data": self.history_data
        }
        for tab in self.tabs:
            tab_data = {
                "tab_name": tab.tab_name,
                "map_data": tab.get_map_data(),
                "metric": tab.metric.get(),
                "p_value": tab.p_value.get(),
                "result": tab.result.get()
            }
            data["tabs"].append(tab_data)

        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if file_path:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)

    def load_config(self):
        file_path = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not file_path:
            return

        with open(file_path, "r") as f:
            data = json.load(f)

        for tab in getattr(self, 'tabs', []):
            self.tab_control.forget(tab.frame)
        self.tabs = []
        self.tab_count = 0

        self.history_data = data.get("history_data", [])

        for tab_info in data["tabs"]:
            self.tab_count += 1
            new_tab = MapTab(self.tab_control, self, tab_name=tab_info["tab_name"])
            new_tab.load_map_data(tab_info["map_data"])
            new_tab.metric.set(tab_info["metric"])
            new_tab.p_value.delete(0, tk.END)
            new_tab.p_value.insert(0, tab_info["p_value"])
            new_tab.result.config(state="normal")
            new_tab.result.delete(0, tk.END)
            new_tab.result.insert(0, tab_info["result"])
            new_tab.result.config(state="disabled")

            self.tab_control.add(new_tab.frame, text=new_tab.tab_name)
            self.tabs.append(new_tab)
            for c in new_tab.cities.values():
                new_tab.city_ids.add(c.id)

        self.tab_control.select(0)

if __name__ == "__main__":
    app = App()
    app.mainloop()