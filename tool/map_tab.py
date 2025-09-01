# map_tab.py
import tkinter as tk
from tkinter import ttk, filedialog, Label, Entry
import itertools
from city import City
from line import Line
import json
import platform
import metrics
import utils
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
})

class MapTab:
    def __init__(self, parent, app, tab_name="New Map"):
        self.tab_name = tab_name
        self.app = app
        self.frame = tk.Frame(parent)
        self.canvas = tk.Canvas(self.frame, background="white")
        self.canvas.pack(fill="both", expand=True, side="left")

        def validate_negative_number(P):
            return P == "" or P == "-" or P.lstrip("-").isdigit()
        self.vcmd = (parent.register(validate_negative_number), "%P")

        self.options = tk.Frame(self.frame, width=200)
        self.options.pack(side="right", fill="y", expand=False)

        # Map state
        self.cities = {}
        self.transitions = {}
        self.lines = []
        self.demand_matrix = []
        self.distance_matrix = []
        self.curr_matrix = tk.StringVar()
        self.curr_matrix.set("Demand")
        self.vars_matrix = [] # Save variables for matrix
        self.selected_city = None # to create railway
        self.dragged_city = None # to move cities
        self.moving_city = False # to move cities
        self.highlighted_city = None # to highlight shortest path
        self.highlighted_lines = [] # to highlight shortest path
        self.city_ids = set()

        self.setup_bindings()

        Line.id_iter = itertools.count(0)

        self.setup_ui(tab_name)
    
    def setup_ui(self, tab_name):
        title = tk.Label(self.options, text=tab_name, font=("Arial", 14, "bold"))
        title.pack(pady=5)

        tk.Label(self.options, text="Metric").pack()
        self.metric = ttk.Combobox(self.options, values=metrics.METRICS, state="readonly")
        self.metric.set("SW")
        self.init = True
        self.metric.bind("<<ComboboxSelected>>", self.display_formula)
        self.metric.pack()

        tk.Label(self.options, text="p-value").pack()
        self.p_value = tk.Entry(self.options)
        self.p_value.insert(0, "1")
        self.p_value.pack()

        self.compute_btn = tk.Button(self.options, text="Compute", command=self.compute)
        self.compute_btn.pack()

        tk.Label(self.options, text="Measured Value").pack()
        self.result = tk.Entry(self.options, state="disabled")
        self.result.pack()

        tk.Label(self.options, text="Map Actions").pack(pady=5)

        tk.Button(self.options, text="Save Map", command=self.save_map).pack(fill=tk.X)
        tk.Button(self.options, text="Load Map", command=self.load_map).pack(fill=tk.X)
        tk.Button(self.options, text="Clear Map", command=self.clear_map).pack(fill=tk.X)
        
        self.history_label = tk.Label(self.options, text="Table of values", font=("Arial", 12, "bold"))
        self.history_label.pack(pady=(10, 0))

        columns = ("tab", "metric", "pval", "value")
        self.history_table = ttk.Treeview(self.options, columns=columns, show="headings", height=self.app.TAB_ROWS)

        for col in columns:
            self.history_table.heading(col, text=col.capitalize())
            self.history_table.column(col, width=80, anchor=tk.CENTER)

        self.history_table.pack(fill=tk.X, pady=5)
        self.formula_frame = tk.Frame(self.options, height=130)
        self.formula_frame.pack(fill=tk.BOTH)
        self.formula_frame.pack_propagate(False)

        self.latex_label = None
        self.init = False
        self.display_formula(None)

        self.matrix_selector = ttk.Combobox(self.options, textvariable=self.curr_matrix, values=["Demand", "Distance"], state="readonly")
        self.matrix_selector.set("Demand")
        self.matrix_selector.bind("<<ComboboxSelected>>", self.on_matrix_type_changed)
        self.matrix_selector.pack()

        tk.Label(self.options, text="Matrix").pack()

        self.load_matrix()
    
    def on_matrix_type_changed(self, _):
        #print("curr matrix:", self.curr_matrix.get())
        self.change_matrix()
    
    def clear_map(self):
        self.clear_matrix_display()
        self.clear_matrices()
        self.canvas.delete("all")
        self.cities.clear()
        self.lines.clear()
        self.transitions.clear()
        self.city_ids = set()
        Line.id_iter = itertools.count()

    def save_map(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not file_path:
            return
        data = self.get_map_data()
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def load_map(self):
        # Based on a file
        file_path = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not file_path:
            return
        with open(file_path, "r") as f:
            data = json.load(f)
        self.clear_map()
        city_map = {}
        for city_data in data.get("cities", []):
            city = City(self, city_data["x"], city_data["y"], city_data["id"])
            name = city_data.get("name", None)
            if name is not None:
                city.name = name
            city.pop = city_data.get("pop", 0)
            city.draw(self.canvas)
            self.cities[city.id] = city
            city_map[city.id] = city
        #print("citymap", city_map)
        for line_data in data.get("lines", []):
            city_a = city_map[line_data["city_a"]]
            city_b = city_map[line_data["city_b"]]
            line = Line(city_a, city_b)
            line.time = line_data["time"]
            line.dist = line_data["dist"]
            line.wait = line_data["wait"]
            line.demand = line_data["demand"]
            line.draw(self.canvas)
            self.lines.append(line)
            self.transitions.setdefault(city_a.id, []).append((city_b.id, line))
            self.transitions.setdefault(city_b.id, []).append((city_a.id, line))

        self.demand_matrix = data.get("demand_matrix", [])
        self.distance_matrix = data.get("distance_matrix", [])
        self.load_matrix()
        Line.id_iter = itertools.count(start=len(self.lines))

    def setup_bindings(self):
        self.canvas.bind("<Button-1>", self.create_city)
        self.canvas.bind("<Button-2>", self.right_click_event)
        self.canvas.bind("<Control-Button-1>", self.ctrl_click_event)
        self.platform = platform.system()
        self.modifier = "Command" if self.platform == "Darwin" else "Alt"
        self.canvas.bind(f"<{self.modifier}-Button-1>", self.cmd_click_event)
        self.canvas.bind("<ButtonRelease-1>", self.release_event)
        self.canvas.bind("<Motion>", self.move_city)
    
    def ctrl_click_event(self, event):
        self.moving_city = True
        self.dragged_city = self.get_selected_city(event.x, event.y)
    
    def move_city(self, event):
        if self.moving_city and self.dragged_city:
            self.reset_highlight()
            self.reset_selection()
            x, y = event.x, event.y
            self.dragged_city.x = x
            self.dragged_city.y = y
            self.dragged_city.delete(self.canvas)
            self.dragged_city.draw(self.canvas)
            lines = self.transitions.get(self.dragged_city.id, [])
            for (_, l) in lines:
                l.delete(self.canvas)
                l.draw(self.canvas)
    
    def release_event(self, _):
        self.moving_city = False
        self.dragged_city = None

    # City/Railway handlers
    def get_selected_city(self, x, y):
        overlaps = self.canvas.find_overlapping(x, y, x, y)
        overlaps = list(filter(lambda x: "id" not in self.canvas.gettags(x), overlaps))
        for c in self.cities.values():
            if overlaps and c.fig == overlaps[-1]: # inverse z-order
                return c
        return None

    def contains_fig(self, x, y):
        overlaps = self.canvas.find_overlapping(x, y, x, y)
        return len(overlaps) > 0

    def get_selected_railway(self, x, y):
        overlaps = self.canvas.find_overlapping(x, y, x, y)
        for l in self.transitions.values():
            for (_, line) in l:
                if overlaps and line.fig == overlaps[-1]:
                    return line
        return None

    def reset_selection(self):
        if self.selected_city:
            self.canvas.itemconfig(self.selected_city.fig, fill="black")
            self.selected_city = None
            return

    def create_city(self, event):
        if len(self.demand_matrix) == 0 and self.canvas_matrix is None:
            self.load_matrix()
        self.reset_highlight()
        in_city = self.contains_fig(event.x, event.y)
        if in_city:
            self.create_railway(event)
        else:
            self.reset_selection()
            city = City(self, event.x, event.y)
            self.cities[city.id] = city
            city.draw(self.canvas)
            self.update_matrix(city.id)
            self.clear_matrix_display()
            self.load_matrix()
        #print("cities", list(map(lambda x: x, self.cities.keys())))

    def right_click_event(self, event):
        self.reset_selection()
        selection = self.get_selected_city(event.x, event.y)
        if not selection:
            railway = self.get_selected_railway(event.x, event.y)
            if railway:
                self.delete_railway(railway)
        else:
            self.delete_city(selection)

    def delete_city(self, selection):
        selection.discard(self.canvas)
        # remove edges
        for (neigh, line) in self.transitions.get(selection.id, []):
            if self.transitions.get(neigh):
                elm = next((p for p in self.transitions.get(neigh, []) if p[0] == selection.id), None)
                if elm:
                    self.transitions[neigh].remove(elm)
                line.delete(self.canvas)
        self.transitions.pop(selection.id, None)
        self.cities.pop(selection.id)
        #print("cities", list(map(lambda x: x, self.cities.keys())))
        self.update_matrix(selection.id, added=False)

    def create_railway(self, event):
        if not self.selected_city:
            self.selected_city = self.get_selected_city(event.x, event.y)
            if self.selected_city:
                self.canvas.itemconfig(self.selected_city.fig, fill="red")
            return
        new_city = self.get_selected_city(event.x, event.y)
        if not new_city or new_city == self.selected_city:
            self.canvas.itemconfig(self.selected_city.fig, fill="black")
            self.selected_city = None
            return

        if new_city.id in [x[0] for x in self.transitions.get(self.selected_city.id, [])]:
            self.canvas.itemconfig(self.selected_city.fig, fill="black")
            self.selected_city = None
            return

        railway = Line(self.selected_city, new_city)
        railway.draw(self.canvas)
        self.transitions.setdefault(self.selected_city.id, []).append((new_city.id, railway))
        self.transitions.setdefault(new_city.id, []).append((self.selected_city.id, railway))
        self.lines.append(railway)
        self.canvas.itemconfig(self.selected_city.fig, fill="black")
        #print("lines", self.transitions)
        self.selected_city = None
    
    def delete_railway(self, railway):
        railway.delete(self.canvas)
        city_a = railway.city_a.id
        city_b = railway.city_b.id
        elm1 = next((p for p in self.transitions[city_a] if p[0] == city_b), None)
        elm2 = next((p for p in self.transitions[city_b] if p[0] == city_a), None)
        if elm1:
            self.transitions[city_a].remove(elm1)
            self.lines.remove(elm1[1])
        if elm2:
            self.transitions[city_b].remove(elm2)
            self.lines.remove(elm2[1])
    
    def cmd_click_event(self, event):
        railway = self.get_selected_railway(event.x, event.y)
        if railway:
            self.reset_highlight()
            self.edit_railway(railway)
        else:
            city_selection = self.get_selected_city(event.x, event.y)
            if city_selection:
                self.highlight_city(city_selection)
            else:
                self.reset_highlight()
                self.reset_selection()
    
    def reset_highlight(self, total=True):
        if self.highlighted_city:
            if total:
                self.canvas.itemconfig(self.highlighted_city.fig, fill="black")
            for f in self.highlighted_lines:
                self.canvas.itemconfig(f, fill="black")
            self.highlighted_lines = []
            if total:
                self.highlighted_city = None

    def highlight_city(self, city):
        if self.highlighted_city is None: # First selection
            self.highlighted_city = city
            self.canvas.itemconfig(city.fig, fill="blue")
        else: # Second selection
            # Highlight shortest path, without time
            self.reset_highlight(False) # only clear previous lines
            shortest_path, _ = utils.get_shortest_path(self.highlighted_city, city, self.transitions, time=True)
            if shortest_path is None:
                self.reset_highlight()
                return
            for line in shortest_path:
                self.canvas.itemconfig(line.fig, fill="red")
                self.highlighted_lines.append(line.fig)

    def edit_railway(self, railway):
        popup = tk.Toplevel(self.frame)
        popup.title("Edit Railway")
        popup.geometry("200x260")
        select_all = lambda event : event.widget.select_range(0, tk.END)

        raillbl = Label(popup, text=f"({railway.city_a.id}) <---> ({railway.city_b.id})")
        raillbl.pack()

        timelbl = Label(popup, text="Time of travel")
        timelbl.pack()
        time = Entry(master=popup, fg="white", width=30, validate="key", validatecommand=self.vcmd)
        time.insert(0, railway.time)
        time.bind("<FocusIn>", select_all)
        time.pack()

        distlbl = Label(popup, text="Distance between cities")
        distlbl.pack()
        dist = Entry(master=popup, fg="white", width=30, validate="key", validatecommand=self.vcmd)
        dist.insert(0, railway.dist)
        dist.bind("<FocusIn>", select_all)
        dist.pack()

        waitlbl = Label(popup, text="Average waiting time")
        waitlbl.pack()
        wait = Entry(master=popup, fg="white", width=30, validate="key", validatecommand=self.vcmd)
        wait.insert(0, railway.wait)
        wait.bind("<FocusIn>", select_all)
        wait.pack()

        dmdlbl = Label(popup, text="Demand")
        dmdlbl.pack()
        dmd = Entry(master=popup, fg="white", width=30, validate="key", validatecommand=self.vcmd)
        dmd.insert(0, railway.demand)
        dmd.bind("<FocusIn>", select_all)
        dmd.pack()

        def apply():
            railway.time = int(time.get())
            railway.dist = int(dist.get())
            railway.wait = int(wait.get())
            railway.demand = int(dmd.get())
            railway.delete(self.canvas)
            railway.draw(self.canvas)
            self.update_demand(railway.city_a.id, railway.city_b.id, railway.demand, redraw=False)
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", apply)
        tk.Button(popup, text="OK", command=apply).pack()
        popup.bind("<Escape>", lambda _: apply())
        popup.attributes("-topmost", True)
        popup.lift()                     
        popup.focus_force()

    def compute(self):
        metric = self.metric.get()
        try:
            p = int(self.p_value.get())
        except ValueError:
            return
        
        def set_val(val):
            self.result.config(state="normal")
            self.result.delete(0, tk.END)
            self.result.insert(0, val)
            self.result.config(state="disabled")

        match metric:
            case "SW":
                val = metrics.SW(p, list(self.cities.values()), self.transitions, self.demand_matrix)
            case "CW1":
                val = metrics.CW1(p, list(self.cities.values()), self.transitions, self.demand_matrix, self.distance_matrix)
            case "CW1N":
                val = metrics.CW1N(p, list(self.cities.values()), self.transitions, self.demand_matrix, self.distance_matrix)
            case "CW2":
                val = metrics.CW2(p, list(self.cities.values()), self.transitions, self.demand_matrix, self.distance_matrix)
            case "CW3":
                val = metrics.CW3(p, list(self.cities.values()), self.transitions, self.demand_matrix, self.distance_matrix)
            case "CW4":
                val = metrics.CW4(p, list(self.cities.values()), self.transitions, self.demand_matrix)
            case "CW5":
                val = metrics.CW5(p, list(self.cities.values()), self.transitions, self.demand_matrix, self.distance_matrix)
            case "GWc":
                val = metrics.GWc(p, list(self.cities.values()), self.transitions, self.demand_matrix, self.distance_matrix)
            case "GW":
                val = metrics.GW(list(self.cities.values()), self.transitions, self.demand_matrix, self.distance_matrix)
            case "GW1":
                val = metrics.GW1(p, list(self.cities.values()), self.transitions, self.demand_matrix, self.distance_matrix, demands_on=True)
            case "GW2":
                val = metrics.GW2(list(self.cities.values()), self.transitions, self.demand_matrix, self.distance_matrix, demands_on=True)
        set_val(round(val, 2))

        self.app.history_data.append((self.tab_name, metric, self.p_value.get(), self.result.get()))
        if len(self.app.history_data) > self.app.TAB_ROWS:
            self.app.history_data.pop(0)

        self.refresh_history_table()

    def refresh_history_table(self):
        self.history_table.delete(*self.history_table.get_children())
        for row in self.app.history_data:
            self.history_table.insert("", tk.END, values=row)

    def get_map_data(self):
        import copy
        def unique_lines(lines):
            seen = set()
            return list(filter(lambda l: (min(l.city_a.id, l.city_b.id), max(l.city_a.id, l.city_b.id)) not in seen and not seen.add((min(l.city_a.id, l.city_b.id), max(l.city_a.id, l.city_b.id))), lines))
        return {
            "cities": [{"id": c.id, "x": c.x, "y": c.y} if not hasattr(c, 'name') or c.name is None else
                       {"id": c.id, "x": c.x, "y": c.y, "name": c.name}
                       for c in self.cities.values()],
            "lines": [{
                "city_a": l.city_a.id,
                "city_b": l.city_b.id,
                "time": l.time,
                "dist": l.dist,
                "wait": l.wait,
                "demand": l.demand
            } for l in unique_lines(self.lines)],
            "demand_matrix": copy.deepcopy(self.demand_matrix),
            "distance_matrix": copy.deepcopy(self.distance_matrix)
        }

    def load_map_data(self, data):
        # to refactor
        self.clear_map()
        city_map = {}
        for city_data in data["cities"]:
            id = city_data["id"]
            city = City(self, city_data["x"], city_data["y"], id)
            name = city_data.get("name", None)
            if name is not None:
                city.name = name
            city.pop = city_data.get("pop", 0)
            city.draw(self.canvas)
            self.cities[city.id] = city
            city_map[city.id] = city

        for line_data in data["lines"]:
            city_a = city_map[line_data["city_a"]]
            city_b = city_map[line_data["city_b"]]
            line = Line(city_a, city_b)
            line.time = line_data["time"]
            line.dist = line_data["dist"]
            line.wait = line_data["wait"]
            line.demand = line_data["demand"]
            line.draw(self.canvas)
            self.lines.append(line)
            self.transitions.setdefault(city_a.id, []).append((city_b.id, line))
            self.transitions.setdefault(city_b.id, []).append((city_a.id, line))
        
        self.demand_matrix = data.get("demand_matrix", [])
        self.distance_matrix = data.get("distance_matrix", [])
        self.load_matrix()

        Line.id_iter = itertools.count(start=len(self.lines))

    def load_matrix(self):
        # Create or load matrix
        self.canvas_matrix = tk.Canvas(self.options, height=250)
        self.scrollbar_matrix = ttk.Scrollbar(self.options, orient="vertical", command=self.canvas_matrix.yview)
        self.scrollable_frame_matrix = ttk.Frame(self.canvas_matrix)

        self.scrollable_frame_matrix.bind(
            "<Configure>", lambda e: self.canvas_matrix.configure(scrollregion=self.canvas_matrix.bbox("all"))
        )

        self.canvas_matrix.create_window((0, 0), window=self.scrollable_frame_matrix, anchor="nw")
        self.canvas_matrix.configure(yscrollcommand=self.scrollbar_matrix.set)

        def on_mousewheel(event):
            self.canvas_matrix.yview_scroll(int(-1*(event.delta/120)), "units")

        self.canvas_matrix.bind_all("<MouseWheel>", on_mousewheel)
        
        self.scrollbar_x = ttk.Scrollbar(self.options, orient="horizontal", command=self.canvas_matrix.xview)
        self.canvas_matrix.configure(xscrollcommand=self.scrollbar_x.set)
        self.scrollbar_x.pack(side="bottom", fill="x")

        self.canvas_matrix.pack(side="left", fill="both", expand=True)
        self.scrollbar_matrix.pack(side="right", fill="y")
        
        def on_cell_change(i, j, var):
            def callback(*args):
                val = var.get()
                try:
                    if self.curr_matrix.get() == "Demand":
                        self.update_demand(i, j, int(val))
                    else:
                        self.update_distance(i, j, int(val))
                except ValueError:
                    pass
            return callback

        if len(self.demand_matrix) > 0:
            n = len(self.demand_matrix)
            self.vars_matrix = [[None] * n for _ in range(n)]
        
        init = True
        matrix = self.demand_matrix if self.curr_matrix.get() == "Demand" else self.distance_matrix
        for id in self.cities.keys():
            if init:
                for kid in self.cities.keys():
                    legendx = ttk.Label(self.scrollable_frame_matrix, text=str(f"({kid})"), justify="center", width=3)
                    legendx.grid(row=0, column=kid+1, padx=2, pady=1)
                    legendy = ttk.Label(self.scrollable_frame_matrix, text=str(f"({kid})"), justify="center", width=3)
                    legendy.grid(row=kid+1, column=0, padx=2, pady=1)
                init = False
            for sid in self.cities.keys():
                var = None
                if len(matrix) == 0:
                    # no demand or distance was loaded
                    return
                if matrix[id] == -1:
                    continue
                value = matrix[id][sid] 
                if self.vars_matrix[id][sid] is None:
                    var = tk.StringVar()
                    var.set(str(value))
                    var.trace_add("write", on_cell_change(id, sid, var))
                else:
                    var = self.vars_matrix[id][sid]
                label = ttk.Entry(self.scrollable_frame_matrix, textvariable=var, justify="center", width=3)
                label.grid(row=id+1, column=sid+1, padx=1, pady=1)

                self.vars_matrix[id][sid] = var

        self.canvas_matrix.pack(fill=tk.BOTH)

    # Display related

    def change_matrix(self):
        for i in self.cities.keys():
            for j in self.cities.keys():
                if self.curr_matrix.get() == "Demand":
                    self.vars_matrix[i][j].set(self.demand_matrix[i][j])
                else:
                    self.vars_matrix[i][j].set(self.distance_matrix[i][j])

    def update_demand(self, i, j, val, redraw=True):
        if type(self.demand_matrix[i]) != int and type(self.demand_matrix[j]) != int:
            self.demand_matrix[i][j] = val
            self.demand_matrix[j][i] = val
            if self.curr_matrix.get() == "Demand":
                self.vars_matrix[j][i].set(str(val))
            if redraw:
                line = next((p[1] for p in self.transitions.get(i, []) if p[0] == j), None) # symetrical
                if line:
                    line.demand = val
                    line.delete(self.canvas)
                    line.draw(self.canvas)
       #print("DEMD matrix :::", self.demand_matrix)

    def update_distance(self, i, j, val, redraw=True):
        if type(self.distance_matrix[i]) != int and type(self.distance_matrix[j]) != int:
            self.distance_matrix[i][j] = val
            self.distance_matrix[j][i] = val
            if self.curr_matrix.get() == "Distance":
                self.vars_matrix[j][i].set(str(val))
        #print("DIST matrix :::", self.distance_matrix)

    def clear_matrix_display(self):
        if hasattr(self, 'canvas_matrix') and self.canvas_matrix is not None:
            self.canvas_matrix.destroy()
            self.canvas_matrix = None
        if hasattr(self, 'scrollbar_x'):
            self.scrollbar_x.destroy()
        if hasattr(self, 'scrollbar_matrix'):
            self.scrollbar_matrix.destroy()
        if hasattr(self, 'scrollable_frame_matrix'):
            self.scrollable_frame_matrix.destroy()
    
    def clear_matrices(self):
        self.demand_matrix.clear()
        self.distance_matrix.clear()
        self.vars_matrix.clear()
    
    def update_matrix(self, city_id, added=True):
        matrix = self.demand_matrix if self.curr_matrix.get() == "Demand" else self.distance_matrix
        # Added or removed city
        if not added:
            if city_id >= len(matrix):
                return
            self.demand_matrix[city_id] = -1 # negative means ignore
            self.distance_matrix[city_id] = -1
            for i in range(len(self.vars_matrix[city_id])):
                if self.vars_matrix[i] and self.vars_matrix[i][city_id]:
                    self.vars_matrix[i][city_id].set(str(0))
                if self.vars_matrix[city_id] and self.vars_matrix[city_id][i]:
                    self.vars_matrix[city_id][i].set(str(0))

            self.clear_matrix_display()
            if city_id in self.cities.keys():
                self.cities.pop(city_id)

            self.load_matrix()
            return
        
        size = len(matrix)

        if city_id < len(self.demand_matrix) and self.demand_matrix[city_id] == -1: # was deleted
            self.demand_matrix[city_id] = [0] * len(self.demand_matrix) # there, already existing vars
            for i in range(len(self.demand_matrix)):
                if type(self.demand_matrix[i]) != int:
                    self.demand_matrix[i][city_id] = 0
                if type(self.demand_matrix[city_id]) != int:
                    self.demand_matrix[city_id][i] = 0
            if not(city_id < len(self.distance_matrix) and self.distance_matrix[city_id] == -1):
                return

        if city_id < len(self.distance_matrix) and self.distance_matrix[city_id] == -1: # was deleted
            self.distance_matrix[city_id] = [0] * len(self.distance_matrix) # there, already existing vars
            for i in range(len(self.distance_matrix)):
                if type(self.distance_matrix[i]) != int:
                    self.distance_matrix[i][city_id] = 0
                if type(self.distance_matrix[city_id]) != int:
                    self.distance_matrix[city_id][i] = 0
            return

        # Extend squarely
        for i in range(len(self.demand_matrix)):
            if self.demand_matrix[i] == -1:
                continue # ignore
            self.demand_matrix[i].append(0)
            self.vars_matrix[i].append(None)
        self.demand_matrix.append([0] * (len(self.demand_matrix) + 1))
        self.vars_matrix.append([None] * (len(self.demand_matrix) + 1))

        for i in range(len(self.distance_matrix)):
            if self.distance_matrix[i] == -1:
                continue # ignore
            self.distance_matrix[i].append(0)
        self.distance_matrix.append([0] * (len(self.distance_matrix) + 1))

        legendy = ttk.Entry(self.scrollable_frame_matrix, justify="center", width=3)
        legendy.insert(0, f"({city_id})")
        legendy.grid(row=size + 1, column=0, padx=1, pady=1)
        legendx = ttk.Entry(self.scrollable_frame_matrix, justify="center", width=3)
        legendx.insert(0, f"({city_id})")
        legendx.grid(row=0, column=size + 1, padx=1, pady=1)

        def on_write(i, j, var):
            def callback(*_):
                try:
                    val = int(var.get())
                    if self.curr_matrix.get() == "Demand":
                        self.update_demand(i, j, val)
                    else:
                        self.update_distance(i, j, val)
                except ValueError:
                    val = 0
            return callback
        
        for j in range(size + 1):
            var = tk.StringVar(value='0')
            var.trace_add("write", on_write(size, j, var))
            entry = ttk.Entry(self.scrollable_frame_matrix, textvariable=var, justify="center", width=3)
            entry.grid(row=size + 1, column=j + 1, padx=1, pady=1)
            self.vars_matrix[size][j] = var

            if j < size:
                var2 = tk.StringVar(value='0')
                var2.trace_add("write", on_write(size, j, var2))
                entry2 = ttk.Entry(self.scrollable_frame_matrix, textvariable=var2, justify="center", width=3)
                entry2.grid(row=j + 1, column=size + 1, padx=1, pady=1)
                self.vars_matrix[j][min(len(self.vars_matrix[j])-1, size)] = var2
            else:
                self.vars_matrix[j][min(len(self.vars_matrix[j])-1, size)] = var  # same var

    def display_formula(self, _):
        from PIL import Image, ImageTk
        import io
        if self.init:
            self.init = False
            return

        from metrics import LATEX_FORMULAS
        formula = LATEX_FORMULAS.get(self.metric.get())
        if not formula:
            if self.latex_label:
                self.latex_label.destroy()
                self.latex_label = None
            return

        fig = Figure(figsize=(5, 1), dpi=300)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, f"${formula}$", fontsize=18, ha="center", va="center")

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)

        image = Image.open(buf)
        new_size = (int(image.width/6), int(image.height/6))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(image)

        if self.latex_label:
            self.latex_label.destroy()
        self.latex_label = tk.Label(self.formula_frame, image=photo, bg='white')
        self.latex_label.image = photo  # keep a reference
        self.latex_label.pack(expand=True)