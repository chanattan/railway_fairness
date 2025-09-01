import itertools
import tkinter

class Line:
    id_iter = itertools.count()
    def __init__(self, city_a, city_b):
        self.city_a = city_a
        self.city_b = city_b
        self.id = next(Line.id_iter) # to fix, we don't use it for now
        self.time = 0
        self.dist = 0
        self.wait = 0
        self.demand = 0

    def draw(self, canvas: tkinter.Canvas):
        self.fig = canvas.create_line(self.city_a.x, self.city_a.y, self.city_b.x, self.city_b.y, fill="black", width=2)
        canvas.tag_lower(self.fig)
        ax, ay = self.city_a.x, self.city_a.y
        bx, by = self.city_b.x, self.city_b.y
        mx, my = (ax + bx) / 2, (ay + by) / 2

        dx, dy = bx - ax, by - ay
        length = (dx**2 + dy**2) ** 0.5
        offset = 15
        if length != 0:
            ox = -dy / length * offset
            oy = dx / length * offset
        else:
            ox = oy = 0

        txt = f"t: {self.time}\nd: {self.dist}\nw: {self.wait}\nτ: {self.demand}"
        self.id_fig = canvas.create_text(mx + ox, my + oy, text=txt, fill="blue", font=("Arial", 24), tags="lid")

    def delete(self, canvas):
        canvas.delete(self.fig)
        canvas.delete(self.id_fig)
    
    def to_dict(self):
        return {
            "id": self.id,
            "city_a_id": self.city_a.id,
            "city_b_id": self.city_b.id,
            "time": self.time,
            "dist": self.dist,
            "wait": self.wait,
            "demand": self.demand
        }

    @staticmethod
    def from_dict(data, cities_by_id):
        city_a = cities_by_id[data["city_a_id"]]
        city_b = cities_by_id[data["city_b_id"]]
        line = Line(city_a, city_b)
        line.id = data["id"]
        line.time = data["time"]
        line.dist = data["dist"]
        line.wait = data["wait"]
        line.demand = data["demand"]
        return line