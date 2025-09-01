import itertools
import tkinter

RADIUS=25

class City:
    def __init__(self, tab, x, y, id=None):
        self.x = x
        self.y = y
        self.r = RADIUS
        self.tab = tab
        self.id = self.get_next_id() if id is None else id
        tab.city_ids.add(self.id)
        self.name = None
        self.pop = 0

    def draw(self, canvas: tkinter.Canvas):
        self.fig = canvas.create_oval(self.x-self.r,self.y-self.r,self.x+self.r,self.y+self.r, fill="black")
        self.id_fig = canvas.create_text(self.x, self.y, text=self.id, fill="white", tags="id")
        txt = ""
        if self.name is not None:
            txt = self.name
        if self.pop > 0:
            if len(txt) > 0:
                txt += f"\n{self.pop}"
            else:
                txt = str(self.pop)
        if len(txt) > 0:
            self.name_fig = canvas.create_text(self.x, self.y + 40, text=txt, font=("Arial", 11), fill="black", tags="name")
    
    def delete(self, canvas):
        canvas.delete(self.fig)
        canvas.delete(self.id_fig)
        if hasattr(self, 'name_fig') and self.name_fig is not None:
            canvas.delete(self.name_fig)
    
    def discard(self, canvas):
        self.delete(canvas)
        self.tab.city_ids.discard(self.id)

    def get_next_id(self):
        i = 0
        while i in self.tab.city_ids:
            i += 1
        return i

    def to_dict(self):
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y
        }

    @staticmethod
    def from_dict(data, tab):
        city = City(tab, data["x"], data["y"], data["id"])
        return city
