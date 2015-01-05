import nxsvg
import networkx as nx
from sys import stdout

g = nx.DiGraph()
g.add_node('Vous', position=(-1, 3), fill='blue', font_size=30)
g.add_node('Tu', position=(1, 3), fill='green', font_size=30)
g.add_node('I am confused', position=(0, 3), fill='red', font_size=30)
g.add_node('Are you an adult?', position=(0., 0.), rx=0, ry=0, stroke_width='3px', fill='gray', stroke='none', font_size=40)
g.add_node('Are you speaking to a child?', position=(-1., 1.))
g.add_node('Is the child like a prince or something?', position=(0., 1.))
g.add_node('Are you speaking to an adult?', position=(1., 1.))
g.add_node('Is the adult a family member?', position=(1., 2.))
g['Are you an adult?']['Are you speaking to a child?'] = "Yes"
g['Are you an adult?']['Are you speaking to an adult?'] = "No"
g['Are you speaking to an adult?']['Is the adult a family member?'] = "Yes"
g['Are you speaking to an adult?']['Tu'] = "Yes"
g['Is the adult a family member?']['Tu'] = "Yes"
g['Is the adult a family member?']['Vous'] = "No"
g['Are you speaking to a child?']['Is the child like a prince or something?'] = "Yes"
g['Are you speaking to a child?']['I am confused'] = "No"
g['Is the child like a prince or something?']['Vous'] = "Yes"
g['Is the child like a prince or something?']['Tu'] = "No"

rend = nxsvg.SVGRenderer()
stdout.write(rend.draw(g, size=('1000px', '400px')))
