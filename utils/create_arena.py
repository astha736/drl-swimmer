import xml.etree.ElementTree as ET

# AgnathaX head is at [x=0, y=0] at simulation start


def get_peg_positions():
    """Returns a list of peg positions as string"""
    return ["1.0 1.0 0 0 0 0"]


# get peg positions
peg_positions = get_peg_positions()

# load models
geom = ET.parse("models/peg_geom.xml")
geom_root = geom.getroot()
tree = ET.parse("models/raw_model.sdf")
root = tree.getroot()


for peg_position in peg_positions:
    # add collision tag
    col = ET.Element("collision")
    col.set("name", "peg_X")
    pose = ET.SubElement(col, "pose")
    pose.text = peg_position
    col.append(geom_root)
    root.find("./world/model/link").append(col)

    # add visual tag
    vis = ET.Element("visual")
    vis.set("name", "peg_X")
    pose = ET.SubElement(col, "pose")
    pose.text = peg_position
    vis.append(geom_root)
    root.find("./world/model/link").append(vis)

# write to file
tree.write("models/pegs_generated.sdf")
