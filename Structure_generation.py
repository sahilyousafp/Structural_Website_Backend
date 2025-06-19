# Structure generation module: defines StructuralSystem class for computing structural system JSON
import numpy as np
from shapely.geometry import Polygon, Point

class StructuralSystem:
    def __init__(self, meshes):
        """
        Initialize with a list of trimesh mesh objects.
        """
        self.meshes = meshes
        self.building_polys, self.wall_data, self.global_max_z = self.extract_wall_data()

    def extract_wall_data(self):
        """
        Extract wall polygon footprints and heights from mesh bounding boxes.
        """
        wall_data = []
        building_polys = []
        max_z = 0.0
        for mesh in self.meshes:
            bounds = mesh.bounds  # [[minx,miny,minz],[maxx,maxy,maxz]]
            minx, miny, _ = bounds[0]
            maxx, maxy, maxz = bounds[1]
            poly = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
            wall_data.append({'polygon': poly, 'max_z': maxz})
            building_polys.append(poly)
            max_z = max(max_z, maxz)
        return building_polys, wall_data, max_z

    def get_wall_height(self, x, y):
        """
        Get height at a point based on closest wall footprint.
        """
        pt = Point(x, y)
        closest = None
        dist0 = float('inf')
        for w in self.wall_data:
            d = w['polygon'].exterior.distance(pt)
            if d < dist0:
                dist0 = d
                closest = w
        return closest['max_z'] if closest else self.global_max_z

    def compute(self, num_floors):
        """
        Compute column and beam layout for given number of floors.
        Returns a dictionary with 'num_floors', 'columns', and 'beams'.
        """
        MaxS, MinS = 6.0, 3.0
        generated = []
        beams_2d = []
        # sort rooms by area
        rooms = sorted([(poly, poly.area) for poly in self.building_polys], key=lambda x: -x[1])
        for poly, _ in rooms:
            minx, miny, maxx, maxy = poly.bounds
            nx = int(np.ceil((maxx - minx) / MaxS))
            ny = int(np.ceil((maxy - miny) / MaxS))
            xs = np.linspace(minx, maxx, nx + 1)
            ys = np.linspace(miny, maxy, ny + 1)
            # grid columns
            for x in xs:
                for y in ys:
                    if all(np.linalg.norm([x - gx, y - gy]) >= MinS for gx, gy in generated):
                        generated.append((x, y))
            # beams
            for x in xs:
                beams_2d.append(((x, miny), (x, maxy)))
            for y in ys:
                beams_2d.append(((minx, y), (maxx, y)))
            # corners
            for c in poly.exterior.coords:
                if all(np.linalg.norm([c[0] - gx, c[1] - gy]) >= MinS * 0.5 for gx, gy in generated):
                    generated.append(c)
        # assemble JSON
        result = {'num_floors': num_floors, 'columns': [], 'beams': []}
        for x, y in generated:
            h = self.get_wall_height(x, y)
            result['columns'].append({'base': [x, y, 0.0], 'top': [x, y, h]})
        for (x1, y1), (x2, y2) in beams_2d:
            h1 = self.get_wall_height(x1, y1)
            h2 = self.get_wall_height(x2, y2)
            floor_h = min(h1, h2)
            for floor in range(1, num_floors + 1):
                z = floor_h / num_floors * floor
                if z > h1 or z > h2:
                    continue
                result['beams'].append({'start': [x1, y1, z], 'end': [x2, y2, z], 'floor': floor})
        return result
