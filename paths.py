from vector3 import Vector3
from typing import Union, List, Tuple

"""
Paths.py

3D Path data utilities

Author: Logan R Boehm
"""

class Point:
    """
    Position in 3D space consiting of a 3D vector and a time
    """
    def __init__(self, position: Vector3, time: float):
        """
        Creates a new point object given:
        position: Physical location in 3D space
        time: Time in seconds
        """
        self.position = position
        self.time = time
    
    def __repr__(self):
        return f"Point: {self.position} @ {self.time}s"

class Path:
    """
    Path through 3D space
    """
    def __init__(self, starting_list: Union[List[Point], List[Tuple[Vector3, float]]] = []):
        """
        Create a new path through 3D space.

        If no starting list is specified, the path will be empty

        The starting list can be a list of either Point objects, 
        or a tuple containing the Vector3 position and time

        Point objects can be passed in any order, they will be sorted by time automatically
        """
        self.points = []
        self.start_time = None
        self.end_time = None

        if len(starting_list) == 0:
            return
        
        while starting_list:
            new = starting_list.pop()
            self.add_point(new)
    
    def add_point(self, new_point: Union[Point, Tuple[Vector3, float]]) -> None:
        """
        Adds a new position to the path, inserting at the proper position
        Takes either a point object, or a tuple containing the position and the time in seconds
        """

        if isinstance(new_point, Point):
            # New point is a Point, no action needs to be taken
            pass
        elif isinstance(new_point, tuple) and len(new_point) >= 2 and isinstance(new_point[0], Vector3) and (isinstance(new_point[1], float) or isinstance(new_point[1], int)): 
            # New point needs to be converted to a Point object
            new_point = Point(new_point[0], new_point[1])
        else:
            raise ValueError(f"An incorrectly formatted point object was passed: {new_point}")

        if self.start_time is None:
            # No points have been added yet
            self.start_time = new_point.time
            self.end_time = new_point.time
            self.points = [new_point]
        else:
            # Find insertion point
            index = 0
            while index < len(self.points) and self.points[index].time < new_point.time:
                index += 1
            
            if index == 0:
                self.start_time = new_point.time
            
            if index == len(self.points):
                self.end_time = new_point.time

            self.points.insert(index, new_point)


if __name__ == "__main__":
    p1 = (Vector3(1,2,3),2)
    p2 = Point(Vector3(4,5,6),1)
    p3 = (Vector3(7,8,9),3)
    path = Path([p1,p2,p3])
    print(path.points)
    print(path.start_time)
    print(path.end_time)