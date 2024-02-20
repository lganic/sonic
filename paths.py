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
    
    def get_at_time(self, t: float, end_mode = True)-> Vector3:
        """
        Interpolate the path to find the state of the path at time t

        end_mode dictates how the path behaves when t is outside the path
        If end mode is true, then a path will "guess" the position based
        on the position of the first two, or last two values
         
        If end mode is false, then the vectors and the end will be returned

        If the path is empty, zero vector is returned 
        """

        if len(self.points) == 0:
            return Vector3.zero()
        
        if len(self.points) == 1:
            return self.points[0].position

        if t == self.start_time:
            return self.points[0].position
        
        if t == self.end_time:
            return self.points[-1].position

        if t >= self.start_time and t <= self.end_time:
            # Value falls within path
            index = 0
            while t > self.points[index].time:
                index += 1
            t_range = self.points[index].time - self.points[index - 1].time
            t_proc = t - self.points[index - 1].time
            t_lerp = t_proc / t_range
            output = self.points[index - 1].position + ((self.points[index].position - self.points[index - 1].position) * t_lerp)
            return output
        
        if not end_mode:
            # Return end points mode
            if t < self.start_time:
                return self.points[0].position
            return self.points[-1].position
        else:
            # Predict from end points mode
            if t < self.start_time:
                t_del = self.points[1].time - self.points[0].time
                t_offset = self.start_time - t
                t_mul = t_offset / t_del
                dir_vec = self.points[0].position - self.points[1].position
                offset_vec = dir_vec * t_mul
                return self.points[0].position + offset_vec
            t_del = self.points[-1].time - self.points[-2].time
            t_offset = t - self.end_time
            t_mul = t_offset / t_del
            dir_vec = self.points[-1].position - self.points[-2].position
            offset_vec = dir_vec * t_mul
            return self.points[-1].position + offset_vec



if __name__ == "__main__":
    p1 = (Vector3(1,2,3),2)
    p2 = Point(Vector3(6,5,4),1)
    p3 = (Vector3(9,8,7),3)
    path = Path([p1,p2,p3])
    print(path.points)
    print(path.start_time)
    print(path.end_time)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    t = 0
    x = []
    y = []
    z = []
    while t < 5:
        pos = path.get_at_time(t,True)
        x.append(pos.x)
        y.append(pos.y)
        z.append(pos.z)
        t+=.1
    ax.plot(x, y, z, label='3D Line')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Line Graph')
    ax.legend()
    plt.show()