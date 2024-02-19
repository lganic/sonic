from typing import Union, List, Tuple, Generator
import numpy as np
import math

"""Vector3.py

3D Vector datatype, with common mathematical functions built in

Under the hood, this is a wrapper library for numpy vectors.
Just makes the implementation way easier.

Author : Logan R Boehm
"""

def _eZip(*items):
    #enumerate and zip specified items
    return enumerate(zip(*items))

class Vector3:
    """
    3D vector datatype using NumPy arrays
    """
    def __init__(self, *components: Union[Tuple[float, float, float], List[float], np.ndarray]):
        if len(components) == 0 or components[0] is None:
            # User gave nothing or None, return the zero vector
            self.components = np.array([0.0, 0.0, 0.0])      
        elif isinstance(components[0], (tuple, list, np.ndarray)):
            # Components were passed as a tuple, list, or np.ndarray, use it directly
            self.components = np.array(components[0], dtype=float)
        else:
            # Components were passed separately, convert to np.ndarray
            self.components = np.array(components, dtype=float)
        
        # Ensure the vector is 3D
        if self.components.size != 3:
            raise ValueError("Vector3 must have exactly 3 components.")

    @staticmethod
    def zero() -> 'Vector3':
        """
        Return the zero vector
        """
        return Vector3()
    
    @staticmethod
    def X() -> 'Vector3':
        """
        Return the x axis vector
        """
        return Vector3(1, 0, 0)

    @staticmethod
    def Y() -> 'Vector3':
        """
        Return the y axis vector
        """
        return Vector3(0, 1, 0)

    @staticmethod
    def Z() -> 'Vector3':
        """
        Return the z axis vector
        """
        return Vector3(0, 0, 1)

    def __str__(self) -> str:
        """
        Return string representation of self.
        """
        return "Vector3: <" + ", ".join([str(k) for k in self.components]) + ">"

    def __repr__(self) -> str:
        return str(self)

    def __getattr__(self, attr) -> float:
        """
        Adds support for individual attribute retrieval, ie:

        vector = Vector3(1,2,3)
        print(vector.y) # prints 2
        """
        attr = attr.lower()
        if attr in {'x', 'y', 'z'}:
            index = {'x': 0, 'y': 1, 'z': 2}[attr]
            return self.components[index]
        else:
            raise AttributeError(f"'Vector3' object has no attribute '{attr}'")

    def copy(self) -> 'Vector3':
        """
        Return copy of the vector
        """
        return Vector3(self.components)

    def __add__(self, other_vector: 'Vector3') -> 'Vector3':
        """
        Add one vector to another, and return a new result vector
        """

        return Vector3(self.components + other_vector.components)

    def __iadd__(self, other_vector: 'Vector3') -> None:
        """
        Add another vector to this vector
        """
        #Loop over all components from the new vector
        for index, otherComponent in enumerate(other_vector.components):
            #Add new component to the internal component
            self.components[index] += otherComponent

        return self

    def __sub__(self, other_vector: 'Vector3') -> 'Vector3':
        """
        Subtract one vector from another, and return a new result vector
        """
        return Vector3(self.components - other_vector.components)

    def __isub__(self, other_vector: 'Vector3') -> 'Vector3':
        """
        Subtract a vector from this vector
        """
        #Loop over all components from the new vector
        for index, otherComponent in enumerate(other_vector.components):
            self.components[index] -= otherComponent # Subtract the new component from this vector

        return self

    def dot(self, other_vector: 'Vector3') -> float:
        """
        Return the dot product of two vectors
        """
#        summation = 0
#        # Loop over all components in both vectors
#        for v1comp, v2comp in zip(self.components,other_vector.components):
#            # Multiply the components from the vectors, and add them to the total
#            summation+= v1comp * v2comp
        
        summation = np.dot(self.components, other_vector.components)

        return summation

    def cross(self, other_vector: 'Vector3') -> 'Vector3':
        """
        Return the cross product of two vectors
        """
#        #Calculate the components using matrix determinant
#        c1 = matrixDet2x2(self.y, self.z, other_vector.y, other_vector.z)
#        c2 = -matrixDet2x2(self.x, self.z, other_vector.x, other_vector.z)
#        c3 = matrixDet2x2(self.x, self.y, other_vector.x, other_vector.y)
#        return Vector3([c1, c2, c3])
        
        new_components = np.cross(self.components, other_vector.components)
        return Vector3(new_components)

    def projection(self, other_vector: 'Vector3') -> 'Vector3':
        """
        Returns the projection of self onto other
        """
        dotProduct = self.dot(other_vector)
        return other_vector.norm() * (dotProduct / (other_vector.len() ** 2))

    def projectionLength(self, other_vector: 'Vector3') -> 'Vector3':
        """
        Returns the magnitude of the projection vector of self onto other
        (this is faster if all you need is the magnitude of the projection)
        This method also preserves the sign of the projection
        """
        dotProduct = self.dot(other_vector)
        return dotProduct / other_vector.len()

    def __mul__(self, const: float) -> 'Vector3':
        """
        Multiply a vector by a constant, and return a new vector
        """
        out=self.copy()
        for a in range(3):
            out.components[a]*=const

        return out

    def __imul__(self, const: float) -> None:
        """
        Multiply this vector by a constant
        """
        return None

    def __neg__(self) -> 'Vector3':
        """
        Create a new vector in the opposite direction to this vector
        """
        return self * -1
    
    def __iter__(self) -> Generator[float, None, None]:
        """
        Enables unpacking of values
        """
        yield self.x
        yield self.y
        yield self.z

    def len(self) -> float:
        """
        Return the length of the vector
        """
        summa = 0
        for component in self.components:
            summa += component**2

        return summa ** .5

    def __eq__(self, other_vector : 'Vector3') -> bool:
        """
        Compare two vectors to see if they are equal

        Also this usage is also supported for cleaner implementations:
        vector == 0 # Compares the vector to the zero vector
        """
        if other_vector == 0:
            other_vector = Vector3.zero()

        for v1comp, v2comp in zip(self.components,other_vector.components):
            if v1comp!=v2comp:
                return False

        return True

    def norm(self) -> 'Vector3':
        """
        Return the vector, but normalized to a length of 1
        """
        if self == 0:
            return self
        return self * (1/self.len())
    
    def apply_rot_matrix(self, matrix: np.ndarray) -> 'Vector3':
        """
        Apply a 3x3 rotation matrix to this vector, while preserving length
        return the result as a vector
        """
        length = self.len()
        vec_mat = self.norm().components.transpose()
        result = matrix @ vec_mat
        return Vector3(result * length)

    def apply_matrix(self, matrix: np.ndarray) -> 'Vector3':
        """
        Apply a matrix to this vector
        return the result as a vector
        """
        vec_mat = self.components.transpose()
        result = matrix @ vec_mat
        return Vector3(result)
    
    def set_length(self, new_length: float) -> None:
        """
        Set the length of this vector to the specified length
        """
        mult = new_length / self.len()
        self.components *= mult

def areColinear(v1: Vector3, v2: Vector3, v3: Vector3) -> bool:
    """
    Determine if the 3 positional vectors are colinear
    """
    d1 = (v2-v1).norm()
    d2 = (v3-v1).norm()
    if d1==d2 or -d1==d2:
        return True
    return False

def matrixDet2x2(tl: float,tr: float,bl: float,br: float)-> float:
    """
    Return matrix determinant for 2x2 matrix specified
    """
    return tl * br - tr * bl

def create_rotation_matrix_from_axis(axis: 'Vector3', angle: float):
    """
    Use the Rodrigues' rotation formula to create the rotation matrix around the rotation axis by the calculated angle.

    Angle is in radians
    """
    axis = axis.norm()
    skew_sym_mat = np.array([[0, -axis.z, axis.y], [axis.z, 0, -axis.x], [-axis.y, axis.x, 0]], dtype = float)
    ident = np.identity(3, dtype = float)
    return ident + (math.sin(angle) * skew_sym_mat) + ((1 - math.cos(angle)) * (skew_sym_mat @ skew_sym_mat))

def create_rotation_matrix_from_vectors(vector: Vector3, axis: Vector3):
    """
    Create a rotation matrix that rotates the target vector to the axis
    """
    vec_norm = vector.norm()
    axis = axis.norm()
    rotation_axis = vec_norm.cross(axis).norm()
    theta = math.acos(vec_norm.dot(axis))
    return create_rotation_matrix_from_axis(rotation_axis, theta)


def main_1():
    v1 = Vector3(3,-3,1)
    v2 = Vector3([4,9,2])
    cp = v1.cross(v2)
    print(f"{v1} X {v2} = {cp}")
    cpNorm = cp.norm()
    print(cpNorm)
    print(cpNorm.len())
    v1 = Vector3(3,0,0)
    v2 = Vector3(1,0,0)
    v3 = Vector3(2,0,0)
    print(areColinear(v1,v2,v3))
    v=Vector3(3,4,0)
    u=Vector3(5,-12,0)
    proj = v.projection(u)
    print(proj)
    print(proj.len())
    print(v.projectionLength(u))

def main():
    d = Vector3(1, 2, 3)
    z_axis = Vector3(0, 0, 1)
    rot_mat = create_rotation_matrix_from_vectors(d, z_axis)
    new_vec = d.apply_rot_matrix(rot_mat)
    print(new_vec)

if __name__ == '__main__':
    main()
