import numpy as np
import vector3
from vector3 import Vector3, areColinear
from typing import Tuple, Generator, List
import math

Point = Tuple[float, float]

"""Reflection.py

Calculating the behavior of diffuse wave reflections over time.

The method employed approximates the received reflection intensity over time
by approximating the integral of the intersection of
an expanding ellipsoid with the diffuse reflection plane.

using documentation from:

P. Klein, "On the Ellipsoid and Plane Intersection Equation,"
Applied Mathematics, Vol. 3 No. 11, 2012, pp. 1634-1640.
doi: 10.4236/am.2012.311226.

Author : Logan R Boehm
"""

class Ray:
    """
    3D Ray with a starting position, and a direction
    """
    def __init__(self, position: Vector3, direction:Vector3):
        if direction == 0:
            raise ValueError("direction cannot be zero")
        self.position = position.copy()
        self.direction = direction.copy()
    
    @staticmethod
    def from_2_points(point1: Vector3, point2: Vector3):
        """
        Create a ray from two points
        """
        directionVector = point2 - point1
        return Ray(point1, directionVector)

    def t_form(self) -> tuple:
        """
        Return the Ray in the form of a line with the form:

        ((a, b), (c, d), (e, f))

        Which fits the form : 
        x = a + bt
        y = c + dt
        z = e + ft
        """
        return tuple(zip(self.position.components,self.direction.components))
    
    def calculate_point(self,t_position : float):
        """
        Given a value of t, return the corresponding position on the Ray
        """
        position = Vector3()
        for index in range(3):
            position.components[index] = self.position.components[index] + t_position * self.direction.components[index]
        return position

class Plane:
    "3D plane"
    def __init__(self,location: Vector3, orientation1: Vector3, orientation2: Vector3):
        """
        Create a plane using a location on the plane and two directions,
        the direction vectors originate from the location and run parallel
        to the plane itself.
        """
        self.location = location.copy()
        self.orientation1 = orientation1.copy()
        self.orientation2 = orientation2.copy()
        if self.orientation1 == 0 or self.orientation2 == 0:
            raise ValueError("Orientation cannot be zero")
        #formula for plane is given by normal = coef_eq
        self.normal = self.orientation1.cross(self.orientation2).norm()
        self.coef_eq = self.normal.dot(self.location)
    
    @staticmethod
    def from_points(centerpoint: Vector3, point1: Vector3, point2: Vector3)-> 'Plane':
        """
        Given three points, return a plane that intersects all 3
        """
        if areColinear(centerpoint, point1, point2):
            raise ValueError("Points cannot be colinear")
        direction1 = centerpoint - point1
        direction2 = centerpoint - point2
        return Plane(centerpoint, direction1, direction2)
    
    @staticmethod
    def from_coefficients(xCoeff : float, yCoeff : float, zCoeff : float, eq):
        """
        Given a plane with the function 
        xCoeff * x + yCoeff * y + zCoeff * z = eq

        Create a plane
        """
        v1 = Vector3(eq/xCoeff, 0, 0)
        v2 = Vector3(0, eq/yCoeff, 0)
        v3 = Vector3(0, 0, eq/zCoeff)
        return Plane.from_points(v1,v2,v3)
    
    def __repr__(self):
        """
        For easier debugging, this returns a function representation to be pasted into graphing software
        """
        x, y, z = self.normal
        return f"{x:.20f}x+{y:.20f}y+{z:.20f}z={self.coef_eq}"
        
    def intersection(self,input_ray: Ray)-> Vector3:
        """
        Compute the intersection between a Ray and this plane,
        returns the position in 3D space.
        """
        # Represent the Ray in its line form
        Ray_t = input_ray.t_form()
        # Perform some linear algebra
        reg_sum = 0
        t_sum = 0
        for coeff, RayPart in zip(self.normal.components,Ray_t):
            reg_sum += coeff * RayPart[0]
            t_sum += coeff * RayPart[1]
        t_pos = (self.coef_eq - reg_sum) / t_sum
        return input_ray.calculate_point(t_pos)
    
    def intersection2d(self,Ray: Ray)-> Point:
        """
        Returns the X,Y coordinate of the intersection between Ray and plane
        The X and Y axis are the orientations specified, so orthogonal vectors
        are recommended, but not required.
        orientation1 = X axis
        orientation2 = Y axis
        """
        intersectionPoint = self.intersection(Ray)
        delta = intersectionPoint - self.location
        xDelta = delta.projectionLength(self.orientation1)
        yDelta = delta.projectionLength(self.orientation2)
        return (xDelta, yDelta)
    
    def transform(self, rotation: np.ndarray, translation: Vector3):
        """
        Translate and rotate the plane using the translation vector,
        and rotation matrix, and return a new plane
        """
        new_location = (self.location + translation).apply_rot_matrix(rotation)
        new_orient1 = self.orientation1.apply_rot_matrix(rotation)
        new_orient2 = self.orientation2.apply_rot_matrix(rotation)
        return Plane(new_location, new_orient1, new_orient2)


class Ellipsoid:
    """
    Triaxial ellipsoid
    """
    def __init__(self, center: Vector3, ax1 : Vector3, ax2 : Vector3, ax3 : Vector3):
        """
        Create a new ellipsoid, given the center, and the 3 primary axis

        The classification of which axis is the semi-major, semi-minor and intermediate
        is performed automatically
        """
        self.center = center

        # Check whether the axis are orthogonal to each other
        dot_1 = abs(ax1.dot(ax2))
        dot_2 = abs(ax2.dot(ax3))
        dot_3 = abs(ax3.dot(ax1))
        if dot_1 > 1e-5 or dot_2 > 1e-5 or dot_3 > 1e-5:
            raise ValueError("Axis are not orthogonal")
        
        # Get the lengths of each axis
        length_1 = ax1.len()
        length_2 = ax2.len()
        length_3 = ax3.len()

        # Determine which axis is which
        lengths = [length_1, length_2, length_3]

        biggest = lengths.index(max(lengths))
        smallest = lengths.index(min(lengths))

        temp_index = [0, 1, 2]
        temp_index.remove(biggest)
        temp_index.remove(smallest)

        middle = temp_index[0]

        # Assign all axis to their corresponding values
        self.all_axis = [ax1.copy(), ax2.copy(), ax3.copy()]
        self.semi_minor = self.all_axis[smallest]
        self.semi_major = self.all_axis[biggest]
        self.intermediate = self.all_axis[middle]
    
    @staticmethod
    def from_focal_points(focal_1: Vector3, focal_2: Vector3, distance: float) -> 'Ellipsoid':
        """
        Given two focal points, and the distance from the ellipsoid 
        to the focal points, construct an ellipsoid

        BTW, This constructs a revolved ellipse
        """

        # I'd check if focal_1 == focal_2, but im not really sure if it would break or not

        center = (focal_1 + focal_2) * .5

        focal_to_focal = focal_2 - focal_1

        center_dist_d2 = (focal_1 - center).len()

        major_axis_length = distance / 2
        minor_axis_length = math.sqrt(math.pow(major_axis_length, 2) - math.pow(center_dist_d2, 2))

        major_axis = focal_to_focal.norm() * major_axis_length
        
        cross_axis = Vector3.Z()
        if major_axis.x == 0 and major_axis.y == 0:
            # Major axis lies on z axis, we need another axis to choose cross product
            cross_axis = Vector3.X()
        
        minor_axis_1 = major_axis.cross(cross_axis).norm() * minor_axis_length
        minor_axis_2 = major_axis.cross(minor_axis_1).norm() * minor_axis_length

        return Ellipsoid(center, major_axis, minor_axis_1, minor_axis_2)

    def __repr__(self):
        """
        For easier debugging, this returns a function representation to be pasted into graphing software
        """
        a, b, c = self.coefficients()
        return f"((x^2)/({(a**2):.20f}))+((y^2)/({(b**2):.20f}))+((z^2)/({(c**2):.20f}))=1"

    def move_and_align_with_origin(self) -> tuple[np.ndarray, Vector3]:
        """
        Moves the ellipsoid to the origin, and rotates it to align:

        semi major axis with x axis
        intermediate axis with y axis
        semi minor axis with z axis

        Returns the corresponding rotation matrix, and translation used
        """

        # Create translation vector
        translation = - self.center
        self.center = Vector3.zero()

        # Rotate semi major axis to x
        rotation_1 = vector3.create_rotation_matrix_from_vectors(self.semi_major, Vector3.X())
        self.semi_major = self.semi_major.apply_rot_matrix(rotation_1)
        self.semi_minor = self.semi_minor.apply_rot_matrix(rotation_1)
        self.intermediate = self.intermediate.apply_rot_matrix(rotation_1)

        rotation_2 = vector3.create_rotation_matrix_from_vectors(self.semi_minor, Vector3.Y())
        self.semi_minor = self.semi_minor.apply_rot_matrix(rotation_2)
        self.intermediate = self.intermediate.apply_rot_matrix(rotation_2)
        
        if self.intermediate.z < 0:
            self.intermediate = -self.intermediate # Lazy, but works
        
        self.all_axis = [self.semi_major, self.semi_minor, self.intermediate]
        return (rotation_1 @ rotation_2, translation)
    
    def coefficients(self) -> Tuple[float, float, float]:
        """
        Find a,b,c in the ellipsoid equation:

        (x^2 / a^2) + (y^2 / b^2) + (z^2 / c^2) = 1

        and return them as an ellipse

        NOTE: the ellipse MUST BE AT, AND ALIGNED WITH ORIGIN
        run move_and_align_with_origin to transform the scene properly
        """

        if not self.center == 0:
            # Its 1 AM, im gonna ignore the case where its at the origin, but not aligned
            # TODO : Put that in
            raise ValueError("Ellipsoid not centered at origin, read docs")

        a = self.semi_major.len()
        b = self.semi_minor.len()
        c = self.intermediate.len()

        return (a, b, c)
    
    def size_ellipse_intersection(self, intersect_plane: Plane, intersect: Vector3) -> Tuple[float, float]:
        """
        Given a plane, and a point known to intersect both the plane and the 
        ellipsoid, return the size of the ellipse produced by the intersection

        Returns a tuple containing the length of the semi axes in no particular order
        """

        # Algorithm used here is described in "On the Ellipsoid and Plane Intersection Equation"
        # Citation above, big thanks to Dr Klein for that masterwork
        # A foreword before reading it though (a,b) = a dot b 
        
        a, b, c = self.coefficients()
        d_mat = np.diag([a, b, c])

        # function to perform the weird dot product matrix stuff hes doing
        def weird_dot_matrix(vec1, vec2):
            return (vec1.x * vec2.x / math.pow(a, 2)) + (vec1.y * vec2.y / math.pow(b, 2)) + (vec1.z * vec2.z / math.pow(c,2))

        # This is going to be incomprehensible to read, 
        # its just formula 11 from the paper, variables named after bolded letters
        q = intersect
        r = intersect_plane.orientation1
        s = intersect_plane.orientation2

        qq = weird_dot_matrix(q, q)
        qr = weird_dot_matrix(q, r)
        rr = weird_dot_matrix(r, r)
        qs = weird_dot_matrix(q, s)
        ss = weird_dot_matrix(s, s)

#        print("Should be zero", weird_dot_matrix(r, s))

        d = qq - (math.pow(qr, 2) / rr) - (math.pow(qs, 2) / ss)

        # not onto formula 10
        A = math.sqrt((1 - d) / rr)
        B = math.sqrt((1 - d) / ss)

        return (A, B)


        

def find_ellipsoid_tangent_point(focal_1 : Vector3, focal_2: Vector3, targetPlane: Plane) -> Vector3:
    """
    Given two focal points of an ellipse, find the point in 3D space
    where the plane would be tangent to the ellipsoid
    """
    # There is an enormous amount of calculus behind this, but it works
    # I leave the proof as a task to the reader :)

    # Cast rays from the focal points to the plane
    focal_ray_1 = Ray(focal_1, targetPlane.normal)
    focal_ray_2 = Ray(focal_2, targetPlane.normal)
    
    # Find where those rays intersect
    focal_intersect_1 = targetPlane.intersection(focal_ray_1)
    focal_intersect_2 = targetPlane.intersection(focal_ray_2)

    # Find distances from focal points to plane
    focal_dist_1 = (focal_1 - focal_intersect_1).len()
    focal_dist_2 = (focal_2 - focal_intersect_2).len()

    # distance from both focal intersects
    intersect_diff_vec = focal_intersect_2 - focal_intersect_1
    intersect_distance = (intersect_diff_vec).len()

    if focal_dist_1 == focal_dist_2:
        dx = intersect_diff_vec.len() / 2
    elif focal_dist_2 > focal_dist_1:
        dx = intersect_distance * (1 - (focal_dist_2 - math.sqrt(focal_dist_1 * focal_dist_2)) / (focal_dist_2 - focal_dist_1))
    else:
        dx = intersect_distance * (focal_dist_1 - math.sqrt(focal_dist_1 * focal_dist_2)) / (focal_dist_1 - focal_dist_2)
    
    
    return focal_intersect_1 + (intersect_diff_vec.norm() * dx)


def reflection_signature(transmitter_point: Vector3, receiver_point: Vector3, reflection_plane: Plane, rate: float = 44100, max_time: float = 1, speed = 343) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate the reflection signature given the transmitter, receiver, and reflection plane

    The reflection signature is the amount of area reflecting back to the receiver over time
    The return is an two arrays of area values and a float, at a rate of 'rate' samples per second
    The first area is the amount of area reflecting back at that time, the second is the amount 
    of distance traveled to get there
    The float is the approximate distance to the plane to use when calculating intensity

    if distance = 0, that means that the wave did not reflect in that amount of time

    max_time is the amount of time between the first and last reflection output.
    NOTE: the actual reflection signature may be longer than the max_time might suggest, 
    # since the wave takes some amount of time to actually make it to the receiver
    # Therefore length of reflection signature = time till first reflection receiver + max_time

    speed is the speed of the wave, in m/s

    oh yeah all units in meters of course
    """

    receiver_transmitter_d2 = (transmitter_point - receiver_point).len() / 2

    intersect_point = find_ellipsoid_tangent_point(transmitter_point, receiver_point, reflection_plane)

    dist_1 = (transmitter_point - intersect_point).len()
    dist_2 = (receiver_point - intersect_point).len()

    first_dist = dist_1 + dist_2
    first_time = first_dist / speed

    dist_delta = speed / rate # Amount of distance between area calculations

    n_sample = int(rate * (max_time + first_time))
    signature = np.zeros(n_sample)
    distances = np.zeros(n_sample)

    start_point = round(first_time) # Yeah this isn't accurate, it will result in the reflection being mistimed by at most (1 / (2 * rate)) seconds, so for rate = 44100, around 11 microseconds
    index = start_point
    current_distance = first_dist
    old_area = 0

    # We now need to move everything around, since the math works by expanding
    # The ellipsoid from the origin, we need to move everything around so that 
    # They are still at the same relative positions when the ellipsoid is moved there

    # Create a the ellipsoid that will be expanded during sim
    expansion_ellipsoid: Ellipsoid = Ellipsoid.from_focal_points(transmitter_point, receiver_point, first_dist)

    # Calculate approximate distance to the plane
    approx_distance = (expansion_ellipsoid.center - intersect_point).len()

    # Transform all scene elements so they are still in the same relative positions
    rotation, translation = expansion_ellipsoid.move_and_align_with_origin()
    plane_transformed = reflection_plane.transform(rotation, translation)
    intersect_transformed = (intersect_point + translation).apply_matrix(rotation)
#    transmitter_transformed = (transmitter_point + translation).apply_rot_matrix(rotation)
#    receiver_transformed = (receiver_point + translation).apply_rot_matrix(rotation)
    
    # Calculate the coefficient of Lambertian reflectance
    lambert_coeff = abs((transmitter_point - intersect_point).norm().dot(reflection_plane.normal))

    while index < n_sample:
        # Find the intersection size
        ax1, ax2 = expansion_ellipsoid.size_ellipse_intersection(plane_transformed, intersect_transformed)

        # Area of an ellipse
        area = math.pi * ax1 * ax2

        # Calculate ring area, and apply Lambertian reflectance
        signature[index] = lambert_coeff * (area - old_area)

        # Record distance
        distances[index] = current_distance

        index += 1

        old_area = area

        current_distance += dist_delta

        # Resize the ellipsoid for next round

        major_axis_length = current_distance / 2
        minor_axis_length = math.sqrt(math.pow(major_axis_length, 2) - math.pow(receiver_transmitter_d2, 2))
        expansion_ellipsoid.semi_major.set_length(major_axis_length)
        expansion_ellipsoid.semi_minor.set_length(minor_axis_length)
        expansion_ellipsoid.intermediate.set_length(minor_axis_length)
    
    # due to floating point arithmetic error, the first signal element tends to be overestimated
    # Approximate true value by assuming l[0] - l[1] = l[1] - l[2]
    signature[start_point] = 2 * signature[start_point + 1] - signature[start_point + 2]
    
    return (signature, distances, approx_distance)

def convert_to_db_loss(area: float, dist_to_reflection: float, total_distance: float, albedo = 1):
    """
    Given:
    area : Area of the effective reflection
    dist_to_reflection : Distance from transmitter to reflection
    total_distance : Total distance traveled by wave from transmitter to receiver
    albedo = 1 : The reflectivity of the surface (0 = total absorption, 1 = total reflection)

    Calculate the expected db loss over that distance
    """

    # The method employed here is an approximation, as the true solution requires integration
    # We instead remap the innactive area to a sphere of equal surface area around the transmitter
    # The radius of this sphere acts as the "first transmission distance"
    # The "second transmission distance" is the approximate distance from the plane to the receiver
    # Then by calculating the inverse square law over the transmission distance we get our dB loss
    
    # Calculate the amount of area that is actually "transmitting"
    transmitting_area = area * albedo

    # Area of nontransmitting surface
    effective_area = 4 * math.pi * math.pow(dist_to_reflection, 2) - transmitting_area

    # Remap to the area of a sphere
    sphere_radius = math.sqrt(effective_area / (4 * math.pi))

    # Calculate effective distance
    distance = sphere_radius + total_distance - dist_to_reflection

    # Calculate the dB loss over the effective transmission distance
    return 20 * math.log10(distance) + 11



if __name__ == "__main__":
#    p1 = Vector3(-10, .0000001, 0)
#    p2 = Vector3(10, 0, 0)
#    plane = Plane(Vector3(0,-3, 0), Vector3.Z(), Vector3.X())
#    k = find_ellipsoid_tangent_point(p1, p2, plane)
#    print(k)
#    d1 = (p1 - k).len()
#    d2 = (p2 - k).len()
#    ell = Ellipsoid.from_focal_points(p1, p2, d1 + d2)
#    ell.center = Vector3.zero()
#    print(plane)
#    print(ell)
#    exit()
    listener = Vector3(0,2,0)
    transmitter = Vector3(-75,8,0)
    plane = Plane(Vector3(0,0,0), Vector3.Z(), Vector3.X())
    sig, dist, p_dist = reflection_signature(transmitter, listener, plane, max_time=.2)
#    sig = sig * (1 / dist ** 2)
    
    inter_point = find_ellipsoid_tangent_point(listener, transmitter, plane)
    lambert_coeff = abs((transmitter - inter_point).norm().dot(plane.normal))
    f_dist = (transmitter - inter_point).len() + (listener - inter_point).len()
    f_time = f_dist / 343

    theo = np.zeros(len(dist))
    for index, d in enumerate(dist):
        if index == 0:
            continue
        theo[index] = lambert_coeff * math.pi * (math.pow(dist[index] / 2, 2) - math.pow(dist[index - 1] / 2, 2))
    theo[0] = theo[1]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,3)
    ax[0].plot(dist / 343, sig)
    ax[0].plot(dist / 343, theo)
    ax[0].set_title('Reflection Area vs Time')
    ax[1].plot(dist / 343, sig / (dist ** 2))
    ax[1].plot(dist / 343, theo / (dist ** 2))
    ax[1].set_title('ISL Adjusted Reflection vs Time')
    ax[1].plot(dist / 343, sig / (dist ** 2))
    ax[1].plot(dist / 343, theo / (dist ** 2))
    ax[2].set_title('dB loss vs Time')
    ax[2].plot(dist /343, [convert_to_db_loss(s, p_dist, d) for s, d in zip(sig, dist)])
    ax[2].plot(dist /343, [convert_to_db_loss(t, p_dist, d) for t, d in zip(theo, dist)])
    fig.suptitle('Reflection Signature')
    plt.show()



