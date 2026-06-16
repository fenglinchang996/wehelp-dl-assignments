from math import pi, pow, sqrt, fabs
from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int

    def __sub__(self, other: Point) -> Vector:
        return Vector(self.x - other.x, self.y - other.y)


@dataclass
class Vector:
    i: int
    j: int

    def dot(self, other: Vector) -> float:
        return self.i * other.i + self.j * other.j

    def cross(self, other: Vector) -> float:
        return self.i * other.j - other.i * self.j


class Line:
    def __init__(self, a: Point, b: Point) -> None:
        self.a = a
        self.b = b

    @property
    def length(self) -> float:
        return sqrt(pow(self.a.x - self.b.x, 2) + pow(self.a.y - self.b.y, 2))

    @property
    def vector(self) -> Vector:
        return self.b - self.a

    def isParallel(self, other: Line) -> bool:
        return self.vector.cross(other.vector) == 0

    def isPerpendicular(self, other: Line) -> bool:
        return self.vector.dot(other.vector) == 0


class Polygon:
    def __init__(self, a: Point, b: Point, c: Point, d: Point) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.ab = Line(a, b)
        self.bc = Line(b, c)
        self.cd = Line(c, d)
        self.da = Line(d, a)

    @property
    def perimeter(self) -> float:
        return self.ab.length + self.bc.length + self.cd.length + self.da.length


class Circle:
    def __init__(self, c: Point, r: float) -> None:
        self.c = c
        self.r = r

    @property
    def area(self) -> float:
        return pi * pow(self.r, 2)

    def isIntersect(self, another: Circle) -> bool:
        center_line = Line(self.c, another.c)
        center_distance = center_line.length
        return fabs(self.r - another.r) <= center_distance <= self.r + another.r


class Enemy:
    def __init__(self, position: Point, step: Vector, life: int) -> None:
        self.life = life
        self.position = position
        self.step = step

    @property
    def isDead(self) -> bool:
        return self.life == 0

    def move(self) -> None:
        self.position.x = self.position.x + self.step.i
        self.position.y = self.position.y + self.step.j

    def takeDamage(self, damage: int) -> None:
        remainingLife = self.life - damage
        self.life = remainingLife if remainingLife > 0 else 0


class Tower:
    def __init__(self, position: Point, attackPoint: int, attackRange: int) -> None:
        self.position = position
        self.attackPoint = attackPoint
        self.attackRange = attackRange

    def isInAttackRange(self, position: Point) -> bool:
        return (
            sqrt(
                pow(self.position.x - position.x, 2)
                + pow(self.position.y - position.y, 2)
            )
            <= self.attackRange
        )


def main():
    # Task 1
    print("Task 1:")
    lineA = Line(Point(-6, 1), Point(2, 4))
    lineB = Line(Point(-6, -1), Point(2, 2))
    lineC = Line(Point(-4, -4), Point(-1, 6))
    polygonA = Polygon(Point(2, 0), Point(5, -1), Point(4, -4), Point(-1, -2))
    circleA = Circle(Point(6, 3), 2)
    circleB = Circle(Point(8, 1), 1)
    print(f"Are Line A and Line B parallel? {lineA.isParallel(lineB)}")
    print(f"Are Line C and Line A perpendicular? {lineC.isPerpendicular(lineA)}")
    print(f"The area of Circle A is {circleA.area}")
    print(f"Do Circle A and Circle B intersect? {circleA.isIntersect(circleB)}")
    print(f"The perimeter of Polygon A is {polygonA.perimeter}")

    # Task 2
    print("Task 2:")
    ENEMY_LIFE = 10
    e1 = Enemy(Point(-10, 2), Vector(2, -1), ENEMY_LIFE)
    e2 = Enemy(Point(-8, 0), Vector(3, 1), ENEMY_LIFE)
    e3 = Enemy(Point(-9, -1), Vector(3, 0), ENEMY_LIFE)
    enemies = [e1, e2, e3]
    BASIC_TOWER_STATS = {"attackPoint": 1, "attackRange": 2}
    ADVANCED_TOWER_STATS = {"attackPoint": 2, "attackRange": 4}
    t1 = Tower(
        Point(-3, 2), BASIC_TOWER_STATS["attackPoint"], BASIC_TOWER_STATS["attackRange"]
    )
    t2 = Tower(
        Point(-1, -2),
        BASIC_TOWER_STATS["attackPoint"],
        BASIC_TOWER_STATS["attackRange"],
    )
    t3 = Tower(
        Point(4, 2), BASIC_TOWER_STATS["attackPoint"], BASIC_TOWER_STATS["attackRange"]
    )
    t4 = Tower(
        Point(7, 0), BASIC_TOWER_STATS["attackPoint"], BASIC_TOWER_STATS["attackRange"]
    )
    a1 = Tower(
        Point(1, 1),
        ADVANCED_TOWER_STATS["attackPoint"],
        ADVANCED_TOWER_STATS["attackRange"],
    )
    a2 = Tower(
        Point(4, -3),
        ADVANCED_TOWER_STATS["attackPoint"],
        ADVANCED_TOWER_STATS["attackRange"],
    )
    towers = [t1, t2, t3, t4, a1, a2]

    for _ in range(10):
        for enemy in enemies:
            if not enemy.isDead:
                enemy.move()
        for tower in towers:
            for enemy in enemies:
                if not enemy.isDead and tower.isInAttackRange(enemy.position):
                    enemy.takeDamage(tower.attackPoint)

    print(f"E1: position {e1.position.x, e1.position.y} and life {e1.life}")
    print(f"E2: position {e2.position.x, e2.position.y} and life {e2.life}")
    print(f"E3: position {e3.position.x, e3.position.y} and life {e3.life}")


if __name__ == "__main__":
    main()
