import json

# class Triangle:
#     n_dots = 3
#     def __init__(self, a, b, c):
#         self.a = a
#         self.b = b
#         self.c = c
#
#         self.p = (a + b + c) / 2
#
#         if self.a + self.b <= self.c or self.a + self.c <= self.b or self.b + self.c <= self.a:
#             raise ValueError("triangle inequality does not hold")
#
#     def area(self):
#         return (self.p * (self.p - self.a) * (self.p - self.b) * (self.p - self.c)) ** 0.5
#
# tr_1 = Triangle(4, 5, 6)
# tr_2 = Triangle(3, 4, 5)
#
# square_1 = tr_1.area()
# square_2 = tr_2.area()

# class Rectangle(Triangle):
#     n_dots: int = 4
#     def __init__(self, a, b):
#         self.a = a
#         self.b = b
#
#     def area(self):
#         return self.a * self.b

class BaseFigure:
    n_dots = None

    def __init__(self):
        self.validate()

    def validate(self):
        raise NotImplementedError

    def area(self):
        raise NotImplementedError

class Triangle(BaseFigure):
    n_dots = 3
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        self.p = (a + b + c) / 2
        super().__init__()

    def validate(self):
        if self.a + self.b <= self.c or self.a + self.c <= self.b or self.b + self.c <= self.a:
            raise ValueError("triangle inequality does not hold")
        return self.a, self.b, self.c

    def area(self):
        return (self.p * (self.p - self.a) * (self.p - self.b) * (self.p - self.c)) ** 0.5


class Rectangle(BaseFigure):
    n_dots = 4
    def __init__(self, a, b):
        self.a = a
        self.b = b
        super().__init__()

    def validate(self):
        return self.a, self.b

    def area(self):
        return self.a * self.b

class Circle(BaseFigure):
    n_dots = float('inf')

    def __init__(self, r):
        self.r = r
        super().__init__()

    def validate(self):
        return

    def area(self):
        return 3.14 * self.r**2

class Vector:
    coords = []

    def __init__(self, coords):
        self.coords = coords

    def validate(self, other):
        if not isinstance(other, Vector):
            return NotImplemented

        if len(self.coords) != len(other.coords):
            raise ValueError(f"left and right lengths differ: {len(self.coords)} != {len(other.coords)}")

    def __add__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented

        self.validate(other)

        new_coords = [x + y for x, y in zip(self.coords, other.coords)]

        return Vector(new_coords)

    def __mul__(self, other):
        if isinstance(other, int):
            return Vector([x * other for x in self.coords])
        self.validate(other)

        return sum(x * y for x, y in zip(self.coords, other.coords))

    def __eq__(self, other):
        if not isinstance(other, Vector):
            return False

        return self.coords == other.coords

    def __abs__(self):
        return sum(x**2 for x in self.coords)**0.5

    def __repr__(self):
        return f"{self.coords}"


class ParsesCookies:
    def cookies(self):
        return self.request['cookies']

    def is_authed(self):
        return 'auth_key' in self.request['cookies']

class ParsesBody:
    def body(self):
        return self.request['body']

class ParsesHeaders:
    def headers(self):
        return self.request['headers']

    def need_json(self):
        return self.request['headers'].get('content-type', '') == 'application/json'

class JsonHandler(ParsesBody, ParsesHeaders):
    def __init__(self, request):
        self.request = request

    def process(self):
        if not self.need_json():
            return None

        try:
            parsed = json.loads(self.body())
            if isinstance(parsed, dict):
                return len(parsed)
            return None
        except Exception:
            return None

class SecureTextHandler(ParsesBody, ParsesCookies):
    def __init__(self, request):
        self.request = request

    def process(self):
        if not self.is_authed():
            return None

        try:
            return len(self.body())
        except Exception:
            return None
