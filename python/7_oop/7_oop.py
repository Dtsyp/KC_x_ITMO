class Triangle:
    n_dots = 3
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

        self.p = (a + b + c) / 2

        if self.a + self.b <= self.c or self.a + self.c <= self.b or self.b + self.c <= self.a:
            raise ValueError("triangle inequality does not hold")

    def area(self):
        return (self.p * (self.p - self.a) * (self.p - self.b) * (self.p - self.c)) ** 0.5

tr_1 = Triangle(4, 5, 6)
tr_2 = Triangle(3, 4, 5)

square_1 = tr_1.area()
square_2 = tr_2.area()