from transonic import boost


@boost
class Class:
    @boost
    def func(self, a: int):
        return 2 * a
