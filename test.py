import gc

class foo:
    pass

a = foo()
for ob in gc.get_objects():
    if isinstance(ob, foo):
        print(id(ob))