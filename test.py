import gc

class foo:
    pass

a = foo()
b = a
print(gc.get_referrers(a)[0])