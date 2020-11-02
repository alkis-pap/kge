from contextlib import contextmanager
import time
import ast
import inspect

import numpy as np


def int_dtype_for(max_value):
    types = [np.int8, np.int16, np.int32, np.int64]
    for t in types:
        if np.iinfo(t).max >= max_value:
            return t
    raise ValueError('no int is enough')


def make_object(module, expression):
    root = ast.parse(expression)
    if len(root.body) != 1 or not isinstance(root.body[0].value, ast.Call):
        raise ValueError("Invalid object instantiation: %s" % (expression))
    call_expr = root.body[0].value
    class_name = call_expr.func.id
    cls = get_class(module, class_name)
    args = [ast.literal_eval(arg) for arg in call_expr.args]
    kwargs = {arg.arg: ast.literal_eval(arg.value) for arg in call_expr.keywords}
    return cls(*args, **kwargs)


def get_class(module, class_name):
    cls = getattr(module, class_name, None)
    if not inspect.isclass(cls):
        raise ValueError("No class named %s in module %s" % (class_name, module.__name__))
    return cls


@contextmanager
def timeit(message):
    print("Started " + message)
    t0 = time.time()
    try:
        yield
    except:
        print("%s raised an exception after %.3f sec." % (message, (time.time() - t0)))
        raise
    else:
        print("%s done in %.3f sec." % (message, (time.time() - t0)))
