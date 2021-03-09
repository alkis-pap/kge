from contextlib import contextmanager
import time
import ast
import inspect
import functools
from dataclasses import dataclass

import numpy as np

from torch import nn


class DataclassModule(type):
    def __new__(cls, clsname, bases, clsdict):
        cls = type.__new__(cls, clsname, bases, clsdict)
        cls = dataclass(eq=False)(cls)
        old_init = cls.__init__
        @functools.wraps(old_init)
        def new_init(self, *args, **kwargs):
            super(Module, self).__init__()
            return old_init(self, *args, **kwargs)
        cls.__init__ = new_init
        return cls


class Module(nn.Module, metaclass=DataclassModule):
    def init(self, graph, device=None):
        return self

    def set_params(self, **params):
        for key, val in params.items():
            keys = key.split('__', 1)
            if len(keys) > 1:
                getattr(self, keys[0]).set_params(**{keys[1] : val})
            setattr(self, key, val)


def strip_whitespace(s):
    return ' '.join(s.split())


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


def no_op(*_args, **_kwargs):
    pass
