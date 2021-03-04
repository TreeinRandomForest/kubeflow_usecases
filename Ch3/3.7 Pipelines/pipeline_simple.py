
import kfp
from kfp.components import func_to_container_op
import kfp_tekton
BASE_IMAGE = 'docker.io/pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime'

def gen_data() -> float:
    import numpy as np

    return np.random.uniform()

gen_data_op = func_to_container_op(gen_data, base_image=BASE_IMAGE)

def f1(x:float) -> float:
    return x**2

f1_op = func_to_container_op(f1, base_image=BASE_IMAGE)

def f2(x:float) -> float:
    import numpy as np

    return np.exp(x)

f2_op = func_to_container_op(f2, base_image=BASE_IMAGE)

def g(x:float,y:float) -> float:
    import numpy as np

    return x + y
    
g_op = func_to_container_op(g, base_image=BASE_IMAGE)

@kfp.dsl.pipeline(
    name='Full pipeline'
)
def run_all():
    x = gen_data_op()
    
    y = f1_op(x.output)
    z = f2_op(x.output)

    h = g_op(y.output, z.output)
    
#kfp.compiler.Compiler().compile(run_all, 'run_all.zip')
from kfp_tekton.compiler import TektonCompiler
TektonCompiler().compile(run_all, 'run_all_simple.yaml')

'''
if __name__=='__main__':
    x = gen_data(SIZE)

    y = f1(x)
    z = f2(x)

    h = g(y,z)
    print(h)
'''
