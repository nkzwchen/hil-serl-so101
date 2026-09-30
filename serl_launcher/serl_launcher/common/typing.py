from typing import Any, Callable, Dict, Sequence, Union

import flax
import jax.numpy as jnp
import numpy as np

# 注：原代码 `import tensorflow as tf` 仅为下面 Array 的类型标注服务。
# 为了甩掉 tensorflow（它和 jax/tfp 的版本组合是 HIL-SERL 最大的依赖冲突源），
# 这里去掉 tf.Tensor。运行时行为完全不变（类型标注不参与运行）。
PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]
Array = Union[np.ndarray, jnp.ndarray]
Data = Union[Array, Dict[str, "Data"]]
Batch = Dict[str, Data]
# A method to be passed into TrainState.__call__
ModuleMethod = Union[str, Callable, None]
