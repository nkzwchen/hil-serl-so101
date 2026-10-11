"""Weights-only initialization into a new run. Never modifies the source checkpoint."""
import hashlib,json
from pathlib import Path

def load_weights(agent, checkpoint, spec, out):
    import jax
    from flax import serialization
    path=Path(checkpoint).resolve()
    if path.parent==Path(out).resolve():
        raise ValueError('Warm start requires a separate output directory')
    saved=json.loads(Path(str(path)+'.json').read_text())
    old=saved['spec']
    for key,value in spec.items():
        if key not in ('env_hz','action_scale_rad') and old.get(key)!=value:
            raise ValueError(f'Warm-start incompatible specification: {key}')
    if old.get('allow_provisional'):
        raise ValueError('Cannot migrate provisional weights')
    payload=path.read_bytes()
    restored=serialization.from_bytes(agent.state,payload)
    for name in ('params','target_params'):
        a=getattr(agent.state,name);b=getattr(restored,name)
        if jax.tree_util.tree_structure(a)!=jax.tree_util.tree_structure(b):
            raise ValueError('Weight tree mismatch')
        import numpy as np
        for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
            if x.shape!=y.shape or not np.isfinite(np.asarray(y)).all():
                raise ValueError('Invalid migrated weights')
    # Retain new optimizer slots, RNG and update counter; copy both network trees exactly.
    agent=agent.replace(state=agent.state.replace(params=restored.params,target_params=restored.target_params))
    return agent,dict(source_checkpoint=str(path),sha256=hashlib.sha256(payload).hexdigest(),
        source_step=saved['step'],source_spec=old,target_spec=spec,
        optimizer='reset',replay='new',weights='exact_params_and_target_params',
        action_scale_warning='Same normalized output has different physical magnitude; offline adaptation required before robot deployment')
