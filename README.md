# GB-DivRW Core Code

This repository provides the core computational modules of **GB-DivRW** for research reference.

GB-DivRW is a division-profile-driven granular-ball random walk method for unsupervised outlier detection. The public code exposes three mechanisms used by the model:

1. attribute-wise division-profile construction;
2. uniform-restart saliency propagation on a granular-ball dissimilarity graph; and
3. local--global residual fusion for intra-ball object ranking.

## Installation

```bash
git clone https://github.com/nqy-0219/GB-DivRW.git
cd GB-DivRW
python -m pip install -e .
```

## Core Modules

```python
from gbdivrw import ball_saliency, division_profile, intra_ball_residual

# x is a normalized encoded feature matrix and r contains one attribute radius per column.
q = division_profile(x, r)

# Ball centres, profile radii, and ball sizes are supplied by an external granular-ball procedure.
b = ball_saliency(profile_centres, profile_radii, ball_sizes)

# Object-level residual evidence can then be fused within the assigned ball.
residual = intra_ball_residual(attribute_residual, global_profile_residual, local_profile_residual)
```

All returned values are normalized so that a larger value represents stronger separation or anomaly evidence.

## Citation

Please cite the associated manuscript when using this software:

```text
Mijia Li, Ran Li, Hongchang Chen, Shuxin Liu, and Chen Wang.
Division-profile-driven granular-ball random walk with intra-ball object ranking
for unsupervised outlier detection.
```

Citation metadata is available in [CITATION.cff](CITATION.cff).

## License

This repository is released under the [MIT License](LICENSE).
