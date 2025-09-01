# Swin
### Relative Position Bias
Example of Swin's relative position bias

Assume we have a window size of `M = 3` -> `(3, 3)`. This means we have 9 patches in a window shown below.

<img width="200" height="200" alt="local_window" src="https://github.com/user-attachments/assets/41a27539-1541-4c39-b1f0-40369c5a6e4f" />

Each patch can have a relative position to another patch in the set `{-M+1, …, M-1}` -> `{-2, -1, 0, 1, 2}`

E.g., the position from `patch 1 -> patch 2` = `[2, 2]` and `patch 2 -> patch 1` = `[-2, -2]` `[row, col]`

If we sample every patch’s relative position to all the other patches, we can form a bias matrix of size: `[2M-1, 2M-1]` -> `[5, 5]` with the patches _centered_ in the middle. This matrix is shown below. NOTE: in code the top_left of the bias matrix will start at index `[0, 0]` but for intuition the following matrix center will be `[0, 0]`

<img width="300" height="300" alt="bias_matrix" src="https://github.com/user-attachments/assets/66e247c3-566b-419b-a0d8-8982f1d6108d" />

The way I think about this bias matrix is that `patch 1` in the `3x3` window is responsible for the bottom right `3x3` region of the `5x5` matrix; likewise, `patch 6` is responsible for the middle left `3x3` region of the `5x5` matrix as illustrated below. Note that the green shading is showing the overlap.

<img width="600" height="337" alt="window_and_bias_mat" src="https://github.com/user-attachments/assets/b1703207-2346-4cf8-b227-b954136c46b5" />

Once we have this bias matrix, we need to add it to the attention scores at each head. The `3x3` matrix shows the patches such that they are spatially correct (they look like an image), but when we compute the attention scores we do it on the flattened patches; i.e, `3x3` -> `9 patches` so to compute attention we'll perform matrix multiplication between `(9, head_dim)` and `(head_dim, 9)` to get a `9, 9` attention matrix, where `9 = number of patches`. Since the relative position bias is added after the attention scores are computed (just before the softmax), we'll need to transform this `5x5` bias matrix into a `9x9` in order to add these relative positions to the attention matrix.

The relative position biawe can create a matrix of $` M^2 x M^2 `$



