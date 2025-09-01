# Swin
### Relative Position Bias
Example of Swin's relative position bias

Assume we have a window size of `M = 3` -> `(3, 3)`. This means we have 9 patches in a window shown below.

<img width="200" height="200" alt="local_window" src="https://github.com/user-attachments/assets/41a27539-1541-4c39-b1f0-40369c5a6e4f" />

Each patch can have a relative position to another patch in the set `{-M+1, …, M-1}` -> `{-2, -1, 0, 1, 2}`

E.g., the position from `patch 1 -> patch 2` = `[2, 2]` and `patch 2 -> patch 1` = `[-2, -2]` `[row, col]`


