xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
