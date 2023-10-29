import math
import numpy as np
def PTSconvert2str(points):
  assert isinstance(points, np.ndarray) and len(points.shape) == 2, 'The points is not right : {}'.format(points)
  assert points.shape[0] == 2 or points.shape[0] == 3, 'The shape of points is not right : {}'.format(points.shape)
  string = ''
  num_pts = points.shape[1]
  for i in range(num_pts):
    ok = False
    if points.shape[0] == 3 and bool(points[2, i]) == True:
      ok = True
    elif points.shape[0] == 2:
      ok = True

    if ok:
      string = string + '{:02d} {:.2f} {:.2f} True\n'.format(i+1, points[0, i], points[1, i])
  string = string[:-1]
  return string

def PTSconvert2box(points, expand_ratio=None):
  assert isinstance(points, np.ndarray) and len(points.shape) == 2, 'The points is not right : {}'.format(points)
  assert points.shape[0] == 2 or points.shape[0] == 3, 'The shape of points is not right : {}'.format(points.shape)
  if points.shape[0] == 3:
    points = points[:2, points[-1,:].astype('bool') ]
  elif points.shape[0] == 2:
    points = points[:2, :]
  else:
    raise Exception('The shape of points is not right : {}'.format(points.shape))
  assert points.shape[1] >= 2, 'To get the box of points, there should be at least 2 vs {}'.format(points.shape)
  box = np.array([ points[0,:].min(), points[1,:].min(), points[0,:].max(), points[1,:].max() ])
  W = box[2] - box[0]
  H = box[3] - box[1]
  assert W > 0 and H > 0, 'The size of box should be greater than 0 vs {}'.format(box)
  if expand_ratio is not None:
    box[0] = int( math.floor(box[0] - W * expand_ratio) )
    box[1] = int( math.floor(box[1] - H * expand_ratio) )
    box[2] = int( math.ceil(box[2] + W * expand_ratio) )
    box[3] = int( math.ceil(box[3] + H * expand_ratio) )
  return box