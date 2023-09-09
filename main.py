import mesh
import dshape as ds
import numpy as np
import matplotlib.pyplot as plt
import processed as ps
import time
start_time = time.time()

train, valid, test = ps.load()

train.reset_index(inplace=True)
valid.reset_index(inplace=True)
test.reset_index(inplace=True)

# Number of input and target
n_train = len(train)
n_valid = len(valid)
n_test = len(test)

print(f"training:{n_train}  validation:{n_valid} test:{n_test}")

# number of prisms in any direction
dims = (2,2,2)

# Creating meshing of prisms
prism, vertices = mesh.create(dims, limits=[(0,100),(0,100),(0,500)])
#mesh.plot(prism, vertices, "Mesh")

Ndof = np.prod(vertices.shape)

# force vectors, input
Force = np.zeros((Ndof, n_train))
Force_valid = np.zeros((Ndof, n_valid))
Force_test = np.zeros((Ndof, n_test))

# displacement vectors, target
Displacement = np.zeros((Ndof, n_train))
Displacement_valid = np.zeros((Ndof, n_valid))
Displacement_test = np.zeros((Ndof, n_test))

# Apply forces, displacements
effect_idx = []
prism_idx = 0
for i in range(dims[0]):
  for j in range(dims[1]):
    for k in range(dims[2]):
      row = prism[prism_idx]
      # plate yz in i = 0
      if (i == 0) and (j == 0) and (k == dims[2]-1):
        result_list = []
        effect = []
        for m, r in enumerate(row):
          # [0,3,4,7]
          if np.any(np.array([7]) == m):
            values = np.array(range(r*3, (3*r+1)))
            result_list.append(values)
            effect.append(r)
        indices = np.array(result_list ,dtype=np.int64).flatten()
        Force[indices, :] = np.tile([t for t in train['AirPollutionLevel']], (len(indices), 1))
        Force_valid[indices, :] = np.tile([t for t in valid['AirPollutionLevel']], (len(indices), 1))
        Force_test[indices, :] = np.tile([t for t in test['AirPollutionLevel']], (len(indices), 1))

        indices = np.array(effect ,dtype=np.int64).flatten()
        effect_idx.append(indices)
      # plate yz in i = end
      if (i == dims[2]-1) and (j == 0) and (k == dims[2]-1):
        result_list = []
        effect = []
        for m, r in enumerate(row):
          # [1,2,5,6]
          if np.any(np.array([6]) == m):
            values = np.array(range(r*3, (3*r+1)))
            result_list.append(values)
            effect.append(r)
        indices = np.array(result_list ,dtype=np.int64).flatten()
        Displacement[indices, :] = np.tile([t.timestamp() for t in train['DatetimeBegin']], (len(indices), 1))
        Displacement_valid[indices, :] = np.tile([t.timestamp() for t in valid['DatetimeBegin']], (len(indices), 1))
        Displacement_test[indices, :] = np.tile([t.timestamp() for t in test['DatetimeBegin']], (len(indices), 1))

        indices = np.array(effect ,dtype=np.int64).flatten()
        effect_idx.append(indices)

      prism_idx += 1

effect_idx = np.concatenate(effect_idx)
#mesh.plot(prism, vertices, "Effect Position", effect_idx)

# Boundary conditions
unsupported_idx = []
supported_idx = []
prism_idx = 0
for i in range(dims[0]):
  for j in range(dims[1]):
    for k in range(dims[2]):
      row = prism[prism_idx]
      if (k == 0):
        unsupport = []
        support = []
        for m, r in enumerate(row):
          if m > 3:
            unsupport.append(r)
          else:
            support.append(r)
        indices = np.array(unsupport ,dtype=np.int64).flatten()
        unsupported_idx.append(indices)

        indices = np.array(support ,dtype=np.int64).flatten()
        supported_idx.append(indices)

        Displacement[indices, :] = np.zeros((len(indices), n_train))
        Displacement_valid[indices, :] = np.zeros((len(indices), n_valid))
        Displacement_test[indices, :] = np.zeros((len(indices), n_test))
      else:
        indices = np.array(row,dtype=np.int64).flatten()
        unsupported_idx.append(indices)
      prism_idx += 1

unsupported_idx = np.concatenate(unsupported_idx)
supported_idx = np.concatenate(supported_idx)

#mesh.plot(prism, vertices, "Support Position", supported_idx)

St = np.zeros((Ndof, Ndof), dtype=np.double)

for idx, row in enumerate(prism):
  print(f"pseudo-stiffness prism ({idx+1}/{len(prism)})")

  points = vertices[row]
  indices = np.array([[range(r*3,3*r+3)] for r in row],dtype=np.int64).flatten()

  Sc = ds.compute(points)

  rows, cols = np.ix_(indices, indices)
  St[rows, cols] = Sc

eta = 42
epsilon = 1e-8

prev_cost = 2
cost = 1
epoch = 1

Xb = np.dot(St, Displacement)

P = Force.T
K = Xb.T

M = 0.5 * K

# omega
W = np.sqrt(2)

Strength = np.ones((Ndof,Ndof))
analysis = []

print("Compute the pseudo-strength")
while np.abs(prev_cost - cost) > epsilon:
  F = np.dot(K ,Strength)
  R = P - F

  invM = np.linalg.pinv(M)

  A = - eta * (1/(W**2)) * np.dot(invM, R) * np.cos(W)
  Strength = Strength - A * np.cos(W)

  prev_cost = cost
  cost = np.average(np.dot(R.T, R) / R.shape[1])

  Sv = np.dot(Strength.T, St)
  Fv = np.dot(Sv, Displacement_valid)
  Rv = np.abs(Force_valid-Fv)
  cost_avg = np.average(np.dot(Rv.T, Rv) / Rv.shape[1])

  Sv = Strength.T
  cost_w = []
  for idx, row in enumerate(prism):
    indices = np.array([[range(r*3,3*r+3)] for r in row],dtype=np.int64).flatten()
    rows, cols = np.ix_(indices, indices)
    Rv = Sv[rows, cols]
    cost_v = np.diag(np.dot(Rv.T, Rv))
    cost_w.append(cost_v / Rv.shape[1])

  analysis.append([epoch, cost, cost_avg, cost_w])

  print(f"\tepoch {epoch}: train cost ({cost}), valid cost ({cost_avg})")
  epoch += 1

Strength = Strength.T
Stiffness = np.dot(Strength, St)
print("training is complate")
print(f"epoch {epoch}: cost ({cost})")

F = np.dot(Stiffness, Displacement_test)
R = np.abs(Force_test-F)
cost_avg = np.average(np.dot(R.T, R) / R.shape[1])
print(f"cost with test data ({cost_avg})")

end_time = time.time()
total_time = end_time - start_time
print(f"Time running: {total_time} s")


X1 = np.array([x for x,_,_,_ in analysis])
Y1 = np.array([y for _,y,_,_ in analysis])
Y2 = np.array([y for _,_,y,_ in analysis])
Y3 = np.array([y for _,_,_,y in analysis])

plt.figure(figsize=(10,6))
plt.plot(X1, Y1, label='Train')
plt.plot(X1, Y2, label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
for idx, row in enumerate(prism):
  Y5 = Y3[:,idx]
  Y6 = np.mean(Y5, axis=1)
  plt.plot(X1, Y6, label='layer-{}'.format(idx))
plt.xlabel('Epochs')
plt.ylabel('Weight')
plt.legend()
plt.show()

Materials = []
print("Compute the material")
for idx, row in enumerate(prism):
  print(f"strength prism ({idx+1}/{len(prism)})")

  points = vertices[row]
  indices = np.array([[range(r*3,3*r+3)] for r in row] ,dtype=np.int64).flatten()

  rows, cols = np.ix_(indices, indices)
  D = Strength[rows, cols]

  M = ds.strength(D, points)
  Materials.append(M)







prev_cost = 2
cost = 1
epoch = 1

P = Force.T
K = Displacement.T

M = 0.5 * K

# omega
W = np.sqrt(2)

Stiffness = np.ones((Ndof,Ndof))
analysis = []

print("Compute the stiffness")
while np.abs(prev_cost - cost) > epsilon:
  F = np.dot(K ,Stiffness)
  R = P - F

  invM = np.linalg.pinv(M)

  A = - eta * (1/(W**2)) * np.dot(invM, R) * np.cos(W)
  Stiffness = Stiffness - A * np.cos(W)

  prev_cost = cost
  cost = np.average(np.dot(R.T, R) / R.shape[1])

  Fv = np.dot(Stiffness.T, Displacement_valid)
  Rv = np.abs(Force_valid-Fv)
  cost_avg = np.average(np.dot(Rv.T, Rv) / Rv.shape[1])

  Sv = Stiffness.T
  cost_w = []
  for idx, row in enumerate(prism):
    indices = np.array([[range(r*3,3*r+3)] for r in row],dtype=np.int64).flatten()
    rows, cols = np.ix_(indices, indices)
    Rv = Sv[rows, cols]
    cost_v = np.diag(np.dot(Rv.T, Rv))
    cost_w.append(cost_v / Rv.shape[1])

  analysis.append([epoch, cost, cost_avg, cost_w])

  print(f"\tepoch {epoch}: train cost ({cost}), valid cost ({cost_avg})")
  epoch += 1

Stiffness = Stiffness.T
print("training is complate")
print(f"epoch {epoch}: cost ({cost})")

F = np.dot(Stiffness, Displacement_test)
R = np.abs(Force_test-F)
cost_avg = np.average(np.dot(R.T, R) / R.shape[1])
print(f"cost with test data ({cost_avg})")

X1 = np.array([x for x,_,_,_ in analysis])
Y1 = np.array([y for _,y,_,_ in analysis])
Y2 = np.array([y for _,_,y,_ in analysis])
Y3 = np.array([y for _,_,_,y in analysis])

plt.figure(figsize=(10,6))
plt.plot(X1, Y1, label='Train-Without-Sc')
plt.plot(X1, Y2, label='Validation-Without-Sc')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
for idx, row in enumerate(prism):
  Y5 = Y3[:,idx]
  Y6 = np.mean(Y5, axis=1)
  plt.plot(X1, Y6, label='layer-{}'.format(idx))
plt.xlabel('Epochs')
plt.ylabel('Weight')
plt.legend()
plt.show()