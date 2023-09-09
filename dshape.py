import sympy as sp
import numpy as np
  
def compute(Points): 
  a, b, c = sp.symbols('a b c')
  N = sp.Matrix([
    (1/8)*(1-a)*(1-b)*(1-c),
    (1/8)*(1+a)*(1-b)*(1-c),
    (1/8)*(1+a)*(1+b)*(1-c),
    (1/8)*(1-a)*(1+b)*(1-c),
    (1/8)*(1-a)*(1-b)*(1+c),
    (1/8)*(1+a)*(1-b)*(1+c),
    (1/8)*(1+a)*(1+b)*(1+c),
    (1/8)*(1-a)*(1+b)*(1+c)
  ])
  Ndiff = sp.Matrix([
    [N[i].diff(a) for i in range(8)],
    [N[i].diff(b) for i in range(8)],
    [N[i].diff(c) for i in range(8)]
  ])

  J = Ndiff * Points
  Ji = J.inv()
  B = sp.Matrix([])
  for i in range(8):
    Edn = Ndiff[:,i]
    Edc = Ji * Edn
    Bi = sp.Matrix([
      [Edc[0],0,0],
      [0,Edc[1],0],
      [0,0,Edc[2]],
      [Edc[1],Edc[0],0],
      [0,Edc[2],Edc[1]],
      [Edc[2],0,Edc[0]]
    ])
    for j in range(Bi.shape[1]):
      B = B.row_join(Bi[:, j])

  value = {a: 0.57735, b: 0.57735, c: 0.57735}
  Bn = B.subs(value)
  Jn = J.subs(value)
  Sc = Bn.transpose() * Bn * Jn.det()

  for idx in range(7):
    Sc += Sc
    
  return np.array(Sc)


def strength(D, Points): 
  a, b, c = sp.symbols('a b c')
  N = sp.Matrix([
    (1/8)*(1-a)*(1-b)*(1-c),
    (1/8)*(1+a)*(1-b)*(1-c),
    (1/8)*(1+a)*(1+b)*(1-c),
    (1/8)*(1-a)*(1+b)*(1-c),
    (1/8)*(1-a)*(1-b)*(1+c),
    (1/8)*(1+a)*(1-b)*(1+c),
    (1/8)*(1+a)*(1+b)*(1+c),
    (1/8)*(1-a)*(1+b)*(1+c)
  ])
  Ndiff = sp.Matrix([
    [N[i].diff(a) for i in range(8)],
    [N[i].diff(b) for i in range(8)],
    [N[i].diff(c) for i in range(8)]
  ])

  J = Ndiff * Points
  Ji = J.inv()
  B = sp.Matrix([])
  for i in range(8):
    Edn = Ndiff[:,i]
    Edc = Ji * Edn
    Bi = sp.Matrix([
      [Edc[0],0,0],
      [0,Edc[1],0],
      [0,0,Edc[2]],
      [Edc[1],Edc[0],0],
      [0,Edc[2],Edc[1]],
      [Edc[2],0,Edc[0]]
    ])
    for j in range(Bi.shape[1]):
      B = B.row_join(Bi[:, j])

  value = {a: 0.57735, b: 0.57735, c: 0.57735}
  Bn = np.array(B.subs(value), dtype=np.double)
  Jn = np.array(J.subs(value), dtype=np.double)

  detJ = np.linalg.det(Jn)
  invBt = np.linalg.pinv(Bn.T)
  M = np.dot(np.dot(invBt, D), Bn.T) * detJ

  for idx in range(7):
    M += M
    
  return M * (1/detJ)