from sympy.polys.orderings import monomial_key
from sympy.polys.monomials import itermonomials
import sympy as sp
import numpy as np


# Define symbols
x1, x2, x3 = sp.symbols('x1 x2 x3', real=True)
const = sp.symbols('const', real=True) # represented by 1

# Original variables including trigonometric functions
variables = [sp.sin(x3), sp.cos(x3), x3, x2, x1]

deg = 1
monomials = sorted(itermonomials(variables, deg, 1), key=monomial_key('grlex', variables))
basis = sp.lambdify((x1,x2,x3),monomials,"numpy")
monomials.append(const)

print("Monomials:", monomials)

# Substitute trigonometric functions with auxiliary symbols
cos_x3_aux = sp.Symbol('cos_x3_aux', real=True)  # Auxiliary variable for cos(x3)
sin_x3_aux = sp.Symbol('sin_x3_aux', real=True)  # Auxiliary variable for sin(x3)
monomials = [monom.subs({sp.cos(x3): cos_x3_aux, sp.sin(x3): sin_x3_aux}) for monom in monomials]

# Variables after substitution
variables_aux = [sin_x3_aux, cos_x3_aux, x3, x2, x1]

A = sp.randMatrix(len(monomials), len(monomials), min=-10, max=10)
B1 = sp.randMatrix(len(monomials), len(monomials), min=-10, max=10)
B2 = sp.randMatrix(len(monomials), len(monomials), min=-10, max=10)

u1 = sp.symbols('u1', real=True) # Control input 1
u2 = sp.symbols('u2', real=True) # Control input 2
aug_deg = 2

z_monomials = sorted(itermonomials(monomials[:-1], aug_deg, 1), key=monomial_key('grlex', variables_aux)) # monomials of degree 1 to deg
z_monomials.insert(len(variables),const)

z_dot = A*sp.Matrix(monomials) + B1*sp.Matrix(monomials)*u1 + B2*sp.Matrix(monomials)*u2 # Learnt from EDMD or derived by hand
z_monomials_dot = sp.expand(sp.Matrix(z_monomials).jacobian(monomials)*z_dot)
AplusB_augmented = sp.Matrix([[z_monomials_dot[i].coeff(m) for m in z_monomials] for i in range(len(z_monomials))]).subs({monomials[i]: 0 for i in range(len(monomials))})
# Monomial augmented dynamics: d/dt (z_monomials) = A_augmented*z_monomials

A = np.array(AplusB_augmented.subs({u1: 0, u2: 0})).astype(np.float64)
B1 = np.array(AplusB_augmented.subs({u1: 1, u2: 0})).astype(np.float64) - A
B2 = np.array(AplusB_augmented.subs({u1: 0, u2: 1})).astype(np.float64) - A

# Finally, update z_monomials to replace auxiliary variables with actual expressions cos(x3) and sin(x3)
z_monomials = [monom.subs({cos_x3_aux: sp.cos(x3), sin_x3_aux: sp.sin(x3)}) for monom in z_monomials]

print("z_monomials:", z_monomials)
test_monomials = sorted(itermonomials([x1, x2, x3, sp.cos(x3), sp.sin(x3)], deg*aug_deg, 1), key=monomial_key('grlex', [sp.sin(x3), sp.cos(x3), x3, x2, x1]))
test_monomials.insert(len(variables),const)
print("Test monoms:", test_monomials)
with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=200):
    print(AplusB_augmented)
    print(A)
    print(B1)
    print(B2)