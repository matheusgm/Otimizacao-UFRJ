import math
import sympy as sp
import numpy as np

# Função
def funcao(x1, x2, x3, x4, x5):
	return math.sqrt(x1**2 + (math.exp(x1) - x2)**2 + (x3 + x4)**2 + (math.exp(x3+x4) - x5)**2)

# Gradiente
def gradiente(x1, x2, x3, x4, x5):
	dx1 = ((x1**2 + (math.exp(x1) - x2)**2 + (x3 + x4)**2 + (math.exp(x3+x4) - x5)**2)**(-0.5))*(x1 + (math.exp(x1)-x2)*math.exp(x1))
	dx2 = ((x1**2 + (math.exp(x1) - x2)**2 + (x3 + x4)**2 + (math.exp(x3+x4) - x5)**2)**(-0.5))*(-1*(math.exp(x1)-x2))
	dx3 = ((x1**2 + (math.exp(x1) - x2)**2 + (x3 + x4)**2 + (math.exp(x3+x4) - x5)**2)**(-0.5))*((x3+x4)+(math.exp(x3+x4) - x5)*math.exp(x3+x4))
	dx4 = dx3
	dx5 = ((x1**2 + (math.exp(x1) - x2)**2 + (x3 + x4)**2 + (math.exp(x3+x4) - x5)**2)**(-0.5))*(-1*(math.exp(x3+x4) - x5))
	return [dx1, dx2, dx3, dx4, dx5]


# Hessiana
def hessiana(x1, x2, x3, x4, x5):
	#1
	dx1x1 = ((math.exp(x1)*(math.exp(x1) - x2)+math.exp(2*x1)+1)/((x1**2 + (math.exp(x1) - x2)**2 + (x3 + x4)**2 + (math.exp(x3+x4) - x5)**2)**(0.5))) - (((x1+math.exp(x1)*(math.exp(x1) - x2))**2)/((x1**2 + (math.exp(x1) - x2)**2 + (x3 + x4)**2 + (math.exp(x3+x4) - x5)**2)**(1.5)))
	dx1x2 = (((x1+math.exp(x1)*(math.exp(x1) - x2))*(math.exp(x1) - x2))/((x1**2 + (math.exp(x1) - x2)**2 + (x3 + x4)**2 + (math.exp(x3+x4) - x5)**2)**(1.5))) - (math.exp(x1)/(x1**2 + (math.exp(x1) - x2)**2 + (x3 + x4)**2 + (math.exp(x3+x4) - x5)**2)**(0.5))
	dx1x3 = -(((2*x1+2*math.exp(x1)*(math.exp(x1)-x2))*(2*(x3+x4)+2*math.exp(x3+x4)*(math.exp(x3+x4)-x5)))/(4*((x1**2 + (math.exp(x1) - x2)**2 + (x3 + x4)**2 + (math.exp(x3+x4) - x5)**2)**(1.5))))
	dx1x4 = dx1x3
	dx1x5 = ((2*x1 + 2*math.exp(x1)*(math.exp(x1)-x2))*(math.exp(x3+x4)-x5))/(2*((x1**2 + (math.exp(x1) - x2)**2 + (x3 + x4)**2 + (math.exp(x3+x4) - x5)**2)**(1.5)))
	
    #2
	dx2x1 = dx1x2
	dx2x2 = (1/((x1**2 + (math.exp(x1) - x2)**2 + (x3 + x4)**2 + (math.exp(x3+x4) - x5)**2)**(0.5)))-(((math.exp(x1)-x2)**2)/((x1**2 + (math.exp(x1) - x2)**2 + (x3 + x4)**2 + (math.exp(x3+x4) - x5)**2)**(1.5)))
	dx2x3 = ((math.exp(x1)-x2)*(2*(x3+x4)+ 2*math.exp(x3+x4)*(math.exp(x3+x4)-x5)))/(2*((x1**2 + (math.exp(x1) - x2)**2 + (x3 + x4)**2 + (math.exp(x3+x4) - x5)**2)**(1.5)))
	dx2x4 = dx2x3
	dx2x5 = -((math.exp(x1)-x2)*(math.exp(x3+x4)-x5))/(((x1**2 + (math.exp(x1) - x2)**2 + (x3 + x4)**2 + (math.exp(x3+x4) - x5)**2)**(1.5)))
	
    #3
	dx3x1 = dx1x3
	dx3x2 = dx2x3
	dx3x3 = ((2*math.exp(x3+x4)*(math.exp(x3+x4)-x5)+2*math.exp(2*x3+2*x4)+2)/(2*((x1**2 + (math.exp(x1) - x2)**2 + (x3 + x4)**2 + (math.exp(x3+x4) - x5)**2)**(0.5))))-(((2*(x3+x4)+2*math.exp(x3+x4)*(math.exp(x3+x4)-x5))**2)/(4*((x1**2 + (math.exp(x1) - x2)**2 + (x3 + x4)**2 + (math.exp(x3+x4) - x5)**2)**(1.5))))
	dx3x4 = dx3x3
	dx3x5 = (((2*(x3+x4)+2*math.exp(x3+x4)*(math.exp(x3+x4)-x5))*(math.exp(x3+x4)-x5))/(2*((x1**2 + (math.exp(x1) - x2)**2 + (x3 + x4)**2 + (math.exp(x3+x4) - x5)**2)**(1.5))))-((math.exp(x3+x4))/(((x1**2 + (math.exp(x1) - x2)**2 + (x3 + x4)**2 + (math.exp(x3+x4) - x5)**2)**(0.5)))) 
	
    #4
	dx4x1 = dx1x4
	dx4x2 = dx2x4
	dx4x3 = dx3x4
	dx4x4 = dx4x3
	dx4x5 = dx3x5
	
    #5
	dx5x1 = dx1x5
	dx5x2 = dx2x5
	dx5x3 = dx3x5
	dx5x4 = dx4x5
	dx5x5 = (1/((x1**2 + (math.exp(x1) - x2)**2 + (x3 + x4)**2 + (math.exp(x3+x4) - x5)**2)**(0.5)))-(((math.exp(x3+x4)-x5)**2)/((x1**2 + (math.exp(x1) - x2)**2 + (x3 + x4)**2 + (math.exp(x3+x4) - x5)**2)**(1.5)))
	
	
	return [
		[dx1x1, dx1x2, dx1x3, dx1x4, dx1x5],
		[dx2x1, dx2x2, dx2x3, dx2x4, dx2x5],
		[dx3x1, dx3x2, dx3x3, dx3x4, dx3x5],
		[dx4x1, dx4x2, dx4x3, dx4x4, dx4x5],
		[dx5x1, dx5x2, dx5x3, dx5x4, dx5x5]
		]

def phi(x1, x2, x3, x4, x5, d, t):
	phi_x1 = x1 + (d[0] * t)
	phi_x2 = x2 + (d[1] * t)
	phi_x3 = x3 + (d[2] * t)
	phi_x4 = x4 + (d[3] * t)
	phi_x5 = x5 + (d[4] * t)
	return funcao(phi_x1, phi_x2, phi_x3, phi_x4, phi_x5)

def armijo(N, gamma, x1, x2, x3, x4, x5, d):
	t = 1
	chamadas = 0
	while(phi(x1, x2, x3, x4, x5, d, t) > (funcao(x1, x2, x3, x4, x5) + (N * t * np.dot(gradiente(x1, x2, x3, x4, x5), d)))):
		t *= gamma
		chamadas+=1
	return t, chamadas

# Gradient Method
def gradient_method(x1, x2, x3, x4, x5, tol, max_interation):
	k = 0 # Condição de parada: limite de iteracoes;
	N = 0.25
	gamma = 0.8
	t = 1
	chamadasArmijo = 0
	d = -np.array(gradiente(x1, x2, x3, x4, x5))
	
	while (np.linalg.norm(d) > tol and k < max_interation):
		t, ca = armijo(N, gamma, x1, x2, x3, x4, x5, d)
		chamadasArmijo+=ca
		x1 = x1 + (t * d[0])
		x2 = x2 + (t * d[1])
		x3 = x3 + (t * d[2])
		x4 = x4 + (t * d[3])
		x5 = x5 + (t * d[4])
		d = -np.array(gradiente(x1, x2, x3, x4, x5))
		k+=1
	print(f"Iter: {k} | x1: {x1} | x2: {x2} | x3: {x3} | x4: {x4} | x5: {x5} | value: {funcao(x1, x2, x3, x4, x5)}  | gradiente:{np.linalg.norm(gradiente(x1, x2, x3, x4, x5))}  | # Armijo: {chamadasArmijo}")
	return [x1, x2, x3, x4, x5, k, chamadasArmijo]

# Newton Method
def newton_method(x1, x2, x3, x4, x5, tol, max_interation):
	k = 0 # Condição de parada: limite de iteracoes;
	N = 0.25
	gamma = 0.8
	t = 0
	chamadasArmijo = 0
	hess = hessiana(x1, x2, x3, x4, x5)
	if(np.linalg.det(hess) < tol):
		print("Determinante proximo de 0. (start)")
		return [x1, x2, x3, x4, x5, k]

	d = -np.matmul(np.linalg.inv(hess), np.array(gradiente(x1, x2, x3, x4, x5)))

	while (np.linalg.norm(d) > tol and k < max_interation):
		t, ca = armijo(N, gamma, x1, x2, x3, x4, x5, d)
		chamadasArmijo+=ca
		x1 = x1 + t * d[0]
		x2 = x2 + t * d[1]
		x3 = x3 + t * d[2]
		x4 = x4 + t * d[3]
		x5 = x5 + t * d[4]
		if(np.linalg.det(hess) < tol):
			print("Determinante proximo de 0.")
			break
		d = -np.matmul(np.linalg.inv(hess), np.array(gradiente(x1, x2, x3, x4, x5)))
		k+=1
	print(f"Iter: {k} | x1: {x1} | x2: {x2} | x3: {x3} | x4: {x4} | x5: {x5} | value: {funcao(x1, x2, x3, x4, x5)} | # Armijo: {chamadasArmijo}")
	return [x1, x2, x3, x4, x5, k, chamadasArmijo]

# Quase-Newton Method (DFP)
def quase_newton_DFP_method(x1, x2, x3, x4, x5, tol, max_interation):
	k = 0 # Condição de parada: limite de iteracoes;
	N = 0.25
	gamma = 0.8
	t = 0
	chamadasArmijo = 0
	H = np.identity(5)
	d = -np.matmul(H, np.array(gradiente(x1, x2, x3, x4, x5)))

	while (np.linalg.norm(d) > tol and k < max_interation):
		t, ca = armijo(N, gamma, x1, x2, x3, x4, x5, d)
		chamadasArmijo+=ca
		x1_novo = x1 + t * d[0]
		x2_novo = x2 + t * d[1]
		x3_novo = x3 + t * d[2]
		x4_novo = x4 + t * d[3]
		x5_novo = x5 + t * d[4]
		p = np.array([x1_novo - x1, x2_novo - x2, x3_novo - x3, x4_novo - x4, x5_novo - x5])
		q = np.array(gradiente(x1_novo, x2_novo, x3_novo, x4_novo, x5_novo)) - np.array(gradiente(x1, x2, x3, x4, x5))
		
		numerador1 = np.outer(p, p.T)
		denominador1 = np.dot(p.T, q)
		numerador2 = np.outer(np.dot(H, q), np.dot(q.T, H))
		denominador2 = np.matmul(np.matmul(q.T, H),q)

		H_novo = (numerador1 / denominador1) - (numerador2 / denominador2)
		H = H + H_novo
		x1 = x1_novo
		x2 = x2_novo
		x3 = x3_novo
		x4 = x4_novo
		x5 = x5_novo
		d = -np.matmul(H, np.array(gradiente(x1, x2, x3, x4, x5)))
		k+=1
	print(f"Iter: {k} | x1: {x1} | x2: {x2} | x3: {x3} | x4: {x4} | x5: {x5} | value: {funcao(x1, x2, x3, x4, x5)} | gradiente:{np.linalg.norm(gradiente(x1, x2, x3, x4, x5))} | # Armijo: {chamadasArmijo}")
	return [x1, x2, x3, x4, x5, k, chamadasArmijo]



# print(hessiana(1, 1, 1, 1, 10)[2][3])
gradient_method(0.1, 0.1, 0.1, 0.1, 0.1, 0.0000001, 100)
# print(newton_method(-0.4, -0.5, -0.6, -0.5, -0.5, 0.00001, 100))
# quase_newton_DFP_method(1,1,1,1,1, 0.0000001, 1000)

############### Teste do Gradiente e da Hessiana ###############
# x1, x2, x3, x4, x5 = sp.symbols('x1 x2 x3 x4 x5')
# variables = [x1, x2, x3, x4, x5]
# f = sp.sqrt(x1**2 + (sp.exp(x1) - x2)**2 + (x3 + x4)**2 + (sp.exp(x3+x4) - x5)**2)

# gradient = [sp.diff(f, var) for var in variables]  # Gradiente da função f em relação às variáveis
# hessian = sp.Matrix(sp.hessian(f, variables))  # Hessiana da função f

# # Conversão das expressões do gradiente e da hessiana em funções lambdas para avaliação numérica
# grad_func = sp.lambdify(variables, gradient, 'numpy')
# hess_func = sp.lambdify(variables, hessian, 'numpy')

# h = hess_func(1,1,1,1,1)
# print(h)
# print(sp.det(sp.Matrix(h)))

# g = grad_func(0.0000000000000001,0.9999999999999999,0.00000000000000001,0.000000000000000001,0.999999999999999)
# print(funcao(0.0000000000000001,0.9999999999999999,0.00000000000000001,0.000000000000000001,0.999999999999999))
# print(g)
# print(gradiente(0.001,0.999,0.001,0.001,0.999))


############### Teste do "np.dot" e "np.outer" ###############
# sk = np.array([1, 2, 3])  # Vetor s_k
# yk = np.array([4, 5, 6])  # Vetor y_k
# Bk = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]])  # Matriz B_k

# # Primeira parcela
# skTyk = np.dot(sk, yk)  # Produto interno entre sk e yk
# first_term = np.outer(sk, sk) / skTyk  # Produto externo dividido por skTyk

# # Segunda parcela
# ykTBk = np.dot(yk, np.dot(Bk, yk))  # yk^T * Bk * yk
# second_term = -np.outer(np.dot(Bk, yk), np.dot(yk, Bk)) / ykTBk  # Produto externo dividido por ykTBk

# # Atualização de Bk
# Bk_plus_1 = Bk + first_term + second_term

# print("Primeira parcela:")
# print(first_term)
# print("\nSegunda parcela:")
# print(second_term)
# print("\nMatriz Bk atualizada:")
# print(Bk_plus_1)