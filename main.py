import math
import sympy as sym

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
	dx2x3 = ((math.exp(x1)-x2)*(2*(x3+x4))+2*math.exp(x3+x4)*(math.exp(x3+x4)-x5))/(2*((x1**2 + (math.exp(x1) - x2)**2 + (x3 + x4)**2 + (math.exp(x3+x4) - x5)**2)**(1.5)))
	dx2x4 = dx2x3
	dx2x5 = ((math.exp(x1)-x2)*(math.exp(x3+x4)-x5))/(((x1**2 + (math.exp(x1) - x2)**2 + (x3 + x4)**2 + (math.exp(x3+x4) - x5)**2)**(1.5)))
	
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
	phi_x1 = x1 + d[0] * t
	phi_x2 = x2 + d[1] * t
	phi_x3 = x3 + d[2] * t
	phi_x4 = x4 + d[3] * t
	phi_x5 = x5 + d[4] * t
	return funcao(phi_x1, phi_x2, phi_x3, phi_x4, phi_x5)

def gradiente_x_d(x1, x2, x3, x4, x5, d):
    grad = gradiente(x1,x2, x3, x4, x5)
    return grad[0]*d[0] + grad[1]*d[1] + grad[2]*d[2] + grad[3]*d[3] + grad[4]*d[4]

def armijo(N, gamma, x1, x2, x3, x4, x5, d):
	t = 1.0
	while(phi(x1,x2, x3, x4, x5, d, t) > (funcao(x1,x2, x3, x4, x5) + N * t * gradiente_x_d(x1,x2, x3, x4, x5, d))):
		t *= gamma
	return t

def passed_tolerance(d, tol):
	for i in range(len(d)):
		if(abs(d[i]) >= tol): return False
	return True

# Gradient Method
def gradient_method(x1, x2, x3, x4, x5, tol, max_interation):
	k = 0
	N = 0.25
	gamma = 0.8
	t = 0

	d = gradiente(x1, x2, x3, x4, x5)
	for i in range(len(d)):
		d[i] = -d[i]
	
	while ((not passed_tolerance(d, tol)) and k < max_interation):
		t = armijo(N, gamma, x1, x2, x3, x4, x5, d)
		x1 = x1 + t * d[0]
		x2 = x2 + t * d[1]
		x3 = x3 + t * d[2]
		x4 = x4 + t * d[3]
		x5 = x5 + t * d[4]
		d = gradiente(x1, x2, x3, x4, x5)
		for i in range(len(d)):
			d[i] = -d[i]
		k+=1
	print(f"Iter: {k} | x1: {x1} | x2: {x2} | x3: {x3} | x4: {x4} | x5: {x5} | value: {funcao(x1, x2, x3, x4, x5)}")
	return [x1, x2, x3, x4, x5, k]

# Newton Method

# Quase-Newton Method




# print(hessiana(1, 1, 1, 1, 10)[2][3])
gradient_method(1, 1, 1, 1, 1, 0.00001, 1000)



# Teste
# def solve_gradient():
# 	sym.init_printing()
# 	x1,x2,x3,x4,x5 = sym.symbols('x1,x2,x3,x4,x5')

# 	dx1 = sym.Eq(((x1**2 + (sym.exp(x1) - x2)**2 + (x3 + x4)**2 + (sym.exp(x3+x4) - x5)**2)**(-0.5))*(x1 + (sym.exp(x1)-x2)*sym.exp(x1)),0)
# 	dx2 = sym.Eq(((x1**2 + (sym.exp(x1) - x2)**2 + (x3 + x4)**2 + (sym.exp(x3+x4) - x5)**2)**(-0.5))*(-1*(sym.exp(x1)-x2)),0)
# 	dx3 = sym.Eq(((x1**2 + (sym.exp(x1) - x2)**2 + (x3 + x4)**2 + (sym.exp(x3+x4) - x5)**2)**(-0.5))*((x3+x4)+(sym.exp(x3+x4) - x5)*sym.exp(x3+x4)),0)
# 	dx4 = sym.Eq(((x1**2 + (sym.exp(x1) - x2)**2 + (x3 + x4)**2 + (sym.exp(x3+x4) - x5)**2)**(-0.5))*((x3+x4)+(sym.exp(x3+x4) - x5)*sym.exp(x3+x4)),0)
# 	dx5 = sym.Eq(((x1**2 + (sym.exp(x1) - x2)**2 + (x3 + x4)**2 + (sym.exp(x3+x4) - x5)**2)**(-0.5))*(-1*(sym.exp(x3+x4) - x5)),0)

# 	print(sym.solve([dx1,dx2,dx3,dx4,dx5],(x1,x2,x3,x4,x5)))

# solve_gradient()
