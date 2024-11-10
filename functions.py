# Title: fucntions.py
# Author: Steven Kurniawan Chandra
# Date: September 2023
# Type: Source Code


### Hash function
from hashlib import sha256
def H(x):
	return int(sha256(bytes(x)).hexdigest(), 16)

### calculate current K given the K max and distribution type
def K_calculation(K_max, K_min, j, t, K_type, ascending = True):
	if K_type == "Linear":
		if ascending:
			return round((K_max-K_min)*(j+1)/t + K_min)
		else:
			return round((K_max-K_min)*(t-j)/t + K_min)
	elif K_type == "Exponential":
		if ascending:
			return round((K_max-K_min)**((j+1)/t) + K_min)
		else:
			return round((K_max-K_min)**((t-j)/t) + K_min)
	elif K_type == "Sigmoid":
		sigmoid_curviness = 5 ## increase this value to get a "curvier" sigmoid distribution
		from math import e
		if ascending:
			return (1/(1+e**((t/2 - (j+1))/t*2*sigmoid_curviness)))*(K_max-K_min) + K_min
		else:
			return (1/(1+e**(-1*(t/2 - (j))/t*2*sigmoid_curviness)))*(K_max-K_min) + K_min
	elif K_type == "Logarithmic":
		from math import log
		if ascending:
			return round(log(j+1, t)*(K_max-K_min)+K_min)
		else:
			return round(log(t-j, t)*(K_max-K_min)+K_min)
	elif K_type == "Radical":
		from math import sqrt
		if ascending:
			return round(sqrt((j+1)/t)*(K_max-K_min)+K_min)
		else:
			return round(sqrt((t-j)/t)*(K_max-K_min)+K_min)
	elif K_type == "Constant":
		return K_max
	elif K_type == "Every 2":
		if j%2 == 0:
			return K_max
		else:
			return K_min
	elif K_type == "Every 5":
		if j%5 == 0:
			return K_max
		else:
			return K_min

### REDUCTION FUNCTIONS
### vanilla reduction function
def r(x,i,k,t,N):
	return (x+(k*t+i))%N

### multiplication reduction function
def multiplication(x, i, k, t, N):
	return (x*(2*(i+k*t)+1))%N

### Murmurhash3
import mmh3
def murmur3(x, i, k, t, N):
	### use index as seed and do mod N at the end
	return mmh3.hash(hex(x), i + k*t, signed=False)%N

### xxhash
import xxhash
def xx(x, i, k, t, N):
	### use index as seed and do mod N at the end
	return xxhash.xxh64((hex(x)), seed = (i + k*t)).intdigest()%N

### bit picking
def bit_picking(x, bit_comb):
	n = len(bit_comb)
	val = 0
	for i in bit_comb:
		n -= 1
		if x & 1<<i: #check the x th bit
			val += 1<<n #if true increase the value by 2^i
	return val