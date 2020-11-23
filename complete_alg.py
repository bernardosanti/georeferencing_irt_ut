import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
import time
import rasterio as rio
import pymap3d
from scipy.spatial.transform import Rotation as R
from rasterio.windows import Window
from math import cos, sin, atan, atan2, asin, floor, ceil, sqrt

def sigma_points(var_in):
	n=len(var_in)
	var_total = np.diag(var_in)
	var_total = np.multiply(var_total,var_total)
	alpha = 1/sqrt(n)
	beta = 2
	kapa = 0
	lambd = alpha**2*(n+kapa)-n
	sigma_points_pos = np.sqrt(np.multiply((n+lambd),var_total))
	sigma_points_neg = -sigma_points_pos
	first = np.zeros([n,1])
	points = np.concatenate((first,sigma_points_pos, sigma_points_neg),axis=1)

	weights_av = np.zeros([1,2*n+1])
	weights_av[0,0] = lambd/(n+lambd)
	weights_av[0,1:] = 1/(2*(n+lambd))
	weights_cov = np.zeros([1,2*n+1])
	weights_cov[0,0]=weights_av[0,0] + (1-alpha**2+beta)
	weights_cov[0,1:]=weights_av[0,1:]

	return points, weights_av, weights_cov

def ned2geodetic(north,east,down, lat0, lon0, h0, spheroid):
	up = -down
	lat0 = lat0 *np.pi/180 
	lon0 = lon0 * np.pi/180

	a = spheroid.semimajor_axis
	b = spheroid.semiminor_axis
	e = spheroid.eccentricity
	N = a/sqrt(1-e**2 * sin(lat0)**2)

	#origem do NED em coordenadas ECEF
	x0 = (N + h0) * cos(lat0) * cos(lon0)
	y0 = (N + h0) * cos(lat0) * sin(lon0)
	z0 = (N * (b / a) ** 2 + h0) * sin(lat0)

	t = cos(lat0) * up - sin(lat0) * north;
	dx = cos(lon0) * t - sin(lon0) * east
	dy = sin(lon0) * t + cos(lon0) * east
	dz = (sin(lat0) * up + cos(lat0) * north)

	x = x0 + dx
	y = y0 + dy
	z = z0 + dz

	r = sqrt(x**2+y**2)
	F = 54*b**2*z**2
	G = r**2 + (1 - e**2)*z**2 - e**2*(a**2 - b**2)
	c = e**4*F*r**2/G**3
	s = (1+c+sqrt(c**2+2*c))**(1/3)
	T = F/(3*(s+1/s+1)**2*G**2)
	Q = sqrt(1+2*e**4*T)
	r_0 = -(T*e**2*r)/(1+Q) + sqrt((a**2)/2 * (1+1/Q) - (T*(1-e**2)*z**2)/(Q*(1+Q)) - (T*r**2)/2)
	U = sqrt((r-e**2*r_0)**2 + z**2)
	V = sqrt((r-e**2*r_0)**2 + (1-e**2)*z**2)
	z_0 = b**2*z/(a*V)
	e_0 = (a**2 - b**2)/b**2

	h = U*(1 - b**2/(a*V))
	lat = atan((z + e_0*z_0)/r)*180/np.pi
	lon = atan2(y,x)*180/np.pi

	return np.array([lat,lon,h])

def bilinear_interpolation(x, y, points):
	points = sorted(points)               
	(x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

	if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
		raise ValueError('points do not form a rectangle')
	if not x1 <= x <= x2 or not y1 <= y <= y2:
		raise ValueError('(x, y) not within the rectangle')

	return (q11 * (x2 - x) * (y2 - y) +
			q21 * (x - x1) * (y2 - y) +
			q12 * (x2 - x) * (y - y1) +
			q22 * (x - x1) * (y - y1)
		   ) / ((x2 - x1) * (y2 - y1) + 0.0)

def interp_map(src,lat,lon,h):
	bounds = src.bounds
	res = src.res

	if lat>bounds.top or lat<bounds.bottom or lon>bounds.right or lon<bounds.left:
		return np.nan

	lat_multiplier = (bounds.top-lat)/res[1] - 0.5
	lon_multiplier = (lon-bounds.left)/res[0] - 0.5

	lon_floor = floor(lon_multiplier)
	lon_ceil = ceil(lon_multiplier)

	lat_floor = floor(lat_multiplier)
	lat_ceil = ceil(lat_multiplier)

	z = src.read(1, window=Window(lon_floor,lat_floor,2,2))

	points = [[lat_floor,lon_floor,z[0,0]],
		  [lat_floor,lon_ceil,z[0,1]],
		  [lat_ceil,lon_floor,z[1,0]],
		  [lat_ceil,lon_ceil,z[1,1]]]

	z = bilinear_interpolation(lat_multiplier,lon_multiplier, points)
	return z

def irt_project(A, max_height,pixels, origin, rx, ry, rz, r_az, r_el,sigma_point):

	pos_intersection = np.zeros([1,3])
	# parametros da camara
	cx=319.5
	cy=239.5
	fx=481.2
	fy=480.0
	
	# orientacao
	Rxyz = R.from_euler('xyz', [rx+sigma_point[3],ry+sigma_point[4],rz+sigma_point[5]])

	R_gimbal = R.from_euler('xyz',[0,r_el+sigma_point[6],r_az+sigma_point[7]])
	
	R_camera_to_world = R.from_matrix([[0, 0, 1],
									[1, 0, 0],
									[0, 1, 0]])

	#calcula o raio optico no referencial inercial
	dir_ray_camara = [pixels[0]-cx,pixels[1]-cy,(fx+fy)/2]
	dir_ray_inertial = R_camera_to_world.apply(dir_ray_camara)
	dir_ray_inertial = dir_ray_inertial/np.linalg.norm(dir_ray_inertial)
	ray_gimbal = R_gimbal.apply(dir_ray_inertial)
	ray =Rxyz.apply(ray_gimbal);
	
	#comeca a iterar com Z igual a altura maxima do mapa
	if origin[2]>max_height and ray[2] != 0:
		scale_factor = (origin[2]-max_height)/ray[2]
		new_origin_x = scale_factor*ray[0]
		new_origin_y = scale_factor*ray[1]
		origin_loop =np.array([new_origin_x, new_origin_y, origin[2]-max_height])
	else:
		origin_loop= [0,0,0]
	
	origin_loop = origin_loop + sigma_point[0:3]
	step = 100
	step_divider = 10
	step_thresh = 1
	intersection = 0
	pos_ray = origin_loop
	i=0
	while intersection != 1:
		i=i+1
		pos_ray = pos_ray + step*ray
		
		#calcula a elevacao do ponto por interpolação
		pos_geodetic = ned2geodetic(pos_ray[0], pos_ray[1], pos_ray[2], origin[0], origin[1], origin[2], pymap3d.Ellipsoid('wgs84')) 

		z = interp_map(A,pos_geodetic[0],pos_geodetic[1], pos_geodetic[2])
		
		if np.isnan(z) or i > 200:
			return np.nan
		#caso a elevacao do mapa seja superior a elevacao do raio, houve intersecao
		#ativa-se o step pequeno para melhorar a estimativa
		elif z>pos_geodetic[2] and step > step_thresh:
			pos_ray = pos_ray - step*ray
			step = step/step_divider
			continue
		#conclui o algoritmo, foi detetada a intersecao com o step pequeno
		elif z>pos_geodetic[2] and step <= step_thresh:
			pos_intersection = pos_ray
			intersection = 1
	return pos_intersection

def unscented_transform(A, max_height,pixels, origin, rx, ry, rz, r_az, r_el):
	points, weights_av, weights_cov = sigma_points([10,10,10,0.01745,0.01745,3*0.01745,0.01745,0.01745])
	n = points.shape[1]
	N = len(pixels)

	media_geodetic = np.zeros([N,3])
	P_xyz = np.zeros([N,3])
	pos_intersection = np.zeros([n,3])
	media = np.zeros([3,])
	for j in range(0,N):
		for i in range(0,n):
			sigma_point = points[:,i]
			pos_intersection[i,:] = irt_project(A, max_height,pixels[j], origin, rx, ry, rz, r_az, r_el,sigma_point)

		#detecao de erro
		if np.isnan(pos_intersection).any():
			return np.nan, np.nan

		media = np.dot(weights_av,pos_intersection)
		media_geodetic[j,:] = ned2geodetic(media[0,0], media[0,1], media[0,2], origin[0], origin[1], origin[2], pymap3d.Ellipsoid('wgs84')) 
		
		P = np.zeros([3,3])     
		for i in range(0,n):
			diff = np.subtract(pos_intersection[i,:],media)
			P_new = np.matmul(np.transpose(diff),diff)*weights_cov[0,i]
			P = np.add(P,P_new)
		
		P_xyz[j,:] = np.sqrt(np.diag(P))
	return media_geodetic, P_xyz

def max_height_finder(src,origin):
	bounds = src.bounds
	res = src.res

	lim = 350
	lat_multiplier = (bounds.top-origin[0])/res[1] - 0.5
	lon_multiplier = (origin[1]-bounds.left)/res[0] - 0.5

	lon_floor = floor(lon_multiplier)
	lon_ceil = ceil(lon_multiplier)

	lat_floor = floor(lat_multiplier)
	lat_ceil = ceil(lat_multiplier)

	z = src.read(1, window=Window(lon_floor,lat_floor,lim,lim))

	if z.size ==0:
		output = np.inf
	else:
		output = z.max()
	return output

def image_reader(frame):
	#img1 = cv2.imread(sys.argv[1],0)
	img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#img = cv2.resize(img1, (1280, 720), interpolation = cv2.INTER_AREA)
	w, h = img.shape[::-1]

	'''
	el = img[196:539, 90:134]
	az = img[65:95,460:840]
	lat = img[640:659,1080:1280]
	lon = img[660:679,1080:1280]
	alt = img[680:699, 1080:1250]
	'''
	el = img[int(h/2-180):int(h/2+180), 85:138]
	az = img[65:95,int(w/2-190):int(w/2+190)]
	lat = img[h-80:h-59,w-190:w]
	lon = img[h-60:h-39,w-190:w]
	alt = img[h-40:h-19, w-190:w-25]
	zoom = img[140:160,w-190:w-75]
	img_list = [el,az,lat,lon,alt,zoom]

	minqual = 0.1
	index = 0
	data = np.zeros(len(img_list),)
	for img in img_list:
		numero = []
		ordem = []
		for i in range(0,13):
			template = cv2.imread(str(i)+'.png',0)
			w, h = template.shape[::-1]

			# Apply template Matching
			res = cv2.matchTemplate(img,template,1)
			min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

			match_locations = np.where(res<=minqual)

			if match_locations[0].size>0:
				for (x, y) in zip(match_locations[1], match_locations[0]):
					if i == 10:
						numero.append('-')
					elif i == 11:
						numero.append('.')
					elif i == 12:
						numero.append('.')
					else:
						numero.append(i)
					ordem.append(x)

		numero = [x for _,x in sorted(zip(ordem,numero))]

		s = [str(i) for i in numero] 
		#nao leu
		#if not s:
		try:
			res = float("".join(s))
		except ValueError:
			return np.nan
		data[index]=res
		index = index+1

	return data

def video_reader(src):
	cap = cv2.VideoCapture(sys.argv[1])
	count = 0
	while cap.isOpened():
		ret,frame = cap.read()
		count = count+1
		if ret and count % 25 == 0:
			data = image_reader(frame)
			if np.isnan(data).any() or data[5] < 1:
				print('Invalid read or zoom')
				continue
			else:
				georef(src,data)
		elif not ret:
			break

	cap.release()

def georef(src, data):
	pixels = np.array([[319.5,239.5],[319.5,480],[1,480],[640,480]])
	origin = np.array([data[2], data[3], data[4]*0.3048])
	max_height = max_height_finder(src,origin)

	#print(max_height)

	t0 = time.time()
	x, P_xyz = unscented_transform(src,max_height, pixels,origin,0,0,0,data[1]*np.pi/180,data[0]*np.pi/180)
	t1 = time.time()
	total = t1-t0
	print('Posições estimadas [lat(º) lon(º) h(m)]:')
	print(x)
	print('Incerteza [x y z](m):')
	print(P_xyz)

def main():
	src = rio.open('../../../dem_portugal/junto_wsg84.tif')
	#src = rio.open('porto_de_mos_4326.tif')
	video_reader(src)
	'''
	data = image_reader()
	
	pixels = np.array([[319.5,239.5]])
	origin = np.array([data[2], data[3], data[4]*0.3048])
	max_height = max_height_finder(src,origin)

	#print(max_height)

	t0 = time.time()
	x, P_xyz = unscented_transform(src,max_height, pixels,origin,0,0,0,data[1]*np.pi/180,data[0]*np.pi/180)
	t1 = time.time()
	total = t1-t0
	print('Posições estimadas [lat(º) lon(º) h(m)]:')
	print(x)
	print('Incerteza [x y z](m):')
	print(P_xyz)
	'''

if __name__ == "__main__":
	main()
