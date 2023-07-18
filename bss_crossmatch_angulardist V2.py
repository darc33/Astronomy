# ---------------------COINCIDENCIAS V2.0---------------------------------------
# ESTE PROYECTO ES LA VERSION MEJORADA, PARA ENCONTRAR COINCIDENCIAS SE 
# COMPARA CADA COORDENADA DE UN ARCHIVO CON SOLO CIERTAS COORDENADAS DEL OTRO
# ARCHIVO(solo se compara si las declinaciones son cercanas)
# ESTO CONLLEVA A DISMINUIR EL TIEMPO DE CALCULO, SI LOS ARCHIVOS CONTIENEN APROX
# 1M DE OBJETOS (ESTRELLAS, GALAXIAS, ETC.) SE DEMORARIA APROX 10 DIAS
from __future__ import division #para hacer la division flotante de python 3.0
import numpy as np
import time

# A function to create a random catalogue of size n
def create_cat(n):
    ras = np.random.uniform(0, 360, size=(n, 1))
    decs = np.random.uniform(-90, 90, size=(n, 1))
    return np.hstack((ras, decs))

#ya no convierte una a una las coordenadas a radianes
def angular_dist(ra1,dec1,ra2,dec2):#ra:right ascencion, dec:declination
  
  a = np.sin(np.abs(dec1 - dec2)/2)**2
  b = np.cos(dec1)*np.cos(dec2)*np.sin(np.abs(ra1 - ra2)/2)**2
  d_rad = 2*np.arcsin(np.sqrt(a+b))
  d = np.degrees(d_rad)
  
  return d

#encuentra el mas cercano sin evaluar todo el catalogo    
def find_closest(cat, ra, dec,max_rad):
  
  lower = dec - max_rad
  upper = dec + max_rad
  
  index = np.searchsorted(cat[:,1], lower, side='left')
  
  tuples = []
  ind = index-1 if index-1 >= 0 else 0
  cat_s = cat[ind:,:]
  
  for i, (ra1,dec1) in enumerate(cat_s):
    if dec1 > upper:
      break
    else:
      d = angular_dist(ra1,dec1,ra,dec)      
      tuples.append((i+ind,d))
  
  try:
    return min(tuples, key = lambda t:t[1])
  except (ValueError, TypeError):
    return 0,np.degrees(max_rad)

#crossmatch tomando el tiempo que se demora en hacerlo
def crossmatch (cat1,cat2,maxd):
  match = []
  no_match = []
  
  cat1 = np.radians(cat1)
  cat2 = np.radians(cat2)
  maxd_rad= np.radians(maxd)
  
  sort_ind = np.argsort(cat2[:, 1])
  sort_cat2 = cat2[sort_ind]
  
  start = time.perf_counter()
    
  for i, (ra,dec) in enumerate(cat1):    
    
    ID_super,d = find_closest(sort_cat2,ra,dec,maxd_rad)
    
    if d < maxd:
      match.append((i,sort_ind[ID_super],d))
    else:
      no_match.append(i)
      
  tim = time.perf_counter() - start
      
  return match,no_match,tim

# Any code inside this `if` statement will be ignored by the automarker.
if __name__ == '__main__':
  
  cat1 = np.array([[180, 30], [45, 10], [300, -45]])
  cat2 = np.array([[180, 32], [55, 10], [302, -44]])
  matches, no_matches, time_taken = crossmatch(cat1, cat2, 5)
  print('matches:', matches)
  print('unmatched:', no_matches)
  print('time taken:', time_taken)  

  
  np.random.seed(0)
  cat1 = create_cat(10)
  cat2 = create_cat(20)
  matches, no_matches, time_taken = crossmatch(cat1, cat2, 5)
  print('matches:', matches)
  print('unmatched:', no_matches)
  print('time taken:', time_taken)