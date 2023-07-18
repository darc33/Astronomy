# ---------------------COINCIDENCIAS V3.0---------------------------------------
# ESTE PROYECTO ES LA ULTIMA VERSION, PARA ENCONTRAR COINCIDENCIAS SE HACE USO DE
# LA LIBRERIA DE ASTROPY ESTO CONLLEVA A QUE SE DISMINUYA EL TIEMPO DE CALCULO,
# ENTRE MAS OBJETOS MAS RAPIDO SE EJECUTA, SI LOS ARCHIVOS CONTIENEN APROX
# 1M DE OBJETOS (ESTRELLAS, GALAXIAS, ETC.) SE DEMORARIA APROX 20 SEG
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
import time

# A function to create a random catalogue of size n
def create_cat(n):
    ras = np.random.uniform(0, 360, size=(n, 1))
    decs = np.random.uniform(-90, 90, size=(n, 1))
    return np.hstack((ras, decs))

#a atraves de la libreria de astropy encuentra las coincidencias entre catalogos
def find_closest(cat1, cat2):
  ID,d_array,useless = cat1.match_to_catalog_sky(cat2)
  
  return ID, d_array.value

def crossmatch (cat1,cat2,maxd):
  no_match = []
  sky_cat1 = SkyCoord(cat1*u.degree, frame='icrs')#icrs is the equatorial coordinates
  sky_cat2 = SkyCoord(cat2*u.degree, frame='icrs')
  ID_cat1 = np.arange(len(cat1)) 
  
  start = time.perf_counter()
  
  ID_super,d = find_closest(sky_cat1,sky_cat2)
  cond = np.where(d < maxd)  
  
  match = list(zip(ID_cat1[cond],ID_super[cond],d[cond]))
  no_match = ID_cat1[np.where(d > maxd)]
      
  tim = time.perf_counter() - start
      
  return match,no_match,tim


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