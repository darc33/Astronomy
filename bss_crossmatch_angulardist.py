# ---------------------COINCIDENCIAS V1.0---------------------------------------
# ESTE PROYECTO ES LA VERSION BASICA, PARA ENCONTRAR COINCIDENCIAS SE 
# COMPARA CADA COORDENADA DE UN ARCHIVO CON CADA COORDENADA DEL OTRO ARCHIVO 
# ESTO CONLLEVA A QUE SEA MUY DEMORADO, SI LOS ARCHIVOS CONTIENEN APROX
# 1M DE OBJETOS (ESTRELLAS, GALAXIAS, ETC.) SE DEMORARIA APROX 24 DIAS

from __future__ import division #para hacer la division flotante de python 3.0
import numpy as np


#Convert right ascension from HMS to decimal degrees
def hms2dec(h,m,s):
  dec = 15*(h + m/60+ s/(60*60))
  return dec

#Convert declination from DMS to decimal degrees
def dms2dec(d,m,s):
  if d < 0:
    dec = -1*(-1*d + m/60 + s/(60*60))
  else:
    dec = d + m/60 + s/(60*60)
  return dec
  
#Calculate the angular distance between 2 objects in the universe
def angular_dist(ra1,dec1,ra2,dec2):#ra:right ascencion, dec:declination
  ra1_rad= np.radians(ra1)
  dec1_rad= np.radians(dec1)
  ra2_rad= np.radians(ra2)
  dec2_rad= np.radians(dec2)
  
  a = np.sin(np.abs(dec1_rad - dec2_rad)/2)**2
  b = np.cos(dec1_rad)*np.cos(dec2_rad)*np.sin(np.abs(ra1_rad - ra2_rad)/2)**2
  d_rad = 2*np.arcsin(np.sqrt(a+b))
  d = np.degrees(d_rad)
  
  return d

#lee un archivo del radiotelescopio AT20G Bright Source Sample(BSS)  
def import_bss():
  
  cat = np.loadtxt('bss.dat', usecols=range(1, 7))
  row,col = cat.shape
  tuples = []
  for i in range(row):
    h=cat[i,0]
    m=cat[i,1]
    s=cat[i,2]
    d=cat[i,3]
    mm=cat[i,4]
    ss=cat[i,5]
    ra_deg = hms2dec(h,m,s)
    dec_deg = dms2dec(d,mm,ss)
    tuples.append((i+1,ra_deg, dec_deg))
  
  return tuples

#lee un archivo del telescopio optico SuperCOSMOS all-sky galaxy
def import_super():
  
  cat = np.loadtxt('super.csv', delimiter=',', skiprows=1, usecols=[0, 1])
  row,col = cat.shape
  tuples = []
  
  for i in range(row):    
    ra_deg = cat[i,0]
    dec_deg = cat[i,1]
    tuples.append((i+1,ra_deg, dec_deg))
 
  return tuples

#find the closest match in catalogue for a target source     
def find_closest(cat, ra, dec):
  l=len(cat)
  tuples = []
  for i in range(l):
    ID,ra1,dec1 = cat[i] 
    d = angular_dist(ra1,dec1,ra,dec) 
    tuples.append((ID,d))    
  
  return min(tuples, key = lambda t:t[1])

#crossmatches 2 catalogues within a maximum distance and gives the matches and non matches  
def crossmatch (cat1, cat2, maxd):
  match = []
  no_match = []
  l =len(cat1)
  
  for i in range(l):
    ID_bss,ra, dec = cat1[i]
    ID_super, d = find_closest(cat2,ra,dec)
    
    if d < maxd:
      match.append((ID_bss,ID_super,d))
    else:
      no_match.append(ID_bss)
      
  return match,no_match
  
if __name__ == '__main__':
  print("hms2dec: %f" % hms2dec(23, 12, 6))
  print("dms2dec: %f" % dms2dec(22, 57, 18))
  print("dms2dec: %f" % dms2dec(-66, 5, 5.1))
  print("")
  
  print("dist: %f" % angular_dist(21.07, 0.1, 21.15, 8.2))
  print("dist: %f" % angular_dist(10.3, -3, 24.3, -29))
  print("")
  
  bss_cat = import_bss()
  super_cat = import_super()
  print("3 prim bss cat: [%s]" % ', '.join(map(str,bss_cat[:3])))
  print("3 prim super cat: [%s]"  % ', '.join(map(str,super_cat[:3])))
  print("")
  
  cat = import_bss()
  print("Closest: [%s]" % ', '.join(map(str,find_closest(cat, 175.3, -32.5))))
  print("Closest: [%s]" % ', '.join(map(str,find_closest(cat, 32.2, 40.7))))
  print("")
  
  max_dist = 40/3600 #max 40 arcseconds of separation
  
  matches, no_matches = crossmatch(bss_cat, super_cat, max_dist)
  print("Matches: [%s]" % ', '.join(map(str,matches[:3])))
  print("Non Matches: [%s]" % ', '.join(map(str, no_matches[:3])))
  print(len(no_matches))#number of objects haven't match
  print("")

  max_dist = 5/3600 #max 5 arcseconds of separation
  matches, no_matches = crossmatch(bss_cat, super_cat, max_dist)
  print("Matches: [%s]" % ', '.join(map(str,matches[:3])))
  print("Non Matches: [%s]" % ', '.join(map(str, no_matches[:3])))
  print(len(no_matches))#number of objects haven't match