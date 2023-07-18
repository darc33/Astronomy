import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import statistics
import time
from helper import running_stats

#hace lo mismo que np.mean
def calculate_mean(num_list):
  mean = sum(num_list)/len(num_list)
  return mean  

#media y mediana de un archivo
def calc_stats(file):
  data = np.loadtxt(file, delimiter=',')#lee el archivo, lo separa por comas, lo coloca en un arreglo y lo transforma en float
  mean = np.round(np.mean(data), 1)
  median = np.round(np.median(data),1)
  
  tupla= (mean, median)
  return tupla

#media de una lista de archivos
def mean_datasets(list_files):
  data =[]
  for file in list_files:
    data.append(np.loadtxt(file, delimiter=','))
  mean = np.round(np.mean(data,axis=0),1)#axis = 0: saca la media de cada (fila,col) entre los archivos
  return mean

#extra las coordenadas del maximo de la imagen  
def load_fits(file):
  hdulist = fits.open(file)
  data = hdulist[0].data#extrae la informacion de la imagen
  tupla= np.unravel_index(np.argmax(data), data.shape)#devuelve el indice del maximo de los datos
  return tupla

#grafica el archivo  
def plot_fits(file):
    hdulist = fits.open(file)
    data = hdulist[0].data
    # Plot the 2D image data
    plt.imshow(data.T, cmap=plt.cm.viridis)
    plt.xlabel('x-pixels (RA)')
    plt.ylabel('y-pixels (Dec)')
    plt.colorbar()
    plt.show()

#media de una lista de archivos fits    
def mean_fits(list_files):
  
  data =[]
  for file in list_files:
    hdulist = fits.open(file)
    data.append(hdulist[0].data)
    
  mean = np.mean(data,axis=0)
  return mean
  
#media y mediana de una lista 
def list_stats(list_num):
  mean = np.mean(list_num)
  list_num.sort()
  l= len(list_num)
  if l % 2 == 0:
    median = (list_num[(l//2 - 1)]+list_num[l//2])/2
  else:
    median = list_num[l//2]
    
  return (median,mean)
  
#Mide el tiempo que se tarda una funcion  
def time_stat(func, size, ntrials):
  tim =[]
  
  for n in range(0,ntrials+1):
    data = np.random.rand(size)
    start = time.perf_counter()
    res = func(data)
    end = time.perf_counter() - start
    tim.append(end)
    
  mean = np.mean(tim)
  
  return mean

#median, time and size for list of FIST files  
def median_fits(list_files):
  data =[]
  size = 0
  
  for file in list_files:
    hdulist = fits.open(file)
    data.append(hdulist[0].data)
    
  start = time.perf_counter()  
  median = np.median(data,axis=0)
  end = time.perf_counter() - start

  for obj in data:
    size += obj.nbytes
  
  size /= 1024 #size in kB
  return (median,end,size)
  
#saca los valores del histograma de una lista de valores  
def median_bins(values, B):
  
  values = np.array(values,dtype=float) 
  mean = np.mean(values)
  std_dev = np.std(values)
  
  minval = mean-std_dev
  maxval = mean+std_dev
  
  width = (2*std_dev)/B
  count = np.size(values[values < minval]) 
  values_biss = values[((minval <= values) & (values < maxval))]
  hist = np.histogram(values_biss, B, range=(minval,maxval))
  count_val = hist[0].astype('float')
  return (mean, std_dev, count, count_val)

#saca la mediana del histograma
def median_approx(values, B):
  N = (len(values) + 1) / 2
  mean, std_dev, count, count_val = median_bins(values,B)
  lower= mean-std_dev
  total = count
  width = (2*std_dev)/B
  for lim,n in enumerate(count_val):
    total += n
    if total >= N:
      break;
      
  midpoint = lower + width*(lim + 0.5)
  return midpoint
  
#extrae los valores del histograma para una lista de archivos fits  
def median_bins_fits(list_files, B):
  mean, std = running_stats(list_files)
  
  dim = mean.shape
  minval = mean-std
  maxval = mean+std
  width = (2*std)/B
  
  left_lim = np.zeros(dim)
  hist_values = np.zeros((dim[0], dim[1], B))
  
  for file in list_files:
    hdulist = fits.open(file)
    data = hdulist[0].data
    
    left_lim[data < minval] += 1
    list_biss = np.logical_and(minval <= data, data < maxval)
    index_list = ((data[list_biss]-minval[list_biss]) / width[list_biss]).astype(int)
    hist_values[list_biss, index_list] += 1
  
  
  return (mean, std, left_lim, hist_values)
  
#extrae la mediana de una lista de archivos fits
def median_approx_fits(list_files, B):
  
  N = (len(list_files) + 1) / 2
  
  mean, std, left_lim, values = median_bins_fits(list_files,B)
  
  dim = mean.shape  
  width = (2*std)/B
  median = np.zeros(dim)
  total = left_lim
  
  for b in range(B):
    
    total += values[:,:,b]
    ind = total >= N
    
    median[ind] = mean[ind] - std[ind] + width[ind]*(b+0.5)
    total[ind] = -1E8
 
  return median 

  
if __name__ == '__main__':
  # Run your `calculate_mean` function with examples:
  mean = calculate_mean([1,2.2,0.3,3.4,7.9])
  print(mean)
  
  mean = calc_stats('Resources/data.csv')
  print(mean)
  
  print(mean_datasets(['Resources/data1.csv', 'Resources/data2.csv', 'Resources/data3.csv']))
  print(mean_datasets(['Resources/data4.csv', 'Resources/data5.csv', 'Resources/data6.csv']))
  
  bright = load_fits('Resources/image1.fits')
  print(bright)
  plot_fits('Resources/image1.fits')
  
  data  = mean_fits(['Resources/image0.fits', 'Resources/image1.fits', 'Resources/image2.fits'])
  print(data[100, 100])
  
  m = list_stats([1.3, 2.4, 20.6, 0.95, 3.1, 2.7])
  print(m)
  
  print('{:.6f}s for statistics.mean'.format(time_stat(statistics.mean, 10**5, 10)))
  print('{:.6f}s for np.mean'.format(time_stat(np.mean, 10**5, 1000)))
  
  result = median_fits(['Resources/image0.fits', 'Resources/image1.fits'])
  print(result[0][100, 100], result[1], result[2])
  
  print(median_bins([1, 1, 3, 2, 2, 6], 3))
  print(median_approx([1, 1, 3, 2, 2, 6], 3))

  print(median_bins([1, 5, 7, 7, 3, 6, 1, 1], 4))
  print(median_approx([1, 5, 7, 7, 3, 6, 1, 1], 4))
  
  mean, std, left_bin, bins = median_bins_fits(['Resources/image0.fits', 'Resources/image1.fits', 'Resources/image2.fits'], 5)
  median = median_approx_fits(['Resources/image0.fits', 'Resources/image1.fits', 'Resources/image2.fits'], 5)
  print(mean[100,100], std[100, 100], left_bin[100, 100], bins[100, 100, :], median[100, 100])