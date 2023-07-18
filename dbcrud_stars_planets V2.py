import psycopg2 
import numpy as np

#Function to return all the data
def select_all (table):             
  # Establish the connection
  conn = psycopg2.connect(dbname='db', user='darc')#initialises a new database 
  cursor = conn.cursor()#object that interfaces with the database
  # Execute an SQL query and receive the output
  query = 'SELECT * FROM ' + table + ';' 
  cursor.execute(query)
  records = cursor.fetchall()#return the output of the last query
  return records

#Function that return mean and median of a column   
def column_stats (table, column):
  conn = psycopg2.connect(dbname='db', user='grok')
  cursor = conn.cursor()  
  
  query = 'SELECT ' + column + ' FROM ' + table + ';' 
  cursor.execute(query)  
  records = cursor.fetchall()
  
  array = np.array(records)
  mean = array.mean()
  median = np.median(array) 
  return mean, median

# Do the query (SELECT kepler_id, radius FROM Star WHERE radius > 1.0 ORDER BY radius ASC;)    
def query (file):  
  table = np.loadtxt(file, delimiter=',', usecols=(0,2))
  table = table[(table[:,1]>1),:]
  table = table[np.argsort(table[:,1])]
  return table

# Do the query (SELECT p.radius/s.radius AS radius_ratio FROM Planet AS p 
#INNER JOIN star AS s USING (kepler_id) WHERE s.radius > 1.0 ORDER BY p.radius/s.radius ASC;)   
def query2 (file, file2):  
  table = np.loadtxt(file, delimiter=',', usecols=(0,2))
  table2 = np.loadtxt(file2, delimiter=',', usecols=(0,5))
  table = table[(table[:,1]>1),:]
  radius = np.array([])
  for row in table:
    for row2 in table2:
      if row[0] == row2[0]:
        radius = np.append(radius,(row2[1]/row[1])) 
 
  radius = radius[np.argsort(radius)]  
  
  return radius[np.newaxis, :].T
  

if __name__ == '__main__':
  print (select_all('Star'))
  print (select_all('Planet'))
  
  print (column_stats('Star', 't_eff'))
  
  result = query('Resources/stars.csv')
  print (result)
  
  result = query2('Resources/stars.csv', 'Resources/planets.csv')
  print(result)