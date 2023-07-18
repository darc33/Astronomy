'''   EXOPLANETS
Attribute                 Description
Kepler ID	Unique target identification number for stars
KOI name	String identifier for the Kepler Object of Interest (KOI)
Teff (K)	Effective temperature of a star in Kelvin
Radius	        Radius of stars and planets in units of solar radius/earth radius respectively
Kepler name	Unique string identifier for a confirmed exoplanet in the planet table
Period	        Orbital period of a planet in units of days
Status	        Status of a discovered KOI in the planet table, e.g. "confirmed" or "false positive"
Teq	        Equilibrium temperature of a planet in Kelvin

'''
#CREATES DATABASE FROM A CSV FOR STARS
CREATE TABLE Star (
  kepler_id INTEGER NOT NULL,
  koi_name VARCHAR(20) NOT NULL,
  t_eff INTEGER,
  radius FLOAT(5),
  PRIMARY KEY (koi_name)
);

COPY Star (kepler_id, koi_name, t_eff, radius) FROM '/Resources/stars.csv' CSV;

#CREATES DATABASE FROM A CSV FOR PLANETS
CREATE TABLE Planet (
  kepler_id INTEGER NOT NULL,
  koi_name VARCHAR(20) NOT NULL,
  kepler_name VARCHAR(20),
  status VARCHAR(20) NOT NULL,
  period FLOAT NOT NULL,
  radius FLOAT NOT NULL,
  t_eq INTEGER NOT NULL,
  PRIMARY KEY (koi_name)
);

COPY Planet (kepler_id, koi_name, kepler_name, status, period, radius, t_eq) FROM 'Resources/planets.csv' CSV;

#QUERIES
Select * from Star; #Select all atributes from the selected database
Select koi_name, radius from Star; #Select specific atributes 
SELECT koi_name, radius FROM Star WHERE radius < 2; #Only returns data which means certain conditions
SELECT radius, t_eff FROM Star WHERE radius > 1; #Stars that are larger than our sun

SELECT radius FROM Star WHERE radius >= 1 AND radius <= 2;#SQL allow conditional queries
SELECT radius FROM  Star WHERE radius BETWEEN 1 AND 2;#Own sintaxis of SQL that do the same of above

SELECT kepler_id, t_eff FROM Star WHERE t_eff BETWEEN 5000 AND 6000;#Select stars where the temperature is between 5000 and 6000 Kelvin

\d Planet;#Get a description from a database or table

SELECT kepler_name, radius FROM Planet 
WHERE kepler_name IS not NULL AND radius BETWEEN 1 AND 3;#Select planets that are confirmed(name not null) and has radius between 1 and 3 times earth
SELECT COUNT(*) FROM Planet;#Return the number of rows 

SELECT MIN(radius), MAX(radius), AVG(radius) #Return the min, max, and average of radius of planets
FROM Planet;
SELECT SUM(t_eff), STDDEV(t_eff)#Return the sum and standard deviation of temperatures
FROM Star;

SELECT koi_name, radius FROM Planet 
ORDER BY radius DESC#Order the table by descending order, ASC for ascending
LIMIT 5;# Only the five largest planets

SELECT MIN(radius), MAX(radius), AVG(radius), STDDEV(radius)#Calculates min, max, avg, stddev for radius
FROM Planet WHERE kepler_name IS NULL ;#of unconfirmed exoplanets (null name)

SELECT radius, COUNT(koi_name) 
FROM Planet 
GROUP BY radius;#Group the query by an atribute in this case by radius and count the planets with the same radius

 SELECT radius, COUNT(koi_name) 
FROM Planet 
GROUP BY radius
HAVING COUNT(koi_name) > 1;#Do the same as above but just select the ones where the count is larger than one 

SELECT kepler_id, COUNT(koi_name) #Find how many planets are in the same star 
FROM Planet 
GROUP BY kepler_id
HAVING COUNT(koi_name) > 1        #Select only the star that have more than 1 planet
ORDER BY COUNT(koi_name) DESC;    #Ordered the stars by the numbers of planets that it has
