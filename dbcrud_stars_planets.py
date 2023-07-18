'''CREATING, DELETING AND UPDATING DATABASES
TYPES:
    SMALLINT: signed two byte integer 
    INTEGER: Signed four byte integer
    FLOAT: Eight byte floating point number
    CHAR(n): Fixed length string with n characters
    VARCHAR(n): Valirable length string with maximum n characters
    
CONSTRAINT:
    NOT NULL: Value cannot be NULL
    UNIQUE: Value must be unique in the table
    DEFAULT: Specifies a default if the field is left blank
    CHECK: Ensures that the value meets a specific condition
    PRIMARY KEY: Combination of NOT NULL and UNIQUE  
    FOREIGN KEY(REFERENCES): Ensures the data matches the specified attribute in another table
'''

#creates the database 
CREATE TABLE Star (
  kepler_id INTEGER PRIMARY KEY,
  t_eff INTEGER,
  radius FLOAT
);

COPY Star (kepler_id, t_eff, radius) FROM 'Resources/stars.csv' CSV;

CREATE TABLE Planet (
  kepler_id INTEGER NOT NULL REFERENCES Star (keplet_id),#foreign key to references a primary key in another table
  kepler_name VARCHAR(20),
  status VARCHAR(20) NOT NULL,
  period FLOAT,
  radius FLOAT,
  t_eq INTEGER
);

COPY Planet (kepler_id, koi_name, kepler_name, status, period, radius, t_eq) FROM 'Resources/planets.csv' CSV;


#ADD NEW VALUES
INSERT INTO Star (kepler_id, t_eff, radius)
VALUES (7115384, 3789, 27.384),
       (8106973, 5810, 0.811),
       (9391817, 6200, 0.958);
       
SELECT * FROM Star #To check if the new elements were added

#UPDATE A VALUE
UPDATE Planet 
SET kepler_name = NULL#Change the name to null to the planets that aren't confirmed
WHERE status != 'CONFIRMED';

#DELETE CERTAIN ROWS
DELETE FROM Planet WHERE radius < 0;#Delete the rows where the radius is negative

#CREATES A TABLE
CREATE TABLE Planet (
  kepler_id INTEGER NOT NULL,
  koi_name VARCHAR(15) NOT NULL UNIQUE,
  kepler_name VARCHAR(15),
  status VARCHAR(20) NOT NULL,
  radius FLOAT NOT NULL
  );
  
INSERT INTO Planet (kepler_id, koi_name, kepler_name, status, radius)
VALUES (6862328, 'K00865.01', NULL, 'CANDIDATE', 119.021),
       (10187017, 'K00082.05', 'Kepler-102 b', 'CONFIRMED', 5.286),
       (10187017, 'K00082.04', 'Kepler-102 c', 'CONFIRMED', 7.071);

#ALTER A TABLE       
ALTER TABLE Star
ADD COLUMN ra FLOAT,#Add column to the table 
ADD COLUMN decl FLOAT;

'''
ALTER TABLE Star
DROP COLUMN ra, #delete column of a table
DROP COLUMN decl;

ALTER TABLE Star
 ALTER COLUMN t_eff SET DATA TYPE FLOAT;#Changes the type of a parameter
 
ALTER TABLE Star
  ADD CONSTRAINT radius CHECK(radius > 0);#add a constraint to a parameter
'''

DELETE FROM Star;#empty the table
COPY Star (kepler_id, t_eff, radius, ra, decl) FROM 'stars_full.csv' CSV;