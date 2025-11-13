#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 15:02:05 2021

@author: supphakit
"""

import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from astropy.io import fits
from astropy.table import Table
from astropy.table import QTable
import plotly.express as px

#Import
Data = fits.open('/home/supphakit/Desktop/Research/gll_psc_v22.fit')
Dat = fits.open('/home/supphakit/Desktop/Kit/imbin/fermi-allsky-000.5-001.0GeV-fwhm000-0256-nopsc.fits')
#check data
Dat.info()
Dat0 = Dat[0].header
Dat1 = Dat[1].header
Data.info()
Data0 = Data[0].header
Data1 = Data[1].header
Data2 = Data[2].header
Data3 = Data[3].header
Data4 = Data[4].header
Data5 = Data[5].header
Data6 = Data[6].header
Data7 = Data[7].header
Data8 = Data[8].header
Source_Name = Data[1].data['Source_Name']
GLON  = Data[1].data['GLON']
GLAT  = Data[1].data['GLAT']
Catalog = np.array([GLON,GLAT,Source_Name])
coordinate = np.array([GLON,GLAT])
#Select
Selectcoordinate = ((coordinate,53.5-1.5, 53.5+1.5),(coordinate,0.3-1.5, 0.3+1.5))
Selectcoordinate = coordinate[coordinate ,53.5-1.5, 53.5+1.5] 
Selectcoordinate = np.array([(53.5-1.5<= GLON <= 53.5+1.5) ,(0.3-1.5 <= GLAT <= 0.3+1.5)])
#plot, not good!!!
plt.plot(coordinate)

#Scatter plot
plt.figure(figsize=(20,10))
plt.scatter(GLON, GLAT,marker=".")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc='upper left')
plt.show()

#Select longitude = 53.5 deg, latitude = 0.3 deg, size of the region ~3x3 deg
plt.scatter(GLON, GLAT,marker=".")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc='upper left')
plt.xlim(53.5-1.5, 53.5+1.5)
plt.ylim(0.3-1.5, 0.3+1.5)
plt.show()
plt.show()

#add string, not work!!!

fig, ax = plt.subplots()
ax.scatter(GLON, GLAT)

for i, txt in enumerate(Source_Name):
    ax.annotate(txt, GLON[i], GLAT[i])
plt.show()
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc='upper left')
plt.xlim(53.5-1.5, 53.5+1.5)
plt.ylim(0.3-1.5, 0.3+1.5)
plt.show()
plt.show()

#Test
Flux_Band = Data[1].data['Flux_Band']
GLON  = Data[1].data['GLON']
GLAT  = Data[1].data['GLAT']
Energy_Flux100 = Data[1].data['Energy_Flux100']
Source_Name = Data[1].data['Source_Name']































































#read

TableData1 = Table(Data[1].data)
TableData1show = TableData1.show_in_browser() 
#Plot
Flux = Data[1].data['Flux1000']
########################
hp.mollview (Flux ,title='Gamma rays flux density',coord='G',cmap=('gray_r'),norm= 'hist',cbar=False)
########################

TB = np.array([GLON,GLAT,Flux])


############
y = [2.56422, 3.77284, 3.52623, 3.51468, 3.02199]
z = [0.15, 0.3, 0.45, 0.6, 0.75]
n = [58, 651, 393, 203, 123]

fig, ax = plt.subplots()
ax.scatter(z, y)

for i, txt in enumerate(n):
    ax.annotate(txt, (z[i], y[i]))

######################################
#plot
#Check again
#plot 3D of Gammar rays to check graph
plt.figure(figsize=(20,10))
ax = plt.axes(projection='3d')
ax.plot_trisurf(TB[0], TB[1], TB[2],
                cmap='viridis', edgecolor='none');
ax.set_title('3D Gammar rays flux of the Milky Way Galaxy');
ax.set_xlabel('Longitude [Degree]')
ax.set_ylabel('Latitude [Degree]')
ax.set_zlabel('Flux (counts/cm^2/s/sr)');
plt.xlim(53.5-3, 53.5+3)
plt.ylim(0.3-3, 0.3+3)
plt.show()



######################################################
plt.figure(figsize=(20,10))
hp.mollview(Flux_Band)

plt.plot(Flux)
plt.figure(figsize=(20,10))
plt.plot(Flux_Band)
plt.figure(figsize=(20,10))
plt.plot(GLON)
plt.figure(figsize=(20,10))
plt.plot(GLAT)
plt.figure(figsize=(20,10))
Source_Name = Data[1].data['Source_Name']   

TB = np.array([GLON,GLAT,Flux])
#plot
#Check again
#plot 3D of Gammar rays to check graph
plt.figure(figsize=(20,10))
ax = plt.axes(projection='3d')
ax.plot_trisurf(TB[0], TB[1], TB[2],
                cmap='viridis', edgecolor='none');
ax.set_title('3D Gammar rays flux of the Milky Way Galaxy');
ax.set_xlabel('Longitude [Degree]')
ax.set_ylabel('Latitude [Degree]')
ax.set_zlabel('Flux (counts/cm^2/s/sr)');
plt.show()

#Full Data. prepare to export Mathematica
TB_Data = np.transpose(TB)

np.savetxt("new-fermi-allsky-000.5-001.0GeV-fwhm000-0256-nopsc.fits", TB_Data)

#change x to -x
Newest_LONGITUDE = np.transpose(New_LONGITUDE)*-1
TBT = np.array([Newest_LONGITUDE,LATITUDE,Flux])
TBT_Data = np.transpose(TBT)

np.savetxt("newest-fermi-allsky-000.5-001.0GeV-fwhm000-0256-nopsc.fits", TBT_Data)