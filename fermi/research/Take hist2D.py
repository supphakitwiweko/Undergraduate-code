# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:43:01 2021

@author: Supphakit Wiweko
"""

import os
import numpy as np
import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import scipy.interpolate
import scipy.stats 
from scipy.interpolate import make_interp_spline, BSpline,interp1d
from scipy.optimize import curve_fit

#plt.style.use('dark_background')
plt.rcParams.update({"figure.facecolor" : '#0f0f0f00'})
plt.style.use('seaborn')

#Add
catalog = fits.open('C:\\Users\\supph\\Documents\\Research\\gll_psc_v22.fit')
#T = Table(catalog[1].data).show_in_browser()
t = Table(catalog[1].data)

#change old to new longitude
#Select
new_glon = np.logical_and(t['GLON'] >=52, t['GLON']<=55)
new_tg = t[new_glon]
new_glat = np.logical_and(new_tg['GLAT'] >= -1.2, new_tg['GLAT'] <=1.8)
new_t = new_tg[new_glat]


fig, ax = plt.subplots(figsize=(30,20))
ax.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'],facecolor='none', edgecolor = 'y' , s=1000)
for i, txt in enumerate(new_t['Source_Name']):           
    ax.annotate(txt, (new_t['GLON'][i]-0.05, new_t['GLAT'][i]+0.05),fontsize = 20)
    
#open map
mapp = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210510051548F357373F73_PH00.fits')
t_mapp = Table(mapp[1].data)
#t_mapp.show_in_browser()

#a day
mapp_day = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210510051548F357373F73_PH00.fits')
t_mapp_day = Table(mapp_day[1].data)
#t_mapp.show_in_browser()

# a week
mapp_week = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210512114438F357373F94_PH00.fits')
t_mapp_week = Table(mapp_week[1].data)

#1 month
mapp_month = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210512115311F357373F34_PH00.fits')
t_mapp_month = Table(mapp_month[1].data)

# 2 months
mapp_month2 = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210525084544F357373F42_PH00.fits')
t_mapp_month2 = Table(mapp_month2[1].data)

# 3 months
mapp_month3 = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210525085025F357373F67_PH00.fits')
t_mapp_month3 = Table(mapp_month3[1].data)

# 4 months
mapp_month4 = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210525085206F357373F88_PH00.fits')
t_mapp_month4 = Table(mapp_month4[1].data)

# 5 months
mapp_5months0 = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210525085307F357373F94_PH00.fits')
mapp_5months1 = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210525085307F357373F94_PH01.fits')
t_mapp_5months0 = Table(mapp_5months0[1].data)
t_mapp_5months1 = Table(mapp_5months1[1].data)

# 6 months
mapp_6months0 = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210512115543F357373F61_PH00.fits')
mapp_6months1 = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210512120358F357373F94_PH01.fits')
t_mapp_6months0 = Table(mapp_6months0[1].data)
t_mapp_6months1 = Table(mapp_6months1[1].data)

# 7 months
mapp_7months0 = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210525085431F357373F54_PH00.fits')
mapp_7months1 = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210525085431F357373F54_PH01.fits')
t_mapp_7months0 = Table(mapp_7months0[1].data)
t_mapp_7months1 = Table(mapp_7months1[1].data)

# 8 months
mapp_8months0 = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210525085624F357373F18_PH00.fits')
mapp_8months1 = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210525085624F357373F18_PH01.fits')
t_mapp_8months0 = Table(mapp_8months0[1].data)
t_mapp_8months1 = Table(mapp_8months1[1].data)

# 9 months
mapp_9months0 = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210525085731F357373F97_PH00.fits')
mapp_9months1 = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210525085731F357373F97_PH01.fits')
mapp_9months2 = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210525085731F357373F97_PH02.fits')
t_mapp_9months0 = Table(mapp_9months0[1].data)
t_mapp_9months1 = Table(mapp_9months1[1].data)
t_mapp_9months2 = Table(mapp_9months2[1].data)

# 10 months
mapp_10months0 = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210525085851F357373F57_PH00.fits')
mapp_10months1 = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210525085851F357373F57_PH01.fits')
mapp_10months2 = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210525085851F357373F57_PH02.fits')
t_mapp_10months0 = Table(mapp_10months0[1].data)
t_mapp_10months1 = Table(mapp_10months1[1].data)
t_mapp_10months2 = Table(mapp_10months2[1].data)

# 11 months
mapp_11months0 = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210525090025F357373F46_PH00.fits')
mapp_11months1 = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210525090025F357373F46_PH01.fits')
mapp_11months2 = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210525090025F357373F46_PH02.fits')
t_mapp_11months0 = Table(mapp_11months0[1].data)
t_mapp_11months1 = Table(mapp_11months1[1].data)
t_mapp_11months2 = Table(mapp_11months2[1].data)

# 12 months

mapp_year0 = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210512115908F357373F22_PH00.fits')
mapp_year1 = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210512115908F357373F22_PH01.fits')
mapp_year2 = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210512115908F357373F22_PH02.fits')
mapp_year3 = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210512115908F357373F22_PH03.fits')

t_mapp_year0 = Table(mapp_year0[1].data)
t_mapp_year1 = Table(mapp_year1[1].data)
t_mapp_year2 = Table(mapp_year2[1].data)
t_mapp_year3 = Table(mapp_year3[1].data)

#mapp_ = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210527050945F357373F51_PH00.fits')
#t_mapp_ = Table(mapp_[1].data)

mapp_ = fits.open('C:\\Users\\supph\\Documents\\Research\\Photon database\\L210527051530F357373F19_PH00.fits')
t_mapp_ = Table(mapp_[1].data)
#show count
#plt

name = ('A day','A week','A month', 'Six months', 'A year')
number =(len(t_mapp_day),len(t_mapp_week),len(t_mapp_month),
         len(t_mapp_month2),len(t_mapp_month3),len(t_mapp_month4),
         len(t_mapp_5months0)+len(t_mapp_5months1),
         len(t_mapp_6months0)+len(t_mapp_6months1),
         len(t_mapp_7months0)+len(t_mapp_7months1),
         len(t_mapp_8months0)+len(t_mapp_8months1),
         len(t_mapp_9months0)+len(t_mapp_9months1)+len(t_mapp_9months2),
         len(t_mapp_10months0)+len(t_mapp_10months1)+len(t_mapp_10months2),
         len(t_mapp_11months0)+len(t_mapp_11months1)+len(t_mapp_11months2),
         len(t_mapp_year0)+len(t_mapp_year1)+len(t_mapp_year2)+len(t_mapp_year3))

day =(1,7,30,61,89,120,151,181,212,242,273,304,334,365)
daynew = np.linspace(1, 365, 365) 
spl = make_interp_spline(day, number, k=3)
number_smooth = spl(daynew)

#plot 2 D
fig , ax = plt.subplots(figsize=(10,5))
ax.plot(daynew, number_smooth)
ax.set_xlabel("Time (days)",fontsize=15)
ax.set_ylabel("Count of photons (counts)",fontsize=15)
fig.suptitle('Observation dates',fontsize = 25)
plt.show()

#-----fit curve----#
fig , ax = plt.subplots(figsize=(15,10))

def objective(day, a, b, c ):
	return (a * day) + (b * day**2)  + (c * day**3)
popt, _ = curve_fit(objective, day, number)
a, b, c = popt
x_line = np.arange(np.min(day), 366,1)
# calculate the output for the range
y_line = objective(x_line, a, b, c)
# create a line plot for the mapping function
ax.plot(x_line, y_line,linewidth=2, label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt),fillstyle='bottom')
ax.scatter(day, number,color='red', label='data')
ax.plot(daynew, number_smooth,'k--',label='interpolate',linewidth=1)
ax.tick_params(labelsize=20) 
ax.set_xlabel('Time (days)',fontsize=25)
ax.set_ylabel('Count of photons (counts)',fontsize=25)
fig.suptitle('Observation dates',fontsize=25)
plt.legend(loc='lower right',fontsize=20)
plt.show()



#-----bar plot----------------------
fig,ax = plt.subplots(figsize = (20, 10))

# creating the bar plot
ax.bar(name[0], number[0], color ='red')
ax.bar(name[1], number[1], color ='yellow')
ax.bar(name[2], number[2], color ='orange')
ax.bar(name[3], number[3], color ='g')
ax.bar(name[4], number[4], color ='blue')
for i in range(len(number)):
    
    ax.text(x = name[i], y = number[i]+500, s = number[i],fontsize=25)

ax.grid(color='black',alpha=1000)
ax.set_xlabel("Time",fontsize=20)
ax.set_ylabel("Number of photon",fontsize=20)
plt.title("Number of photon each the time",fontsize=20)
plt.show()


#make histrogram2D
#แบบแยกกกกก
bin = 20
#data 
#in a day

x = t_mapp_['L']
y = t_mapp_['B']
z = t_mapp_['ENERGY']

#count
fig, ax = plt.subplots(figsize=(40,30))
count = ax.hist2d(x, y, cmap='inferno',bins=bin)
ax.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'],edgecolor = 'white' , s=700,facecolor='none')
for i, txt in enumerate(new_t['Source_Name']):           
    ax.annotate(txt, (new_t['GLON'][i]+0.35, new_t['GLAT'][i]+0.05),fontsize=50,color='white')
ax.set_xlabel("Galactic Longitude (deg)",fontsize=60)
ax.set_ylabel("Galactic Latitude (deg)",fontsize=60)
ax.set_xlim(53.5+1.5, 53.5-1.5)
ax.set_ylim(0.3-1.5, 0.3+1.5)
ax.grid(alpha=0.5)

cbar = fig.colorbar(count[3], ax=ax)
cbar.set_label("Energy ( MeV )",fontsize=60) 
cbar.ax.tick_params(labelsize=40) 
fig.suptitle('Count in bins, bins = 50',fontsize = 100)

#Total energy
fig, ax = plt.subplots(figsize=(40,30))
total = ax.hist2d(x, y, weights=(z), cmap='inferno',bins=bin)
ax.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'],edgecolor = 'white' , s=700,facecolor='none')
for i, txt in enumerate(new_t['Source_Name']):           
    ax.annotate(txt, (new_t['GLON'][i]+0.2, new_t['GLAT'][i]+0.05),fontsize=50,color='white')
ax.set_xlabel("Galactic Longitude (deg)",fontsize=60)
ax.set_ylabel("Galactic Latitude (deg)",fontsize=60)
ax.set_xlim(53.5+1.5, 53.5-1.5)
ax.set_ylim(0.3-1.5, 0.3+1.5)
ax.grid(alpha=0.5)

cbar = fig.colorbar(count[3], ax=ax)
cbar.set_label("Energy ( MeV )",fontsize=60) 
cbar.ax.tick_params(labelsize=40) 
fig.suptitle('total, bins = 20',fontsize = 100)


#average
fig, ax = plt.subplots(figsize=(40,30))

H, xedges, yedges = np.histogram2d(x, y, bins = bin, weights = z)
H_counts, xedges, yedges = np.histogram2d(x, y, bins = bin) 
H = H/H_counts
average = ax.imshow(H.T, origin='lower',  cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
ax.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'],edgecolor = 'white' , s=700,facecolor='none')
for i, txt in enumerate(new_t['Source_Name']):           
    ax.annotate(txt, (new_t['GLON'][i]+0.2, new_t['GLAT'][i]+0.05),fontsize=50,color='white')
    
ax.set_xlabel("Galactic Longitude (deg)",fontsize=60)
ax.set_ylabel("Galactic Latitude (deg)",fontsize=60)
ax.set_xlim(53.5+1.5, 53.5-1.5)
ax.set_ylim(0.3-1.5, 0.3+1.5)
ax.grid(alpha=0.5)
cbar = fig.colorbar(count[3], ax=ax)
cbar.set_label("Energy ( MeV )",fontsize=60) 
cbar.ax.tick_params(labelsize=40) 
fig.suptitle('Count in bins, bins = 50',fontsize = 100)


#try plot to powerpoint



#-----------------------average plot-------------#
#data


x1 = t_mapp_year0['L']
x2 = t_mapp_year1['L']
x3 = t_mapp_year2['L']
x4 = t_mapp_year3['L']

x = np.concatenate((x1,x2,x3,x4))

#y
y1 = t_mapp_year0['B']
y2 = t_mapp_year1['B']
y3 = t_mapp_year2['B']
y4 = t_mapp_year3['B']
y = np.concatenate((y1,y2,y3,y4))

#z
z1 = t_mapp_year0['ENERGY']
z2 = t_mapp_year1['ENERGY']
z3 = t_mapp_year2['ENERGY']
z4 = t_mapp_year3['ENERGY']
z = np.concatenate((z1,z2,z3,z4))

#----------------------------------
x1 = t_mapp_6months0['L']
x2 = t_mapp_6months1['L']
x = np.concatenate((x1,x2))

#y
y1 = t_mapp_6months0['B']
y2 = t_mapp_6months1['B']
y = np.concatenate((y1,y2))

#z
z1 = t_mapp_6months0['ENERGY']
z2 = t_mapp_6months1['ENERGY']
z = np.concatenate((z1,z2))

#-------------------------------------


x = t_mapp_month3['L']
y = t_mapp_month3['B']
z = t_mapp_month3['ENERGY']

#plot

bin = 20
fig, ax = plt.subplots(figsize=(30,20))


H, xedges, yedges = np.histogram2d(x, y, bins = bin, weights = z)
H_counts, xedges, yedges = np.histogram2d(x, y, bins = bin) 
H = H/H_counts
average = ax.imshow(H.T, origin='lower',  cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

ax.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'], edgecolor = 'white' , s=300,facecolor='none')
for i, txt in enumerate(new_t['Source_Name']):           
    ax.annotate(txt, (new_t['GLON'][i]+0.2, new_t['GLAT'][i]+0.05),fontsize=30,color='white')
ax.set_xlabel("Galactic Longitude (deg)",fontsize=50)
ax.set_ylabel("Galactic Latitude (deg)",fontsize=50)
ax.set_xlim(53.5+1.5, 53.5-1.5)
ax.set_ylim(0.3-1.5, 0.3+1.5)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
ax.grid(alpha=0.5)
cbar = fig.colorbar(average, ax=ax)
cbar.set_label("Energy ( MeV )",fontsize=50) 
cbar.ax.tick_params(labelsize=40)  
fig.suptitle('Average energy in bins, bins = 20',fontsize = 50, horizontalalignment = 'center')


bin = 100
fig, ax = plt.subplots(figsize=(30,20))


H, xedges, yedges = np.histogram2d(x, y, bins = bin, weights = z)
H_counts, xedges, yedges = np.histogram2d(x, y, bins = bin) 
H = H/H_counts
average = ax.imshow(H.T, origin='lower',  cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

ax.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'], edgecolor = 'white' , s=300,facecolor='none')
for i, txt in enumerate(new_t['Source_Name']):           
    ax.annotate(txt, (new_t['GLON'][i]+0.2, new_t['GLAT'][i]+0.05),fontsize=15,color='white')
    
ax.set_xlabel("Galactic Longitude (deg)",fontsize=50)
ax.set_ylabel("Galactic Latitude (deg)",fontsize=50)
ax.set_xlim(53.5+1.5, 53.5-1.5)
ax.set_ylim(0.3-1.5, 0.3+1.5)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
ax.grid(alpha=0.5)
cbar = fig.colorbar(average, ax=ax)
cbar.set_label("Energy ( MeV )",fontsize=50) 
cbar.ax.tick_params(labelsize=40)  
fig.suptitle('Average energy in bins, bins = 100',fontsize = 50, horizontalalignment = 'center')


bin = 200
fig, ax = plt.subplots(figsize=(30,20))


H, xedges, yedges = np.histogram2d(x, y, bins = bin, weights = z)
H_counts, xedges, yedges = np.histogram2d(x, y, bins = bin) 
H = H/H_counts
average = ax.imshow(H.T, origin='lower',  cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

ax.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'], edgecolor = 'white' , s=300,facecolor='none')
for i, txt in enumerate(new_t['Source_Name']):           
    ax.annotate(txt, (new_t['GLON'][i]+0.2, new_t['GLAT'][i]+0.05),fontsize=15,color='white')
    
ax.set_xlabel("Galactic Longitude (deg)",fontsize=50)
ax.set_ylabel("Galactic Latitude (deg)",fontsize=50)
ax.set_xlim(53.5+1.5, 53.5-1.5)
ax.set_ylim(0.3-1.5, 0.3+1.5)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)

ax.grid(alpha=0.5)
cbar = fig.colorbar(average, ax=ax)
cbar.set_label("Energy ( MeV )",fontsize=50) 
cbar.ax.tick_params(labelsize=40)  
fig.suptitle('Average energy in bins, bins = 200',fontsize = 50, horizontalalignment = 'center')


#-----------------------average plot with count-------------#
#data


x1 = t_mapp_year0['L']
x2 = t_mapp_year1['L']
x3 = t_mapp_year2['L']
x4 = t_mapp_year3['L']

x = np.concatenate((x1,x2,x3,x4))

#y
y1 = t_mapp_year0['B']
y2 = t_mapp_year1['B']
y3 = t_mapp_year2['B']
y4 = t_mapp_year3['B']
y = np.concatenate((y1,y2,y3,y4))

#z
z1 = t_mapp_year0['ENERGY']
z2 = t_mapp_year1['ENERGY']
z3 = t_mapp_year2['ENERGY']
z4 = t_mapp_year3['ENERGY']
z = np.concatenate((z1,z2,z3,z4))

#----------------------------------
x1 = t_mapp_6months0['L']
x2 = t_mapp_6months1['L']
x = np.concatenate((x1,x2))

#y
y1 = t_mapp_6months0['B']
y2 = t_mapp_6months1['B']
y = np.concatenate((y1,y2))

#z
z1 = t_mapp_6months0['ENERGY']
z2 = t_mapp_6months1['ENERGY']
z = np.concatenate((z1,z2))

#-------------------------------------


x = t_mapp_month['L']
y = t_mapp_month['B']
z = t_mapp_month['ENERGY']

#
bin = 200
#count
fig, ax = plt.subplots(figsize=(30,20))

count, xedges, yedges = np.histogram2d(x, y, bins = bin) 
count = ax.imshow(count.T, origin='lower',  cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
ax.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'],edgecolor = 'white' , s=700,facecolor='none')
for i, txt in enumerate(new_t['Source_Name']):           
    ax.annotate(txt, (new_t['GLON'][i]+0.35, new_t['GLAT'][i]+0.05),fontsize=30,color='white')

ax.set_xlabel("Galactic Longitude (deg)",fontsize=60)
ax.set_ylabel("Galactic Latitude (deg)",fontsize=60)
ax.set_xlim(53.5+1.5, 53.5-1.5)
ax.set_ylim(0.3-1.5, 0.3+1.5)
ax.grid(alpha=0.5)
ax.tick_params(axis="x", labelsize=40)
ax.tick_params(axis="y", labelsize=40)
cbar = fig.colorbar(count, ax=ax)
cbar.set_label("counts in bins ( counts )",fontsize=60) 
cbar.ax.tick_params(labelsize=40) 
fig.suptitle('Count in bins, bins = 200',fontsize = 100)

#average
fig, ax = plt.subplots(figsize=(30,20))
H, xedges, yedges = np.histogram2d(x, y, bins = bin, weights = z)
H_counts, xedges, yedges = np.histogram2d(x, y, bins = bin) 
H = H/H_counts
average = ax.imshow(H.T, origin='lower',  cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

ax.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'], edgecolor = 'white' , s=300,facecolor='none')
for i, txt in enumerate(new_t['Source_Name']):           
    ax.annotate(txt, (new_t['GLON'][i]+0.2, new_t['GLAT'][i]+0.05),fontsize=30,color='white')
    
ax.set_xlabel("Galactic Longitude (deg)",fontsize=60)
ax.set_ylabel("Galactic Latitude (deg)",fontsize=60)
ax.set_xlim(53.5+1.5, 53.5-1.5)
ax.set_ylim(0.3-1.5, 0.3+1.5)
ax.grid(alpha=0.5)
ax.tick_params(axis="x", labelsize=40)
ax.tick_params(axis="y", labelsize=40)
cbar = fig.colorbar(average, ax=ax)
cbar.set_label("Energy ( MeV )",fontsize=50) 
cbar.ax.tick_params(labelsize=40)  
fig.suptitle('Average energy in bins, bins = 200',fontsize = 100, horizontalalignment = 'center')

