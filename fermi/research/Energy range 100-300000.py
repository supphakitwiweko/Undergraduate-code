# -*- coding: utf-8 -*-
"""
Created on Fri May 28 18:50:12 2021

@author: Supphakit Wiweko
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import make_interp_spline, BSpline,interp1d
from scipy.optimize import curve_fit
#make plot style
plt.style.use('seaborn')
#plt.rcParams.update({"figure.facecolor" : '#0f0f0f00'})
#get catalog
catalog = fits.open('C:\\Users\\supph\\Documents\\Research\\gll_psc_v22.fit')
#T = Table(catalog[1].data).show_in_browser()
t = Table(catalog[1].data)
new_glon = np.logical_and(t['GLON'] >=52, t['GLON']<=55)
new_tg = t[new_glon]
new_glat = np.logical_and(new_tg['GLAT'] >= -1.2, new_tg['GLAT'] <=1.8)
new_t = new_tg[new_glat]


#open map #energy range are 100-300,000 MeV
#a day
mapp_day = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530023002F357373F98_PH00.fits')
t_mapp_day = Table(mapp_day[1].data)
#t_mapp.show_in_browser()

# a week
mapp_week = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530022913F357373F73_PH00.fits')
t_mapp_week = Table(mapp_week[1].data)

#1 month
mapp_month = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530013160F357373F53_PH00.fits')
t_mapp_month = Table(mapp_month[1].data)

# 2 months
mapp_month2 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530013251F357373F52_PH00.fits')
t_mapp_month2 = Table(mapp_month2[1].data)

# 3 months
mapp_month3 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530013759F357373F94_PH00.fits')
t_mapp_month3 = Table(mapp_month3[1].data)

# 4 months
mapp_month4 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530013535F357373F30_PH00.fits')
t_mapp_month4 = Table(mapp_month4[1].data)

# 5 months
mapp_5months0 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530014316F357373F84_PH00.fits')

t_mapp_5months0 = Table(mapp_5months0[1].data)


# 6 months
mapp_6months0 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530014423F357373F63_PH00.fits')
mapp_6months1 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530014423F357373F63_PH01.fits')
t_mapp_6months0 = Table(mapp_6months0[1].data)
t_mapp_6months1 = Table(mapp_6months1[1].data)

# 7 months
mapp_7months0 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530014613F357373F80_PH00.fits')
mapp_7months1 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530014613F357373F80_PH01.fits')
t_mapp_7months0 = Table(mapp_7months0[1].data)
t_mapp_7months1 = Table(mapp_7months1[1].data)

# 8 months
mapp_8months0 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530014935F357373F54_PH00.fits')

t_mapp_8months0 = Table(mapp_8months0[1].data)


# 9 months
mapp_9months0 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530015307F357373F00_PH00.fits')
mapp_9months1 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530015307F357373F00_PH01.fits')
mapp_9months2 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530015307F357373F00_PH02.fits')
t_mapp_9months0 = Table(mapp_9months0[1].data)
t_mapp_9months1 = Table(mapp_9months1[1].data)
t_mapp_9months2 = Table(mapp_9months2[1].data)

# 10 months
mapp_10months0 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530015450F357373F63_PH00.fits')
mapp_10months1 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530015450F357373F63_PH01.fits')
mapp_10months2 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530015450F357373F63_PH02.fits')
t_mapp_10months0 = Table(mapp_10months0[1].data)
t_mapp_10months1 = Table(mapp_10months1[1].data)
t_mapp_10months2 = Table(mapp_10months2[1].data)

# 11 months
mapp_11months0 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530015922F357373F73_PH00.fits')
mapp_11months1 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530015922F357373F73_PH01.fits')
mapp_11months2 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530015922F357373F73_PH02.fits')
t_mapp_11months0 = Table(mapp_11months0[1].data)
t_mapp_11months1 = Table(mapp_11months1[1].data)
t_mapp_11months2 = Table(mapp_11months2[1].data)

# 12 months

mapp_year0 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530020136F357373F75_PH00.fits')
mapp_year1 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530020136F357373F75_PH01.fits')
mapp_year2 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530020136F357373F75_PH02.fits')
mapp_year3 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530020136F357373F75_PH03.fits')

t_mapp_year0 = Table(mapp_year0[1].data)
t_mapp_year1 = Table(mapp_year1[1].data)
t_mapp_year2 = Table(mapp_year2[1].data)
t_mapp_year3 = Table(mapp_year3[1].data)

#3 year
mapp_3year0 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530020315F357373F57_PH00.fits')
mapp_3year1 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530020315F357373F57_PH01.fits')
mapp_3year2 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530020315F357373F57_PH02.fits')
mapp_3year3 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530020315F357373F57_PH03.fits')
mapp_3year4 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530020315F357373F57_PH04.fits')
mapp_3year5 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530020315F357373F57_PH05.fits')
mapp_3year6 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530020315F357373F57_PH06.fits')
mapp_3year7 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530020315F357373F57_PH07.fits')
mapp_3year8 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530020315F357373F57_PH08.fits')

t_mapp_3year0 = Table(mapp_3year0[1].data)
t_mapp_3year1 = Table(mapp_3year1[1].data)
t_mapp_3year2 = Table(mapp_3year2[1].data)
t_mapp_3year3 = Table(mapp_3year3[1].data)
t_mapp_3year4 = Table(mapp_3year4[1].data)
t_mapp_3year5 = Table(mapp_3year5[1].data)
t_mapp_3year6 = Table(mapp_3year6[1].data)
t_mapp_3year7 = Table(mapp_3year7[1].data)
t_mapp_3year8 = Table(mapp_3year8[1].data)


#5 year
mapp_5year0 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530021058F357373F33_PH00.fits')
mapp_5year1 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530021058F357373F33_PH01.fits')
mapp_5year2 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530021058F357373F33_PH02.fits')
mapp_5year3 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530021058F357373F33_PH03.fits')
mapp_5year4 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530021058F357373F33_PH04.fits')
mapp_5year5 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530021058F357373F33_PH05.fits')
mapp_5year6 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530021058F357373F33_PH06.fits')
mapp_5year7 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530021058F357373F33_PH07.fits')
mapp_5year8 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530021058F357373F33_PH08.fits')

t_mapp_5year0 = Table(mapp_5year0[1].data)
t_mapp_5year1 = Table(mapp_5year1[1].data)
t_mapp_5year2 = Table(mapp_5year2[1].data)
t_mapp_5year3 = Table(mapp_5year3[1].data)
t_mapp_5year4 = Table(mapp_5year4[1].data)
t_mapp_5year5 = Table(mapp_5year5[1].data)
t_mapp_5year6 = Table(mapp_5year6[1].data)
t_mapp_5year7 = Table(mapp_5year7[1].data)
t_mapp_5year8 = Table(mapp_5year8[1].data)

#8 year
mapp_8year0 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530021644F357373F33_PH00.fits')
mapp_8year1 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530021644F357373F33_PH01.fits')
mapp_8year2 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530021644F357373F33_PH02.fits')
mapp_8year3 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530021644F357373F33_PH03.fits')
mapp_8year4 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530021644F357373F33_PH04.fits')
mapp_8year5 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530021644F357373F33_PH05.fits')
mapp_8year6 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530021644F357373F33_PH06.fits')
mapp_8year7 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530021644F357373F33_PH07.fits')
mapp_8year8 = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530021644F357373F33_PH08.fits')


t_mapp_8year0 = Table(mapp_8year0[1].data)
t_mapp_8year1 = Table(mapp_8year1[1].data)
t_mapp_8year2 = Table(mapp_8year2[1].data)
t_mapp_8year3 = Table(mapp_8year3[1].data)
t_mapp_8year4 = Table(mapp_8year4[1].data)
t_mapp_8year5 = Table(mapp_8year5[1].data)
t_mapp_8year6 = Table(mapp_8year6[1].data)
t_mapp_8year7 = Table(mapp_8year7[1].data)
t_mapp_8year8 = Table(mapp_8year8[1].data)

#### Histogram show each number on time

number =(len(t_mapp_day),len(t_mapp_week),len(t_mapp_month),
         len(t_mapp_month2),len(t_mapp_month3),len(t_mapp_month4),
         len(t_mapp_5months0),
         len(t_mapp_6months0)+len(t_mapp_6months1),
         len(t_mapp_7months0)+len(t_mapp_7months1),
         len(t_mapp_8months0),
         len(t_mapp_9months0)+len(t_mapp_9months1)+len(t_mapp_9months2),
         len(t_mapp_10months0)+len(t_mapp_10months1)+len(t_mapp_10months2),
         len(t_mapp_11months0)+len(t_mapp_11months1)+len(t_mapp_11months2),
         len(t_mapp_year0)+len(t_mapp_year1)+len(t_mapp_year2)+len(t_mapp_year3),
         len(t_mapp_3year0)+len(t_mapp_3year1)+len(t_mapp_3year2)+len(t_mapp_3year3)+len(t_mapp_3year4)+len(t_mapp_3year5)+len(t_mapp_3year6)+len(t_mapp_3year7)+len(t_mapp_3year8),
         len(t_mapp_5year0)+len(t_mapp_5year1)+len(t_mapp_5year2)+len(t_mapp_5year3)+len(t_mapp_5year4)+len(t_mapp_5year5)+len(t_mapp_5year6)+len(t_mapp_5year7)+len(t_mapp_5year8),
         len(t_mapp_8year0)+len(t_mapp_8year1)+len(t_mapp_8year2)+len(t_mapp_8year3)+len(t_mapp_8year4)+len(t_mapp_8year5)+len(t_mapp_8year6)+len(t_mapp_8year7)+len(t_mapp_8year8)
         )
day =(1,7,30,61,89,120,151,181,212,242,273,304,334,365,1103,1833,2922)
#day = (1,7,30,61,89,120,151,181,212,242,273,304,334,365)
#make data smooth
daynew = np.linspace(1, 2922, 2922) 
spl = make_interp_spline(day, number, k=3)
number_smooth = spl(daynew)

#plot 2 D
fig , ax = plt.subplots(figsize=(10,5))
ax.plot(daynew, number_smooth)
ax.scatter(day, number,color='red', label='data')
ax.set_xlabel("Time (days)",fontsize=15)
ax.set_ylabel("Count of photons (counts)",fontsize=15)
fig.suptitle('Observation dates',fontsize = 25)
plt.show()

#-----fit curve----#


def linear(day, a):
	return (a * day)

def parabola(day,a ,b):
    return(a*day+b*day**2)

#def func(x, a, b):
#    return a * np.exp(x*b) 

popt1, pcov1 = curve_fit(linear, day, number,method='lm')
popt2, pcov2 = curve_fit(parabola, day, number,method='lm')
#popt3, pcov3 = curve_fit(func, day, number,method='lm')

a1 = popt1
a2,b2 = popt2
#a3,b3 = popt3
fig , ax = plt.subplots(figsize=(10,10))
x_line = np.arange(np.min(day), np.max(day)+1,1)
x_line1 = np.arange(np.min(day), np.max(day),1)
# calculate the output for the range
y_line1 = linear(x_line, a1)
y_line2 = parabola(x_line, a2,b2)
#y_line3 = func(x_line, a3,b3)
# create a line plot for the mapping function
ax.plot(x_line, y_line1,linewidth=2, label='%.5f * x ' % tuple(popt1),fillstyle='bottom',color='blue')
ax.plot(x_line, y_line2,linewidth=2, label='%.5f * x + %.5f * x^2' % tuple(popt2),fillstyle='bottom',color='green')
#ax.plot(x_line, y_line3,linewidth=2, label='fit: a=%5.3f, b=%5.3f' % tuple(popt3),fillstyle='bottom', color='blue')

ax.scatter(day, number,color='red', label='data')
ax.tick_params(labelsize=20) 
ax.set_xlabel('Time (days)',fontsize=25)
ax.set_ylabel('Count of photons (counts)',fontsize=25)
fig.suptitle('Observation dates',fontsize=25)
plt.legend(loc='lower right',fontsize=20)
#ax.set_yscale('log')
plt.show()



#Show 1-365 day
day = (1,7,30,61,89,120,151,181,212,242,273,304,334,365)
daynew = np.linspace(1,365 , 365)
x = np.arange(np.min(day), np.max(day)+1,1)

y_1 = linear(daynew, a1)
y_2 = parabola(daynew, a2 , b2)
number =(len(t_mapp_day),len(t_mapp_week),len(t_mapp_month),
         len(t_mapp_month2),len(t_mapp_month3),len(t_mapp_month4),
         len(t_mapp_5months0),
         len(t_mapp_6months0)+len(t_mapp_6months1),
         len(t_mapp_7months0)+len(t_mapp_7months1),
         len(t_mapp_8months0),
         len(t_mapp_9months0)+len(t_mapp_9months1)+len(t_mapp_9months2),
         len(t_mapp_10months0)+len(t_mapp_10months1)+len(t_mapp_10months2),
         len(t_mapp_11months0)+len(t_mapp_11months1)+len(t_mapp_11months2),
         len(t_mapp_year0)+len(t_mapp_year1)+len(t_mapp_year2)+len(t_mapp_year3))

fig , ax = plt.subplots(figsize=(10,10))
ax.plot(daynew, y_1,linewidth=2, label='%.5f * x ' % tuple(popt1),fillstyle='bottom',color='blue')
ax.plot(daynew, y_2,linewidth=2, label='%.5f * x + %.5f * x^2' % tuple(popt2),fillstyle='bottom',color='green')
ax.scatter(day, number,color='red', label='data')
ax.tick_params(labelsize=20) 
ax.set_xlabel('Time (days)',fontsize=25)
ax.set_ylabel('Count of photons (counts)',fontsize=25)
fig.suptitle('Observation dates',fontsize=25)
plt.legend(loc='lower right',fontsize=20)
plt.show()
#########################################


#-----------------------average plot with count-------------#
plt.style.use('dark_background')

bin = 50

#day
#change coordiante

new_x = np.logical_and(t_mapp_day['L'] >=52, t_mapp_day['L']<=55)
new_tx =  t_mapp_day[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate = new_tx[new_y]

#data

x = new_coordinate['L']
y = new_coordinate['B']
z = new_coordinate['ENERGY']

#count
fig, (ax, ax0) = plt.subplots(1,2,figsize=(40,20))

count, xedges, yedges = np.histogram2d(x, y, bins = bin) 
count = ax.imshow(count.T, origin='lower',  cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
ax.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'],edgecolor = 'white' , s=700,facecolor='none')
for i, txt in enumerate(new_t['Source_Name']):           
    ax.annotate(txt, (new_t['GLON'][i]+0.35, new_t['GLAT'][i]+0.05),fontsize=30,color='white')

ax.set_xlabel("Galactic Longitude (deg)",fontsize=40)
ax.set_ylabel("Galactic Latitude (deg)",fontsize=40)
ax.set_xlim(53.5+1.5, 53.5-1.5)
ax.set_ylim(0.3-1.5, 0.3+1.5)
ax.grid(alpha=0.5)
ax.tick_params(axis="x", labelsize=40)
ax.tick_params(axis="y", labelsize=40)
cbar = fig.colorbar(count, ax=ax)
cbar.set_label("counts in bins ( counts )",fontsize=50) 
cbar.ax.tick_params(labelsize=40) 

ax.set_title('Count in bins, bins = 50',fontsize = 60)

#average

H, xedges, yedges = np.histogram2d(x, y, bins = bin, weights = z)
H_counts, xedges, yedges = np.histogram2d(x, y, bins = bin) 
H = H/H_counts
average = ax0.imshow(H.T, origin='lower',  cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

ax0.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'], edgecolor = 'white' , s=700,facecolor='none')
for i, txt in enumerate(new_t['Source_Name']):           
    ax0.annotate(txt, (new_t['GLON'][i]+0.2, new_t['GLAT'][i]+0.05),fontsize=30,color='white')
    
ax0.set_xlabel("Galactic Longitude (deg)",fontsize=40)
ax0.set_ylabel("Galactic Latitude (deg)",fontsize=40)
ax0.set_xlim(53.5+1.5, 53.5-1.5)
ax0.set_ylim(0.3-1.5, 0.3+1.5)
ax0.grid(alpha=0.5)
ax0.tick_params(axis="x", labelsize=40)
ax0.tick_params(axis="y", labelsize=40)
cbar = fig.colorbar(average, ax=ax0)
cbar.set_label(" Average energy ( MeV )",fontsize=50) 
cbar.ax.tick_params(labelsize=40)  
ax0.set_title('Average energy in bins, bins = 50',fontsize = 60, horizontalalignment = 'center')

#-------------------------------------
#change coordiante

new_x = np.logical_and(t_mapp_week['L'] >=52, t_mapp_week['L']<=55)
new_tx =  t_mapp_week[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate = new_tx[new_y]

#data


x = new_coordinate['L']
y = new_coordinate['B']
z = new_coordinate['ENERGY']

#week

#count
fig, (ax, ax0) = plt.subplots(1,2,figsize=(40,20))

count, xedges, yedges = np.histogram2d(x, y, bins = bin) 
count = ax.imshow(count.T, origin='lower',  cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
ax.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'],edgecolor = 'white' , s=700,facecolor='none')
for i, txt in enumerate(new_t['Source_Name']):           
    ax.annotate(txt, (new_t['GLON'][i]+0.35, new_t['GLAT'][i]+0.05),fontsize=30,color='white')

ax.set_xlabel("Galactic Longitude (deg)",fontsize=40)
ax.set_ylabel("Galactic Latitude (deg)",fontsize=40)
ax.set_xlim(53.5+1.5, 53.5-1.5)
ax.set_ylim(0.3-1.5, 0.3+1.5)
ax.grid(alpha=0.5)
ax.tick_params(axis="x", labelsize=40)
ax.tick_params(axis="y", labelsize=40)
cbar = fig.colorbar(count, ax=ax)
cbar.set_label("counts in bins ( counts )",fontsize=50) 
cbar.ax.tick_params(labelsize=40) 

ax.set_title('Count in bins, bins = 50',fontsize = 60)

#average

H, xedges, yedges = np.histogram2d(x, y, bins = bin, weights = z)
H_counts, xedges, yedges = np.histogram2d(x, y, bins = bin) 
H = H/H_counts
average = ax0.imshow(H.T, origin='lower',  cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

ax0.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'], edgecolor = 'white' , s=700,facecolor='none')
for i, txt in enumerate(new_t['Source_Name']):           
    ax0.annotate(txt, (new_t['GLON'][i]+0.2, new_t['GLAT'][i]+0.05),fontsize=30,color='white')
    
ax0.set_xlabel("Galactic Longitude (deg)",fontsize=40)
ax0.set_ylabel("Galactic Latitude (deg)",fontsize=40)
ax0.set_xlim(53.5+1.5, 53.5-1.5)
ax0.set_ylim(0.3-1.5, 0.3+1.5)
ax0.grid(alpha=0.5)
ax0.tick_params(axis="x", labelsize=40)
ax0.tick_params(axis="y", labelsize=40)
cbar = fig.colorbar(average, ax=ax0)
cbar.set_label(" Average energy ( MeV )",fontsize=50) 
cbar.ax.tick_params(labelsize=40)  
ax0.set_title('Average energy in bins, bins = 50',fontsize = 60, horizontalalignment = 'center')

#-------------------------------------
#change coordiante

new_x = np.logical_and(t_mapp_month['L'] >=52, t_mapp_month['L']<=55)
new_tx =  t_mapp_month[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate = new_tx[new_y]

#data

x = new_coordinate['L']
y = new_coordinate['B']
z = new_coordinate['ENERGY']

#a month

#count
fig, (ax, ax0) = plt.subplots(1,2,figsize=(40,20))

count, xedges, yedges = np.histogram2d(x, y, bins = bin) 
count = ax.imshow(count.T, origin='lower',  cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
ax.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'],edgecolor = 'white' , s=700,facecolor='none')
for i, txt in enumerate(new_t['Source_Name']):           
    ax.annotate(txt, (new_t['GLON'][i]+0.35, new_t['GLAT'][i]+0.05),fontsize=30,color='white')

ax.set_xlabel("Galactic Longitude (deg)",fontsize=40)
ax.set_ylabel("Galactic Latitude (deg)",fontsize=40)
ax.set_xlim(53.5+1.5, 53.5-1.5)
ax.set_ylim(0.3-1.5, 0.3+1.5)
ax.grid(alpha=0.5)
ax.tick_params(axis="x", labelsize=40)
ax.tick_params(axis="y", labelsize=40)
cbar = fig.colorbar(count, ax=ax)
cbar.set_label("counts in bins ( counts )",fontsize=50) 
cbar.ax.tick_params(labelsize=40) 

ax.set_title('Count in bins, bins = 50',fontsize = 60)

#average

H, xedges, yedges = np.histogram2d(x, y, bins = bin, weights = z)
H_counts, xedges, yedges = np.histogram2d(x, y, bins = bin) 
H = H/H_counts
average = ax0.imshow(H.T, origin='lower',  cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

ax0.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'], edgecolor = 'white' , s=700,facecolor='none')
for i, txt in enumerate(new_t['Source_Name']):           
    ax0.annotate(txt, (new_t['GLON'][i]+0.2, new_t['GLAT'][i]+0.05),fontsize=30,color='white')
    
ax0.set_xlabel("Galactic Longitude (deg)",fontsize=40)
ax0.set_ylabel("Galactic Latitude (deg)",fontsize=40)
ax0.set_xlim(53.5+1.5, 53.5-1.5)
ax0.set_ylim(0.3-1.5, 0.3+1.5)
ax0.grid(alpha=0.5)
ax0.tick_params(axis="x", labelsize=40)
ax0.tick_params(axis="y", labelsize=40)
cbar = fig.colorbar(average, ax=ax0)
cbar.set_label(" Average energy ( MeV )",fontsize=50) 
cbar.ax.tick_params(labelsize=40)  
ax0.set_title('Average energy in bins, bins = 50',fontsize = 60, horizontalalignment = 'center')

#-------------------------------------
#change coordiante

new_x = np.logical_and(t_mapp_6months0['L'] >=52, t_mapp_6months0['L']<=55)
new_tx =  t_mapp_6months0[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate0 = new_tx[new_y]

new_x = np.logical_and(t_mapp_6months1['L'] >=52, t_mapp_6months1['L']<=55)
new_tx =  t_mapp_6months1[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate1 = new_tx[new_y]
#change coordiante

#data

x1 = new_coordinate0['L']
x2 = new_coordinate1['L']
x = np.concatenate((x1,x2))

#y
y1 = new_coordinate0['B']
y2 = new_coordinate1['B']
y = np.concatenate((y1,y2))

#z
z1 = new_coordinate0['ENERGY']
z2 = new_coordinate1['ENERGY']
z = np.concatenate((z1,z2))

#6 months

#count
fig, (ax, ax0) = plt.subplots(1,2,figsize=(40,20))

count, xedges, yedges = np.histogram2d(x, y, bins = bin) 
count = ax.imshow(count.T, origin='lower',  cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
ax.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'],edgecolor = 'white' , s=700,facecolor='none')
for i, txt in enumerate(new_t['Source_Name']):           
    ax.annotate(txt, (new_t['GLON'][i]+0.35, new_t['GLAT'][i]+0.05),fontsize=30,color='white')

ax.set_xlabel("Galactic Longitude (deg)",fontsize=40)
ax.set_ylabel("Galactic Latitude (deg)",fontsize=40)
ax.set_xlim(53.5+1.5, 53.5-1.5)
ax.set_ylim(0.3-1.5, 0.3+1.5)
ax.grid(alpha=0.5)
ax.tick_params(axis="x", labelsize=40)
ax.tick_params(axis="y", labelsize=40)
cbar = fig.colorbar(count, ax=ax)
cbar.set_label("counts in bins ( counts )",fontsize=50) 
cbar.ax.tick_params(labelsize=40) 

ax.set_title('Count in bins, bins = 50',fontsize = 60)

#average

H, xedges, yedges = np.histogram2d(x, y, bins = bin, weights = z)
H_counts, xedges, yedges = np.histogram2d(x, y, bins = bin) 
H = H/H_counts
average = ax0.imshow(H.T, origin='lower',  cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

ax0.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'], edgecolor = 'white' , s=700,facecolor='none')
for i, txt in enumerate(new_t['Source_Name']):           
    ax0.annotate(txt, (new_t['GLON'][i]+0.2, new_t['GLAT'][i]+0.05),fontsize=30,color='white')
    
ax0.set_xlabel("Galactic Longitude (deg)",fontsize=40)
ax0.set_ylabel("Galactic Latitude (deg)",fontsize=40)
ax0.set_xlim(53.5+1.5, 53.5-1.5)
ax0.set_ylim(0.3-1.5, 0.3+1.5)
ax0.grid(alpha=0.5)
ax0.tick_params(axis="x", labelsize=40)
ax0.tick_params(axis="y", labelsize=40)
cbar = fig.colorbar(average, ax=ax0)
cbar.set_label(" Average energy ( MeV )",fontsize=50) 
cbar.ax.tick_params(labelsize=40)  
ax0.set_title('Average energy in bins, bins = 50',fontsize = 60, horizontalalignment = 'center')

#-------------------------------------
#change coordiante

new_x = np.logical_and(t_mapp_year0['L'] >=52, t_mapp_year0['L']<=55)
new_tx =  t_mapp_year0[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate0 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_year1['L'] >=52,t_mapp_year1['L']<=55)
new_tx =  t_mapp_year1[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate1 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_year2['L'] >=52, t_mapp_year2['L']<=55)
new_tx = t_mapp_year2[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate2 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_year3['L'] >=52, t_mapp_year3['L']<=55)
new_tx = t_mapp_year3[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate3 = new_tx[new_y]

#data

x1 = new_coordinate0['L']
x2 = new_coordinate1['L']
x3 = new_coordinate2['L']
x4 = new_coordinate3['L']

x = np.concatenate((x1,x2,x3,x4))

#y
y1 = new_coordinate0['B']
y2 = new_coordinate1['B']
y3 = new_coordinate2['B']
y4 = new_coordinate3['B']
y = np.concatenate((y1,y2,y3,y4))

#z
z1 = new_coordinate0['ENERGY']
z2 = new_coordinate1['ENERGY']
z3 = new_coordinate2['ENERGY']
z4 = new_coordinate3['ENERGY']
z = np.concatenate((z1,z2,z3,z4))


#a year

#count
fig, (ax, ax0) = plt.subplots(1,2,figsize=(40,20))
count, xedges, yedges = np.histogram2d(x, y, bins = bin) 
count = ax.imshow(count.T, origin='lower',  cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
ax.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'],edgecolor = 'white' , s=700,facecolor='none')
for i, txt in enumerate(new_t['Source_Name']):           
    ax.annotate(txt, (new_t['GLON'][i]+0.35, new_t['GLAT'][i]+0.05),fontsize=30,color='white')
ax.set_xlabel("Galactic Longitude (deg)",fontsize=40)
ax.set_ylabel("Galactic Latitude (deg)",fontsize=40)
ax.set_xlim(53.5+1.5, 53.5-1.5)
ax.set_ylim(0.3-1.5, 0.3+1.5)
ax.grid(alpha=0.5)
ax.tick_params(axis="x", labelsize=40)
ax.tick_params(axis="y", labelsize=40)
cbar = fig.colorbar(count, ax=ax)
cbar.set_label("counts in bins ( counts )",fontsize=50) 
cbar.ax.tick_params(labelsize=40) 
ax.set_title('Count in bins, bins = 50',fontsize = 60)

#average

H, xedges, yedges = np.histogram2d(x, y, bins = bin, weights = z)
H_counts, xedges, yedges = np.histogram2d(x, y, bins = bin) 
H = H/H_counts
average = ax0.imshow(H.T, origin='lower',  cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

ax0.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'], edgecolor = 'white' , s=700,facecolor='none')
for i, txt in enumerate(new_t['Source_Name']):           
    ax0.annotate(txt, (new_t['GLON'][i]+0.2, new_t['GLAT'][i]+0.05),fontsize=30,color='white')
    
ax0.set_xlabel("Galactic Longitude (deg)",fontsize=40)
ax0.set_ylabel("Galactic Latitude (deg)",fontsize=40)
ax0.set_xlim(53.5+1.5, 53.5-1.5)
ax0.set_ylim(0.3-1.5, 0.3+1.5)
ax0.grid(alpha=0.5)
ax0.tick_params(axis="x", labelsize=40)
ax0.tick_params(axis="y", labelsize=40)
cbar = fig.colorbar(average, ax=ax0)
cbar.set_label(" Average energy ( MeV )",fontsize=50) 
cbar.ax.tick_params(labelsize=40)  
ax0.set_title('Average energy in bins, bins = 50',fontsize = 60, horizontalalignment = 'center')

#-------------------------------------
#change coordiante

new_x = np.logical_and(t_mapp_3year0['L'] >=52, t_mapp_3year0['L']<=55)
new_tx =  t_mapp_3year0[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate0 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_3year1['L'] >=52, t_mapp_3year1['L']<=55)
new_tx =  t_mapp_3year1[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate1 = new_tx[new_y]
#change coordiante

new_x = np.logical_and(t_mapp_3year2['L'] >=52, t_mapp_3year2['L']<=55)
new_tx =  t_mapp_3year2[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate2 = new_tx[new_y]
#change coordiante

new_x = np.logical_and(t_mapp_3year3['L'] >=52, t_mapp_3year3['L']<=55)
new_tx =  t_mapp_3year3[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate3 = new_tx[new_y]
#change coordiante

new_x = np.logical_and(t_mapp_3year4['L'] >=52, t_mapp_3year4['L']<=55)
new_tx =  t_mapp_3year4[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate4 = new_tx[new_y]
#change coordiante

new_x = np.logical_and(t_mapp_3year5['L'] >=52, t_mapp_3year5['L']<=55)
new_tx =  t_mapp_3year5[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate5 = new_tx[new_y]
#change coordiante

new_x = np.logical_and(t_mapp_3year6['L'] >=52, t_mapp_3year6['L']<=55)
new_tx =  t_mapp_3year6[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate6 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_3year7['L'] >=52, t_mapp_3year7['L']<=55)
new_tx =  t_mapp_3year7[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate7 = new_tx[new_y]
#change coordiante

new_x = np.logical_and(t_mapp_3year8['L'] >=52, t_mapp_3year8['L']<=55)
new_tx =  t_mapp_3year8[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate8 = new_tx[new_y]
#data

x1 = new_coordinate0['L']
x2 = new_coordinate1['L']
x3 = new_coordinate2['L']
x4 = new_coordinate3['L']
x5 = new_coordinate4['L']
x6 = new_coordinate5['L']
x7 = new_coordinate6['L']
x8 = new_coordinate7['L']
x9 = new_coordinate8['L']

x = np.concatenate((x1,x2,x3,x4,x5,x6,x7,x8,x9))

#y
y1 = new_coordinate0['B']
y2 = new_coordinate1['B']
y3 = new_coordinate2['B']
y4 = new_coordinate3['B']
y5 = new_coordinate4['B']
y6 = new_coordinate5['B']
y7 = new_coordinate6['B']
y8 = new_coordinate7['B']
y9 = new_coordinate8['B']

y = np.concatenate((y1,y2,y3,y4,y5,y6,y7,y8,y9))

#z
z1 = new_coordinate0['ENERGY']
z2 = new_coordinate1['ENERGY']
z3 = new_coordinate2['ENERGY']
z4 = new_coordinate3['ENERGY']
z5 = new_coordinate4['ENERGY']
z6 = new_coordinate5['ENERGY']
z7 = new_coordinate6['ENERGY']
z8 = new_coordinate7['ENERGY']
z9 = new_coordinate8['ENERGY']

z = np.concatenate((z1,z2,z3,z4,z5,z6,z7,z8,z9))

#3 years

#count
fig, (ax, ax0) = plt.subplots(1,2,figsize=(40,20))

count, xedges, yedges = np.histogram2d(x, y, bins = bin) 
count = ax.imshow(count.T, origin='lower',  cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
ax.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'],edgecolor = 'white' , s=700,facecolor='none')
for i, txt in enumerate(new_t['Source_Name']):           
    ax.annotate(txt, (new_t['GLON'][i]+0.35, new_t['GLAT'][i]+0.05),fontsize=30,color='white')

ax.set_xlabel("Galactic Longitude (deg)",fontsize=40)
ax.set_ylabel("Galactic Latitude (deg)",fontsize=40)
ax.set_xlim(53.5+1.5, 53.5-1.5)
ax.set_ylim(0.3-1.5, 0.3+1.5)
ax.grid(alpha=0.5)
ax.tick_params(axis="x", labelsize=40)
ax.tick_params(axis="y", labelsize=40)
cbar = fig.colorbar(count, ax=ax)
cbar.set_label("counts in bins ( counts )",fontsize=50) 
cbar.ax.tick_params(labelsize=40) 

ax.set_title('Count in bins, bins = 50',fontsize = 60)

#average

H, xedges, yedges = np.histogram2d(x, y, bins = bin, weights = z)
H_counts, xedges, yedges = np.histogram2d(x, y, bins = bin) 
H = H/H_counts
average = ax0.imshow(H.T, origin='lower',  cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

ax0.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'], edgecolor = 'white' , s=700,facecolor='none')
for i, txt in enumerate(new_t['Source_Name']):           
    ax0.annotate(txt, (new_t['GLON'][i]+0.2, new_t['GLAT'][i]+0.05),fontsize=30,color='white')
    
ax0.set_xlabel("Galactic Longitude (deg)",fontsize=40)
ax0.set_ylabel("Galactic Latitude (deg)",fontsize=40)
ax0.set_xlim(53.5+1.5, 53.5-1.5)
ax0.set_ylim(0.3-1.5, 0.3+1.5)
ax0.grid(alpha=0.5)
ax0.tick_params(axis="x", labelsize=40)
ax0.tick_params(axis="y", labelsize=40)
cbar = fig.colorbar(average, ax=ax0)
cbar.set_label(" Average energy ( MeV )",fontsize=50) 
cbar.ax.tick_params(labelsize=40)  
ax0.set_title('Average energy in bins, bins = 50',fontsize = 60, horizontalalignment = 'center')

#-------------------------------------
#change coordiante

new_x = np.logical_and(t_mapp_5year0['L'] >=52, t_mapp_5year0['L']<=55)
new_tx =  t_mapp_5year0[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate0 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_5year1['L'] >=52, t_mapp_5year1['L']<=55)
new_tx =  t_mapp_5year1[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate1 = new_tx[new_y]
#change coordiante

new_x = np.logical_and(t_mapp_5year2['L'] >=52, t_mapp_5year2['L']<=55)
new_tx =  t_mapp_5year2[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate2 = new_tx[new_y]
#change coordiante

new_x = np.logical_and(t_mapp_5year3['L'] >=52, t_mapp_5year3['L']<=55)
new_tx =  t_mapp_5year3[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate3 = new_tx[new_y]
#change coordiante

new_x = np.logical_and(t_mapp_5year4['L'] >=52, t_mapp_5year4['L']<=55)
new_tx =  t_mapp_5year4[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate4 = new_tx[new_y]
#change coordiante

new_x = np.logical_and(t_mapp_5year5['L'] >=52, t_mapp_5year5['L']<=55)
new_tx =  t_mapp_5year5[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate5 = new_tx[new_y]
#change coordiante

new_x = np.logical_and(t_mapp_5year6['L'] >=52, t_mapp_5year6['L']<=55)
new_tx =  t_mapp_5year6[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate6 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_5year7['L'] >=52, t_mapp_5year7['L']<=55)
new_tx =  t_mapp_5year7[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate7 = new_tx[new_y]
#change coordiante

new_x = np.logical_and(t_mapp_5year8['L'] >=52, t_mapp_5year8['L']<=55)
new_tx =  t_mapp_5year8[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate8 = new_tx[new_y]
#data

x1 = new_coordinate0['L']
x2 = new_coordinate1['L']
x3 = new_coordinate2['L']
x4 = new_coordinate3['L']
x5 = new_coordinate4['L']
x6 = new_coordinate5['L']
x7 = new_coordinate6['L']
x8 = new_coordinate7['L']
x9 = new_coordinate8['L']

x = np.concatenate((x1,x2,x3,x4,x5,x6,x7,x8,x9))

#y
y1 = new_coordinate0['B']
y2 = new_coordinate1['B']
y3 = new_coordinate2['B']
y4 = new_coordinate3['B']
y5 = new_coordinate4['B']
y6 = new_coordinate5['B']
y7 = new_coordinate6['B']
y8 = new_coordinate7['B']
y9 = new_coordinate8['B']

y = np.concatenate((y1,y2,y3,y4,y5,y6,y7,y8,y9))

#z
z1 = new_coordinate0['ENERGY']
z2 = new_coordinate1['ENERGY']
z3 = new_coordinate2['ENERGY']
z4 = new_coordinate3['ENERGY']
z5 = new_coordinate4['ENERGY']
z6 = new_coordinate5['ENERGY']
z7 = new_coordinate6['ENERGY']
z8 = new_coordinate7['ENERGY']
z9 = new_coordinate8['ENERGY']

z = np.concatenate((z1,z2,z3,z4,z5,z6,z7,z8,z9))

#5 years

#count
fig, (ax, ax0) = plt.subplots(1,2,figsize=(40,20))

count, xedges, yedges = np.histogram2d(x, y, bins = bin) 
count = ax.imshow(count.T, origin='lower',  cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
ax.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'],edgecolor = 'white' , s=700,facecolor='none')
for i, txt in enumerate(new_t['Source_Name']):           
    ax.annotate(txt, (new_t['GLON'][i]+0.35, new_t['GLAT'][i]+0.05),fontsize=30,color='white')

ax.set_xlabel("Galactic Longitude (deg)",fontsize=40)
ax.set_ylabel("Galactic Latitude (deg)",fontsize=40)
ax.set_xlim(53.5+1.5, 53.5-1.5)
ax.set_ylim(0.3-1.5, 0.3+1.5)
ax.grid(alpha=0.5)
ax.tick_params(axis="x", labelsize=40)
ax.tick_params(axis="y", labelsize=40)
cbar = fig.colorbar(count, ax=ax)
cbar.set_label("counts in bins ( counts )",fontsize=50) 
cbar.ax.tick_params(labelsize=40) 

ax.set_title('Count in bins, bins = 50',fontsize = 60)

#average

H, xedges, yedges = np.histogram2d(x, y, bins = bin, weights = z)
H_counts, xedges, yedges = np.histogram2d(x, y, bins = bin) 
H = H/H_counts
average = ax0.imshow(H.T, origin='lower',  cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

ax0.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'], edgecolor = 'white' , s=700,facecolor='none')
for i, txt in enumerate(new_t['Source_Name']):           
    ax0.annotate(txt, (new_t['GLON'][i]+0.2, new_t['GLAT'][i]+0.05),fontsize=30,color='white')
    
ax0.set_xlabel("Galactic Longitude (deg)",fontsize=40)
ax0.set_ylabel("Galactic Latitude (deg)",fontsize=40)
ax0.set_xlim(53.5+1.5, 53.5-1.5)
ax0.set_ylim(0.3-1.5, 0.3+1.5)
ax0.grid(alpha=0.5)
ax0.tick_params(axis="x", labelsize=40)
ax0.tick_params(axis="y", labelsize=40)
cbar = fig.colorbar(average, ax=ax0)
cbar.set_label(" Average energy ( MeV )",fontsize=50) 
cbar.ax.tick_params(labelsize=40)  
ax0.set_title('Average energy in bins, bins = 50',fontsize = 60, horizontalalignment = 'center')

#-------------------------------------
#change coordiante

new_x = np.logical_and(t_mapp_8year0['L'] >=52, t_mapp_8year0['L']<=55)
new_tx =  t_mapp_8year0[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate0 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year1['L'] >=52, t_mapp_8year1['L']<=55)
new_tx =  t_mapp_8year1[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate1 = new_tx[new_y]
#change coordiante

new_x = np.logical_and(t_mapp_8year2['L'] >=52, t_mapp_8year2['L']<=55)
new_tx =  t_mapp_8year2[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate2 = new_tx[new_y]
#change coordiante

new_x = np.logical_and(t_mapp_8year3['L'] >=52, t_mapp_8year3['L']<=55)
new_tx =  t_mapp_8year3[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate3 = new_tx[new_y]
#change coordiante

new_x = np.logical_and(t_mapp_8year4['L'] >=52, t_mapp_8year4['L']<=55)
new_tx =  t_mapp_8year4[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate4 = new_tx[new_y]
#change coordiante

new_x = np.logical_and(t_mapp_8year5['L'] >=52, t_mapp_8year5['L']<=55)
new_tx =  t_mapp_8year5[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate5 = new_tx[new_y]
#change coordiante

new_x = np.logical_and(t_mapp_8year6['L'] >=52, t_mapp_8year6['L']<=55)
new_tx =  t_mapp_8year6[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate6 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year7['L'] >=52, t_mapp_8year7['L']<=55)
new_tx =  t_mapp_8year7[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate7 = new_tx[new_y]
#change coordiante

new_x = np.logical_and(t_mapp_8year8['L'] >=52, t_mapp_8year8['L']<=55)
new_tx =  t_mapp_8year8[new_x]
new_y = np.logical_and(new_tx['B'] >= -1.2, new_tx['B'] <=1.8)
new_coordinate8 = new_tx[new_y]
#data

x1 = new_coordinate0['L']
x2 = new_coordinate1['L']
x3 = new_coordinate2['L']
x4 = new_coordinate3['L']
x5 = new_coordinate4['L']
x6 = new_coordinate5['L']
x7 = new_coordinate6['L']
x8 = new_coordinate7['L']
x9 = new_coordinate8['L']

x = np.concatenate((x1,x2,x3,x4,x5,x6,x7,x8,x9))

#y
y1 = new_coordinate0['B']
y2 = new_coordinate1['B']
y3 = new_coordinate2['B']
y4 = new_coordinate3['B']
y5 = new_coordinate4['B']
y6 = new_coordinate5['B']
y7 = new_coordinate6['B']
y8 = new_coordinate7['B']
y9 = new_coordinate8['B']

y = np.concatenate((y1,y2,y3,y4,y5,y6,y7,y8,y9))

#z
z1 = new_coordinate0['ENERGY']
z2 = new_coordinate1['ENERGY']
z3 = new_coordinate2['ENERGY']
z4 = new_coordinate3['ENERGY']
z5 = new_coordinate4['ENERGY']
z6 = new_coordinate5['ENERGY']
z7 = new_coordinate6['ENERGY']
z8 = new_coordinate7['ENERGY']
z9 = new_coordinate8['ENERGY']

z = np.concatenate((z1,z2,z3,z4,z5,z6,z7,z8,z9))


#8 years
bin = 2
#count
fig, (ax, ax0) = plt.subplots(1,2,figsize=(40,20))

count, xedges, yedges = np.histogram2d(x, y, bins = bin) 
count = ax.imshow(count.T, origin='lower',  cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
ax.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'],edgecolor = 'white' , s=700,facecolor='none')
for i, txt in enumerate(new_t['Source_Name']):           
    ax.annotate(txt, (new_t['GLON'][i]+0.35, new_t['GLAT'][i]+0.05),fontsize=30,color='white')

ax.set_xlabel("Galactic Longitude (deg)",fontsize=40)
ax.set_ylabel("Galactic Latitude (deg)",fontsize=40)
ax.set_xlim(53.5+1.5, 53.5-1.5)
ax.set_ylim(0.3-1.5, 0.3+1.5)
ax.grid(alpha=0.5)
ax.tick_params(axis="x", labelsize=40)
ax.tick_params(axis="y", labelsize=40)
cbar = fig.colorbar(count, ax=ax)
cbar.set_label("counts in bins ( counts )",fontsize=50) 
cbar.ax.tick_params(labelsize=40) 

ax.set_title('Count in bins, bins = 50',fontsize = 60)

#average

H, xedges, yedges = np.histogram2d(x, y, bins = bin, weights = z)
H_counts, xedges, yedges = np.histogram2d(x, y, bins = bin) 
H = H/H_counts
average = ax0.imshow(H.T, origin='lower',  cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

ax0.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'], edgecolor = 'white' , s=700,facecolor='none')
for i, txt in enumerate(new_t['Source_Name']):           
    ax0.annotate(txt, (new_t['GLON'][i]+0.2, new_t['GLAT'][i]+0.05),fontsize=30,color='white')
    
ax0.set_xlabel("Galactic Longitude (deg)",fontsize=40)
ax0.set_ylabel("Galactic Latitude (deg)",fontsize=40)
ax0.set_xlim(53.5+1.5, 53.5-1.5)
ax0.set_ylim(0.3-1.5, 0.3+1.5)
ax0.grid(alpha=0.5)
ax0.tick_params(axis="x", labelsize=40)
ax0.tick_params(axis="y", labelsize=40)
cbar = fig.colorbar(average, ax=ax0)
cbar.set_label(" Average energy ( MeV )",fontsize=50) 
cbar.ax.tick_params(labelsize=40)  
ax0.set_title('Average energy in bins, bins = 50',fontsize = 60, horizontalalignment = 'center')

