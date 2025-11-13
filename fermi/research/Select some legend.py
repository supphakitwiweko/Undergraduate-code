# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:19:13 2021

@author: Supphakit Wiweko
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
#make plot style
#plt.style.use('dark_background')
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

#all data

#data
#z
z1 = t_mapp_8year0['ENERGY']
z2 = t_mapp_8year1['ENERGY']
z3 = t_mapp_8year2['ENERGY']
z4 = t_mapp_8year3['ENERGY']
z5 = t_mapp_8year4['ENERGY']
z6 = t_mapp_8year5['ENERGY']
z7 = t_mapp_8year6['ENERGY']
z8 = t_mapp_8year7['ENERGY']
z9 = t_mapp_8year8['ENERGY']

z = np.concatenate((z1,z2,z3,z4,z5,z6,z7,z8,z9))



#test
mapp_day = fits.open('C:\\Users\\supph\\Documents\\Research\\New_Photon database\\L210530023002F357373F98_PH00.fits')
t_mapp_day = Table(mapp_day[1].data)
x = t_mapp_day['L']
y = t_mapp_day['B']
z = t_mapp_day['ENERGY']

fig, ax1= plt.subplots(figsize=(20,10))
ax1.hist(z, bins = len(z))
ax1.grid(True, which="both", ls="--",color='black')
ax1.set_xscale('log') 
ax1.set_yscale('log')
ax1.set_xlabel("Energy (MeV)",fontsize=40)
ax1.set_ylabel("Number of even (counts))",fontsize=40)
ax1.tick_params(axis="x", labelsize=40)
ax1.tick_params(axis="y", labelsize=40)
ax1.set_title('Full Legend',fontsize = 60)

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
z_0 = np.concatenate((z1,z2,z3,z4,z5,z6,z7,z8,z9))

#8 years
#histogram for check bin
bin = 50
H, xedges, yedges = np.histogram2d(x, y, bins = bin, weights = z)
H1, xedges, yedges = np.histogram2d(x, y, bins = bin)



#histogram
#Full legend
#bin = 500
bin = np.logspace(np.log10(100),np.log10(1000000), 100)
fig, ax1= plt.subplots(figsize=(40,20))
#ax1.hist(z, bins = len(z) , label ='counts = %a ' % int(len(z)))
ax1.hist(z_0, bins =bin , label ='counts = %a ' % int(len(z_0)))
ax1.set_xscale('log') 
ax1.set_yscale('log')
ax1.set_xlim(0,350000)
ax1.set_xlabel("Energy (MeV)",fontsize=40)
ax1.set_ylabel("Number of events (counts))",fontsize=40)
ax1.grid(True, which="both", ls="--",color='black',lw =0.5)
ax1.tick_params(axis="x", labelsize=40)
ax1.tick_params(axis="y", labelsize=40)
ax1.legend(loc='upper right',fontsize=40)
ax1.set_title('Full Legend',fontsize = 60, horizontalalignment = 'center')

#Energy, xedges, yedges = np.histogram2d(x, y, bins = bin, weights = z)
#count, xedges, yedges = np.histogram2d(x, y, bins = bin) 
#count = ax1.imshow(count.T, origin='lower',  cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
#ax1.scatter((new_t['GLON'] +180)%360 - 180, new_t['GLAT'],edgecolor = 'white' , s=700,facecolor='none')
#for i, txt in enumerate(new_t['Source_Name']):           
#    ax1.annotate(txt, (new_t['GLON'][i]+0.35, new_t['GLAT'][i]+0.05),fontsize=30,color='white')
#plt.imshow(Energy.T ,cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
#plt.imshow(count.T ,cmap='inferno',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
#newhis1 = ax1.hist(Energy,weights=count)


#check max enery which we don't see.
#selected the enerys are [100000,300000]
#check enerny more than 100,000 MeV
z_ = np.logical_and(z >=100000,z<=300000)
z_test =  z[z_]
fig, ax1= plt.subplots(figsize=(40,20))
ax1.hist(z_test, bins = len(z_test), label ='counts = %a ' % int(len(z_test)))
#ax1.set_xscale('log') 
#ax1.set_yscale('log')
ax1.set_xlabel("Energy (MeV)",fontsize=40)
ax1.set_ylabel("Number of events (counts))",fontsize=40)
ax1.grid(True, which="both", ls="--",color='black',lw =0.5)
ax1.tick_params(axis="x", labelsize=40)
ax1.tick_params(axis="y", labelsize=40)
ax1.legend(loc='upper right',fontsize=40)
ax1.set_title('Full Legend, the energy range from 100,000 - 300,000 MeV',fontsize = 60, horizontalalignment = 'center')

#selected the enerys are [100000,300000]
#check enerny more than 100,000 MeV
z_ = np.logical_and(z >=100000,z<=300000)
z_test =  z[z_]
fig, ax1= plt.subplots(figsize=(30,20))
ax1.hist(z_test, bins = bin , label ='counts = %a ' % int(len(z_test)))
#ax1.set_xscale('log') 
#ax1.set_yscale('log')
ax1.set_xlabel("Energy (MeV)",fontsize=40)
ax1.set_ylabel("Number of events (counts))",fontsize=40)
ax1.grid(True, which="both", ls="--",color='black',lw =0.5)
ax1.tick_params(axis="x", labelsize=40)
ax1.tick_params(axis="y", labelsize=40)
ax1.legend(loc='upper right',fontsize=40)
ax1.set_title('Full Legend, the energy range from 100,000 - 300,000 MeV',fontsize = 60, horizontalalignment = 'center')


#background
#change coordiante

new_x = np.logical_and(t_mapp_8year0['L'] >=52, t_mapp_8year0['L']<=55)
new_tx =  t_mapp_8year0[new_x]
new_y = np.logical_and(new_tx['B'] >= 1.3, new_tx['B'] <=1.8)
new_coordinate0 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year1['L'] >=52, t_mapp_8year1['L']<=55)
new_tx =  t_mapp_8year1[new_x]
new_y = np.logical_and(new_tx['B'] >= 1.3, new_tx['B'] <=1.8)
new_coordinate1 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year2['L'] >=52, t_mapp_8year2['L']<=55)
new_tx =  t_mapp_8year2[new_x]
new_y = np.logical_and(new_tx['B'] >= 1.3, new_tx['B'] <=1.8)
new_coordinate2 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year3['L'] >=52, t_mapp_8year3['L']<=55)
new_tx =  t_mapp_8year3[new_x]
new_y = np.logical_and(new_tx['B'] >= 1.3, new_tx['B'] <=1.8)
new_coordinate3 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year4['L'] >=52, t_mapp_8year4['L']<=55)
new_tx =  t_mapp_8year4[new_x]
new_y = np.logical_and(new_tx['B'] >= 1.3, new_tx['B'] <=1.8)
new_coordinate4 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year5['L'] >=52, t_mapp_8year5['L']<=55)
new_tx =  t_mapp_8year5[new_x]
new_y = np.logical_and(new_tx['B'] >= 1.3, new_tx['B'] <=1.8)
new_coordinate5 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year6['L'] >=52, t_mapp_8year6['L']<=55)
new_tx =  t_mapp_8year6[new_x]
new_y = np.logical_and(new_tx['B'] >= 1.3, new_tx['B'] <=1.8)
new_coordinate6 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year7['L'] >=52, t_mapp_8year7['L']<=55)
new_tx =  t_mapp_8year7[new_x]
new_y = np.logical_and(new_tx['B'] >= 1.3, new_tx['B'] <=1.8)
new_coordinate7 = new_tx[new_y]
#change coordiante

new_x = np.logical_and(t_mapp_8year8['L'] >=52, t_mapp_8year8['L']<=55)
new_tx =  t_mapp_8year8[new_x]
new_y = np.logical_and(new_tx['B'] >= 1.3, new_tx['B'] <=1.8)
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
z_1 = np.concatenate((z1,z2,z3,z4,z5,z6,z7,z8,z9))

fig, ax1= plt.subplots(figsize=(40,20))
ax1.hist(z_1, bins = bin,label ='counts = %a ' % int(len(z_1)))
#ax1.hist(z, bins = len(z),label ='counts = %a ' % int(len(z)))
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(0,350000)
ax1.set_xlabel("Energy (MeV)",fontsize=40)
ax1.set_ylabel("Number of events (counts))",fontsize=40)
ax1.grid(True, which="both", ls="--",color='black',lw =0.5)
ax1.tick_params(axis="x", labelsize=40)
ax1.tick_params(axis="y", labelsize=40)
ax1.legend(loc='upper right',fontsize=40)
ax1.set_title('The background',fontsize = 60, horizontalalignment = 'center')

###########################
z_ = np.logical_and(z >=100000,z<=300000)
z_test =  z[z_]
fig, ax1= plt.subplots(figsize=(40,20))
ax1.hist(z_test, bins = 10 , label ='counts = %a ' % int(len(z_test)))
#ax1.set_xscale('log') 
#ax1.set_yscale('log')
ax1.set_xlim(0,350000)
ax1.set_xlabel("Energy (MeV)",fontsize=40)
ax1.set_ylabel("Number of events (counts))",fontsize=40)
ax1.grid(True, which="both", ls="--",color='black',lw =0.5)
ax1.tick_params(axis="x", labelsize=40)
ax1.tick_params(axis="y", labelsize=40)
ax1.legend(loc='upper right',fontsize=40)
ax1.set_title('The background, the energy range from 100,000 - 300,000 MeV',fontsize = 60, horizontalalignment = 'center')

#Middle
#change coordiante

new_x = np.logical_and(t_mapp_8year0['L'] >=52, t_mapp_8year0['L']<=55)
new_tx =  t_mapp_8year0[new_x]
new_y = np.logical_and(new_tx['B'] >= -0.3, new_tx['B'] <=0.5)
new_coordinate0 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year1['L'] >=52, t_mapp_8year1['L']<=55)
new_tx =  t_mapp_8year1[new_x]
new_y = np.logical_and(new_tx['B'] >= -0.3, new_tx['B'] <=0.5)
new_coordinate1 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year2['L'] >=52, t_mapp_8year2['L']<=55)
new_tx =  t_mapp_8year2[new_x]
new_y = np.logical_and(new_tx['B'] >= -0.3, new_tx['B'] <=0.5)
new_coordinate2 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year3['L'] >=52, t_mapp_8year3['L']<=55)
new_tx =  t_mapp_8year3[new_x]
new_y = np.logical_and(new_tx['B'] >= -0.3, new_tx['B'] <=0.5)
new_coordinate3 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year4['L'] >=52, t_mapp_8year4['L']<=55)
new_tx =  t_mapp_8year4[new_x]
new_y = np.logical_and(new_tx['B'] >= -0.3, new_tx['B'] <=0.5)
new_coordinate4 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year5['L'] >=52, t_mapp_8year5['L']<=55)
new_tx =  t_mapp_8year5[new_x]
new_y = np.logical_and(new_tx['B'] >= -0.3, new_tx['B'] <=0.5)
new_coordinate5 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year6['L'] >=52, t_mapp_8year6['L']<=55)
new_tx =  t_mapp_8year6[new_x]
new_y = np.logical_and(new_tx['B'] >= -0.3, new_tx['B'] <=0.5)
new_coordinate6 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year7['L'] >=52, t_mapp_8year7['L']<=55)
new_tx =  t_mapp_8year7[new_x]
new_y = np.logical_and(new_tx['B'] >= -0.3, new_tx['B'] <=0.5)
new_coordinate7 = new_tx[new_y]
#change coordiante

new_x = np.logical_and(t_mapp_8year8['L'] >=52, t_mapp_8year8['L']<=55)
new_tx =  t_mapp_8year8[new_x]
new_y = np.logical_and(new_tx['B'] >= -0.3, new_tx['B'] <=0.5)
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
z_2 = np.concatenate((z1,z2,z3,z4,z5,z6,z7,z8,z9))

fig, ax1= plt.subplots(figsize=(40,20))
ax1.hist(z_2, bins = bin,label ='counts = %a ' % int(len(z_2)))
#ax1.hist(z, bins = len(z),label ='counts = %a ' % int(len(z)))
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(0,350000)
ax1.set_xlabel("Energy (MeV)",fontsize=40)
ax1.set_ylabel("Number of events (counts))",fontsize=40)
ax1.grid(True, which="both", ls="--",color='black',lw =0.5)
ax1.tick_params(axis="x", labelsize=40)
ax1.tick_params(axis="y", labelsize=40)
ax1.legend(loc='upper right',fontsize=40)
ax1.set_title('The brighter region',fontsize = 60, horizontalalignment = 'center')
################

z_ = np.logical_and(z_2 >=100000,z_2<=300000)
z_test =  z[z_]
fig, ax1= plt.subplots(figsize=(30,20))
ax1.hist(z_test, bins = len(z_test) , label ='counts = %a ' % int(len(z_test)))
#ax1.set_xscale('log') 
#ax1.set_yscale('log')
ax1.set_xlabel("Energy (MeV)",fontsize=40)
ax1.set_ylabel("Number of events (counts))",fontsize=40)
ax1.grid(True, which="both", ls="--",color='black',lw =0.5)
ax1.tick_params(axis="x", labelsize=40)
ax1.tick_params(axis="y", labelsize=40)
ax1.legend(loc='upper right',fontsize=40)
ax1.set_title('The brighter region, the energy range from 100,000 - 300,000 MeV',fontsize = 60, horizontalalignment = 'center')

#############
# 4FGL J1932.3+1916, size= 4 bins

new_x = np.logical_and(t_mapp_8year0['L'] >=54.58, t_mapp_8year0['L']<=54.7)
new_tx =  t_mapp_8year0[new_x]
new_y = np.logical_and(new_tx['B'] >= 5.96046e-06, new_tx['B'] <=0.120005)
new_coordinate0 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year1['L'] >=54.58, t_mapp_8year1['L']<=54.7)
new_tx =  t_mapp_8year1[new_x]
new_y = np.logical_and(new_tx['B'] >= 5.96046e-06, new_tx['B'] <=0.120005)
new_coordinate1 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year2['L'] >=54.58, t_mapp_8year2['L']<=54.7)
new_tx =  t_mapp_8year2[new_x]
new_y = np.logical_and(new_tx['B'] >= 5.96046e-06, new_tx['B'] <=0.120005)
new_coordinate2 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year3['L'] >=54.58, t_mapp_8year3['L']<=54.7)
new_tx =  t_mapp_8year3[new_x]
new_y = np.logical_and(new_tx['B'] >= 5.96046e-06, new_tx['B'] <=0.120005)
new_coordinate3 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year4['L'] >=54.58, t_mapp_8year4['L']<=54.7)
new_tx =  t_mapp_8year4[new_x]
new_y = np.logical_and(new_tx['B'] >= 5.96046e-06, new_tx['B'] <=0.120005)
new_coordinate4 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year5['L'] >=54.58, t_mapp_8year5['L']<=54.7)
new_tx =  t_mapp_8year5[new_x]
new_y = np.logical_and(new_tx['B'] >= 5.96046e-06, new_tx['B'] <=0.120005)
new_coordinate5 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year6['L'] >=54.58, t_mapp_8year6['L']<=54.7)
new_tx =  t_mapp_8year6[new_x]
new_y = np.logical_and(new_tx['B'] >= 5.96046e-06, new_tx['B'] <=0.120005)
new_coordinate6 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year7['L'] >=54.58, t_mapp_8year7['L']<=54.7)
new_tx =  t_mapp_8year7[new_x]
new_y = np.logical_and(new_tx['B'] >= 5.96046e-06, new_tx['B'] <=0.120005)
new_coordinate7 = new_tx[new_y]
#change coordiante

new_x = np.logical_and(t_mapp_8year8['L'] >=54.58, t_mapp_8year8['L']<=54.7)
new_tx =  t_mapp_8year8[new_x]
new_y = np.logical_and(new_tx['B'] >= 5.96046e-06, new_tx['B'] <=0.120005)
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
z_3 = np.concatenate((z1,z2,z3,z4,z5,z6,z7,z8,z9))

fig, ax1= plt.subplots(figsize=(40,20))
ax1.hist(z_3, bins = bin,label ='counts = %a ' % int(len(z_3)))
#ax1.hist(z, bins = len(z),label ='counts = %a ' % int(len(z)))
ax1.set_xscale("log")
#ax1.set_yscale("log")
ax1.set_xlim(0,350000)
ax1.set_xlabel("Energy (MeV)",fontsize=40)
ax1.set_ylabel("Number of events (counts))",fontsize=40)
ax1.grid(True, which="both", ls="--",color='black',lw =0.5)
ax1.tick_params(axis="x", labelsize=40)
ax1.tick_params(axis="y", labelsize=40)
ax1.legend(loc='upper right',fontsize=40)
ax1.set_title('4FGL J1932.3+1916, size= 4 bins',fontsize = 60, horizontalalignment = 'center')

z_ = np.logical_and(z_3 >=10000,z_3<=300000)
z_test =  z[z_]
fig, ax1= plt.subplots(figsize=(40,20))
ax1.hist(z_test, bins =7, label ='counts = %a ' % int(len(z_test)))
#ax1.set_xscale('log') 
#ax1.set_yscale('log')
ax1.set_xlim(0,350000)
ax1.set_xlabel("Energy (MeV)",fontsize=40)
ax1.set_ylabel("Number of events (counts))",fontsize=40)
ax1.grid(True, which="both", ls="--",color='black',lw =0.5)
ax1.tick_params(axis="x", labelsize=40)
ax1.tick_params(axis="y", labelsize=40)
ax1.legend(loc='upper right',fontsize=40)
ax1.set_title('4FGL J1932.3+1916, the energy range from 10,000 - 300,000 MeV',fontsize = 60, horizontalalignment = 'center')

#######################
# 4FGL J1932.3+1916, size= 9 bins

new_x = np.logical_and(t_mapp_8year0['L'] >=54.58, t_mapp_8year0['L']<=54.76)
new_tx =  t_mapp_8year0[new_x]
new_y = np.logical_and(new_tx['B'] >= 5.96046e-06, new_tx['B'] <=0.180005)
new_coordinate0 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year1['L'] >=54.58, t_mapp_8year1['L']<=54.76)
new_tx =  t_mapp_8year1[new_x]
new_y = np.logical_and(new_tx['B'] >= 5.96046e-06, new_tx['B'] <=0.180005)
new_coordinate1 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year2['L'] >=54.58, t_mapp_8year2['L']<=54.76)
new_tx =  t_mapp_8year2[new_x]
new_y = np.logical_and(new_tx['B'] >= 5.96046e-06, new_tx['B'] <=0.180005)
new_coordinate2 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year3['L'] >=54.58, t_mapp_8year3['L']<=54.76)
new_tx =  t_mapp_8year3[new_x]
new_y = np.logical_and(new_tx['B'] >= 5.96046e-06, new_tx['B'] <=0.180005)
new_coordinate3 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year4['L'] >=54.58, t_mapp_8year4['L']<=54.76)
new_tx =  t_mapp_8year4[new_x]
new_y = np.logical_and(new_tx['B'] >= 5.96046e-06, new_tx['B'] <=0.180005)
new_coordinate4 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year5['L'] >=54.58, t_mapp_8year5['L']<=54.76)
new_tx =  t_mapp_8year5[new_x]
new_y = np.logical_and(new_tx['B'] >= 5.96046e-06, new_tx['B'] <=0.180005)
new_coordinate5 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year6['L'] >=54.58, t_mapp_8year6['L']<=54.76)
new_tx =  t_mapp_8year6[new_x]
new_y = np.logical_and(new_tx['B'] >= 5.96046e-06, new_tx['B'] <=0.180005)
new_coordinate6 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year7['L'] >=54.58, t_mapp_8year7['L']<=54.76)
new_tx =  t_mapp_8year7[new_x]
new_y = np.logical_and(new_tx['B'] >= 5.96046e-06, new_tx['B'] <=0.180005)
new_coordinate7 = new_tx[new_y]
#change coordiante

new_x = np.logical_and(t_mapp_8year8['L'] >=54.58, t_mapp_8year8['L']<=54.76)
new_tx =  t_mapp_8year8[new_x]
new_y = np.logical_and(new_tx['B'] >= 5.96046e-06, new_tx['B'] <=0.180005)
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
z_6 = np.concatenate((z1,z2,z3,z4,z5,z6,z7,z8,z9))

fig, ax1= plt.subplots(figsize=(40,20))
ax1.hist(z_6, bins = bin,label ='counts = %a ' % int(len(z_6)))
#ax1.hist(z, bins = len(z),label ='counts = %a ' % int(len(z)))
ax1.set_xscale("log")
#ax1.set_yscale("log")
ax1.set_xlim(0,350000)
ax1.set_xlabel("Energy (MeV)",fontsize=40)
ax1.set_ylabel("Number of events (counts))",fontsize=40)
ax1.grid(True, which="both", ls="--",color='black',lw =0.5)
ax1.tick_params(axis="x", labelsize=40)
ax1.tick_params(axis="y", labelsize=40)
ax1.legend(loc='upper right',fontsize=40)
ax1.set_title('4FGL J1932.3+1916, size= 9 bins',fontsize = 60, horizontalalignment = 'center')
############


# 4FGL J1928.4+1801c size= 4 bins

new_x = np.logical_and(t_mapp_8year0['L'] >=53.02, t_mapp_8year0['L']<=53.14)
new_tx =  t_mapp_8year0[new_x]
new_y = np.logical_and(new_tx['B'] >= 0.240004, new_tx['B'] <=0.360004)
new_coordinate0 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year1['L'] >=53.02, t_mapp_8year1['L']<=53.14)
new_tx =  t_mapp_8year1[new_x]
new_y = np.logical_and(new_tx['B'] >= 0.240004, new_tx['B'] <=0.360004)
new_coordinate1 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year2['L'] >=53.02, t_mapp_8year2['L']<=53.14)
new_tx =  t_mapp_8year2[new_x]
new_y = np.logical_and(new_tx['B'] >=0.240004, new_tx['B'] <=0.360004)
new_coordinate2 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year3['L'] >=53.02, t_mapp_8year3['L']<=53.14)
new_tx =  t_mapp_8year3[new_x]
new_y = np.logical_and(new_tx['B'] >= 0.240004, new_tx['B'] <=0.360004)
new_coordinate3 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year4['L'] >=53.02, t_mapp_8year4['L']<=53.14)
new_tx =  t_mapp_8year4[new_x]
new_y = np.logical_and(new_tx['B'] >=0.240004, new_tx['B'] <=0.360004)
new_coordinate4 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year5['L'] >=53.02, t_mapp_8year5['L']<=53.14)
new_tx =  t_mapp_8year5[new_x]
new_y = np.logical_and(new_tx['B'] >= 0.240004, new_tx['B'] <=0.360004)
new_coordinate5 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year6['L'] >=53.02, t_mapp_8year6['L']<=53.14)
new_tx =  t_mapp_8year6[new_x]
new_y = np.logical_and(new_tx['B'] >= 0.240004, new_tx['B'] <=0.360004)
new_coordinate6 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year7['L'] >=53.02, t_mapp_8year7['L']<=53.14)
new_tx =  t_mapp_8year7[new_x]
new_y = np.logical_and(new_tx['B'] >= 0.240004, new_tx['B'] <=0.360004)
new_coordinate7 = new_tx[new_y]
#change coordiante

new_x = np.logical_and(t_mapp_8year8['L'] >=53.02, t_mapp_8year8['L']<=53.14)
new_tx =  t_mapp_8year8[new_x]
new_y = np.logical_and(new_tx['B'] >= 0.240004, new_tx['B'] <=0.360004)
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
z_4 = np.concatenate((z1,z2,z3,z4,z5,z6,z7,z8,z9))

fig, ax1= plt.subplots(figsize=(40,20))
ax1.hist(z_4, bins = bin,label ='counts = %a ' % int(len(z_4)))
#ax1.hist(z, bins = len(z),label ='counts = %a ' % int(len(z)))
ax1.set_xscale("log")
#ax1.set_yscale("log")
ax1.set_xlim(0,350000)
ax1.set_xlabel("Energy (MeV)",fontsize=40)
ax1.set_ylabel("Number of events (counts))",fontsize=40)
ax1.grid(True, which="both", ls="--",color='black',lw =0.5)
ax1.tick_params(axis="x", labelsize=40)
ax1.tick_params(axis="y", labelsize=40)
ax1.legend(loc='upper right',fontsize=40)
ax1.set_title('4FGL J1928.4+1801c, size= 4 bins',fontsize = 60, horizontalalignment = 'center')

# 4FGL J1928.4+1801c size = 9 bins

new_x = np.logical_and(t_mapp_8year0['L'] >=53.02, t_mapp_8year0['L']<=53.2)
new_tx =  t_mapp_8year0[new_x]
new_y = np.logical_and(new_tx['B'] >= 0.180002, new_tx['B'] <=0.360004)
new_coordinate0 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year1['L'] >=53.02, t_mapp_8year1['L']<=53.2)
new_tx =  t_mapp_8year1[new_x]
new_y = np.logical_and(new_tx['B'] >= 0.180002, new_tx['B'] <=0.360004)
new_coordinate1 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year2['L'] >=53.02, t_mapp_8year2['L']<=53.2)
new_tx =  t_mapp_8year2[new_x]
new_y = np.logical_and(new_tx['B'] >= 0.180002, new_tx['B'] <=0.360004)
new_coordinate2 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year3['L'] >=53.02, t_mapp_8year3['L']<=53.2)
new_tx =  t_mapp_8year3[new_x]
new_y = np.logical_and(new_tx['B'] >= 0.180002, new_tx['B'] <=0.360004)
new_coordinate3 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year4['L'] >=53.02, t_mapp_8year4['L']<=53.2)
new_tx =  t_mapp_8year4[new_x]
new_y = np.logical_and(new_tx['B'] >= 0.180002, new_tx['B'] <=0.360004)
new_coordinate4 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year5['L'] >=53.02, t_mapp_8year5['L']<=53.2)
new_tx =  t_mapp_8year5[new_x]
new_y = np.logical_and(new_tx['B'] >= 0.180002, new_tx['B'] <=0.360004)
new_coordinate5 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year6['L'] >=53.02, t_mapp_8year6['L']<=53.2)
new_tx =  t_mapp_8year6[new_x]
new_y = np.logical_and(new_tx['B'] >= 0.180002, new_tx['B'] <=0.360004)
new_coordinate6 = new_tx[new_y]

#change coordiante

new_x = np.logical_and(t_mapp_8year7['L'] >=53.02, t_mapp_8year7['L']<=53.2)
new_tx =  t_mapp_8year7[new_x]
new_y = np.logical_and(new_tx['B'] >= 0.180002, new_tx['B'] <=0.360004)
new_coordinate7 = new_tx[new_y]
#change coordiante

new_x = np.logical_and(t_mapp_8year8['L'] >=53.02, t_mapp_8year8['L']<=53.2)
new_tx =  t_mapp_8year8[new_x]
new_y = np.logical_and(new_tx['B'] >= 0.180002, new_tx['B'] <=0.360004)
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
z_5 = np.concatenate((z1,z2,z3,z4,z5,z6,z7,z8,z9))

fig, ax1= plt.subplots(figsize=(40,20))
ax1.hist(z_5, bins = bin,label ='counts = %a ' % int(len(z_5)))
#ax1.hist(z, bins = len(z),label ='counts = %a ' % int(len(z)))
ax1.set_xscale("log")
#ax1.set_yscale("log")
ax1.set_xlim(0,350000)
ax1.set_xlabel("Energy (MeV)",fontsize=40)
ax1.set_ylabel("Number of events (counts))",fontsize=40)
ax1.grid(True, which="both", ls="--",color='black',lw =0.5)
ax1.tick_params(axis="x", labelsize=40)
ax1.tick_params(axis="y", labelsize=40)
ax1.legend(loc='upper right',fontsize=40)
ax1.set_title('4FGL J1928.4+1801c, size= 9 bins',fontsize = 60, horizontalalignment = 'center')



#to prepare
#data
#the souces in 4 bin
fig, ax1= plt.subplots(figsize=(40,20))
ax1.hist(z_0, bins = bin,label ='Full Region counts = %a ' % int(len(z_0)),color='yellow')
ax1.hist(z_2, bins = bin,label ='The brighter region counts = %a ' % int(len(z_2)),color='b')
ax1.hist(z_1, bins = bin,label ='Background counts = %a ' % int(len(z_1)),color='r')
ax1.hist(z_3, bins = bin,label ='4FGL J1932.3+1916, size= 4 bins counts = %a ' % int(len(z_3)),color='g')
ax1.hist(z_4, bins = bin,label ='4FGL J1928.4+1801c, size= 4 bins counts = %a ' % int(len(z_4)),color='pink')
#ax1.hist(z_0, bins = len(z_0),label ='Full Region counts = %a ' % int(len(z_0)),color='yellow')
#ax1.hist(z_2, bins = len(z_2),label ='The brighter region counts = %a ' % int(len(z_2)),color='b')
#ax1.hist(z_1, bins = len(z_1),label ='Background counts = %a ' % int(len(z_1)),color='r')
#ax1.hist(z_3, bins = len(z_3),label ='4FGL J1932.3+1916 counts = %a ' % int(len(z_3)),color='g')
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(100,350000)
ax1.set_xlabel("Energy (MeV)",fontsize=40)
ax1.set_ylabel("Number of events (counts))",fontsize=40)
ax1.grid(True, which="both", ls="--",color='black',lw =0.5)
ax1.tick_params(axis="x", labelsize=40)
ax1.tick_params(axis="y", labelsize=40)
ax1.legend(loc='upper right',fontsize=40)
ax1.set_title('Total regions',fontsize = 60, horizontalalignment = 'center')

#the sources in 9 bin
fig, ax1= plt.subplots(figsize=(40,20))
ax1.hist(z_0, bins = bin,label ='Full Region counts = %a ' % int(len(z_0)),color='yellow')
ax1.hist(z_2, bins = bin,label ='The brighter region counts = %a ' % int(len(z_2)),color='b')
ax1.hist(z_1, bins = bin,label ='Background counts = %a ' % int(len(z_1)),color='r')
ax1.hist(z_6, bins = bin,label ='4FGL J1932.3+1916,size= 9 bins counts = %a ' % int(len(z_6)),color='g')
ax1.hist(z_5, bins = bin,label ='4FGL J1928.4+1801c,size= 9 bins counts = %a ' % int(len(z_5)),color='pink')
#ax1.hist(z_0, bins = len(z_0),label ='Full Region counts = %a ' % int(len(z_0)),color='yellow')
#ax1.hist(z_2, bins = len(z_2),label ='The brighter region counts = %a ' % int(len(z_2)),color='b')
#ax1.hist(z_1, bins = len(z_1),label ='Background counts = %a ' % int(len(z_1)),color='r')
#ax1.hist(z_3, bins = len(z_3),label ='4FGL J1932.3+1916 counts = %a ' % int(len(z_3)),color='g')
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(100,350000)
ax1.set_xlabel("Energy (MeV)",fontsize=40)
ax1.set_ylabel("Number of events (counts))",fontsize=40)
ax1.grid(True, which="both", ls="--",color='black',lw =0.5)
ax1.tick_params(axis="x", labelsize=40)
ax1.tick_params(axis="y", labelsize=40)
ax1.legend(loc='upper right',fontsize=40)
ax1.set_title('Total regions',fontsize = 60, horizontalalignment = 'center')
#selected energy

z_ = np.logical_and(z_0 >=100000,z_0<=300000)
z_0test =  z_0[z_]
z_ = np.logical_and(z_1 >=100000,z_1<=300000)
z_1test =  z_1[z_]
z_ = np.logical_and(z_2 >=100000,z_2<=300000)
z_2test =  z_2[z_]
z_ = np.logical_and(z_3 >=100000,z_3<=300000)
z_3test =  z_3[z_]
fig, ax1= plt.subplots(figsize=(30,20))
ax1.hist(z_0test, bins = len(z_0test),label ='Full Region counts = %a ' % int(len(z_0test)),color='yellow')
ax1.hist(z_2test, bins = len(z_2test),label ='The brighter region counts = %a ' % int(len(z_2test)),color='b')
ax1.hist(z_1test, bins = 100,label ='Background counts = %a ' % int(len(z_1test)),color='r')
ax1.hist(z_3test, bins =100,label ='4FGL J1932.3+1916 counts = %a ' % int(len(z_3test)),color='g')
#ax1.hist(z_0, bins = len(z_0),label ='Full Region counts = %a ' % int(len(z_0)),color='yellow')
#ax1.hist(z_2, bins = len(z_2),label ='The brighter region counts = %a ' % int(len(z_2)),color='b')
#ax1.hist(z_1, bins = len(z_1),label ='Background counts = %a ' % int(len(z_1)),color='r')
#ax1.hist(z_3, bins = len(z_3),label ='4FGL J1932.3+1916 counts = %a ' % int(len(z_3)),color='g')
#ax1.set_xscale("log")
#ax1.set_yscale("log")
ax1.set_xlabel("Energy (MeV)",fontsize=40)
ax1.set_ylabel("Number of events (counts))",fontsize=40)
ax1.grid(True, which="both", ls="--",color='black',lw =0.5)
ax1.tick_params(axis="x", labelsize=40)
ax1.tick_params(axis="y", labelsize=40)
ax1.legend(loc='upper right',fontsize=40)
ax1.set_title('Total regions',fontsize = 60, horizontalalignment = 'center')

