# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:32:33 2021

@author: exp-lsk-fourier
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RangeSlider
from scipy.signal import savgol_filter 
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle

class InteractiveCursor():
    '''
    Abstract class to create an interactive cursor for matplotlib figures.
    The class instance needs to register a matplotlib axis 
    and method(s) to be called when clicking events occur. This includes a 
    method when the mouse is released (method_offclick) and an optional method to be called when the mouse is
    moved while being clicked (method_moved).
    
    Parameters for initialization
    ----------
    axis : matplotlib axis instance, the cursor will be responsive in this axis only.
    method_offclick : function to be called when the mouse button is released. DEFAULT is NONE
    *method_moved : function to be called when the mouse is moved while being clicked. DEFAULT is equal to method_offclick
    '''
    def __init__(self, axis, method_offclick=None, *method_moved): # Methods are called everytime the corresponding event occurs. When not provided, 'method_moved' is set equal to 'method_offclick'
        self.axis=axis
        self.current_pos=[0,0]
        
        self.fig=plt.gcf() 	### Get current figure for event handling
        
        ### Tools for interactive mouse events
        self.connection_onclick=self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.method_offclick=method_offclick
        
        if method_moved == True:
            self.method_moved=method_moved[0]
        else: self.method_moved=method_offclick

    def onclick(self,event):
        '''
        When the mouse is clicked, a gui connection for each mouse moving and releasing events is established
        '''
        if  event.inaxes !=self.axis:
            return
        self.connection_mouse_clicked=self.fig.canvas.mpl_connect('motion_notify_event', self.move_mouse_while_clicked)
        self.connection_offclick=self.fig.canvas.mpl_connect('button_release_event', self.offclick)
        return
    
    def offclick(self,event):
        '''
        When the mouse is released within the interactive axis, method_offclick() is called and the current position 
        of the mouseis returned; all gui connections are disconnected.
        '''
        if event.inaxes != self.axis:
            return
        self.current_pos=[event.xdata, event.ydata]
        self.fig.canvas.mpl_disconnect(self.connection_mouse_clicked)
        self.fig.canvas.mpl_disconnect(self.connection_offclick)
        self.method_offclick()
        return self.current_pos
    
    def move_mouse_while_clicked(self,event):
        '''
        When the mouse is moved while being clicked,method_moved() is called and the current position is returned.
        '''
        if  event.inaxes != self.axis:
            return
        self.current_pos=[event.xdata, event.ydata]
        self.method_moved()
        return self.current_pos
	
    def get_position(self,swap_coordinates=False):
        '''
        Returns the current position of the cursor.
        '''
        if swap_coordinates == False:
            return self.current_pos
        else:
            return [self.current_pos[1],self.current_pos[0]]	

class plot_voltage_sweep:
    _counter=0 #counts the number of initialized instances  
               #--> changes the name of the respective matplotlib figure and of its GUI handlers to enable multiple instances simultaneously
    '''
    Plots 2-D heatmaps 
    # ========
    # wavelengths and v_vals are 1D arrays containing the parameters of the sweep, spectra is a 2D array 
    # containing the data (first dimension - V, second - spectra);
    # the spectral data should be in wavelength representation
    # ========
    '''
    def __init__(self, spectra, wavelengths, v_vals, secondary_v_vals=None, energy_space = True, background=None, cmap='twilight_shifted', aspect = 5):
        plot_voltage_sweep._counter += 1
        
        self.spectra = spectra
        self.raw_spectra = spectra
        self.wavelengths = wavelengths
        self.v_vals = v_vals
        self.secondary_v_vals = secondary_v_vals
        self.energy_space = energy_space
        self.background = background
        self.cmap = cmap
        
        # ========
        # Data preparation
        # ========
        
        if self.background is not None:
            for i in range(len(self.spectra)):
                self.raw_spectra[i] = (self.background-self.raw_spectra[i])/self.background
                
    
        if self.energy_space:
            self.spectra=self.raw_spectra[:,::-1] #this is necessary, since the plt.imshow plots against the pixel indices

        # ========
        # Plotting Environment 
        # ========
        
        self.fig = plt.figure('Gate Sweep ' + str(plot_voltage_sweep._counter))

        self.ax = self.fig.add_subplot(111)
        self.ax.set_title('Press \'d\' to take a derivative, \'r\' to reset!', fontsize = 7)
        
        if self.secondary_v_vals is not None:
            self.secax = self.ax.secondary_yaxis('right')
            self.secax.get_yaxis().set_major_formatter(self.voltage_formatter(self.secondary_v_vals))
            self.secax.set_ylabel('Topgate Voltage [V]')
        
        if self.energy_space:
            self.ax.xaxis.set_major_formatter(self.E_formatter(self.wavelengths))
            self.ax.set_xlabel('Energy [eV]')
        else:
            self.ax.xaxis.set_major_formatter(self.wl_formatter(self.wavelengths))
            self.ax.set_xlabel('Wavelength [nm]')
        
        self.ax.set_ylabel('Backgate Voltage [V]')
        self.ax.yaxis.set_major_formatter(self.voltage_formatter(self.v_vals))
        #self.ax.set_yticks(self.find_turning_points(self.v_vals))
        
        self.im = self.ax.imshow(self.spectra, cmap=self.cmap, aspect=aspect)
        
        globals()['self.key_conn'+str(plot_voltage_sweep._counter)] = self.fig.canvas.mpl_connect('key_press_event', self.on_key) #the globals namespace ensures that the connection is not garbage-collected
        
        self.fig.colorbar(self.im, shrink=0.5, orientation = 'horizontal')
        
        # Derivative/Reset buttons
        self.derivative_button_axes = self.fig.add_axes([0.30, 0.05, 0.12, 0.08])
        self.reset_button_axes = self.fig.add_axes([0.55, 0.05, 0.12, 0.08])
        
        globals()['self.derivative_button'+str(plot_voltage_sweep._counter)] = Button(self.derivative_button_axes, 'Derivative')
        globals()['self.reset_button'+str(plot_voltage_sweep._counter)] = Button(self.reset_button_axes, 'Reset')
        
        globals()['self.derivative_button'+str(plot_voltage_sweep._counter)].on_clicked(self.derivative)
        globals()['self.reset_button'+str(plot_voltage_sweep._counter)].on_clicked(self.reset)
        
        plt.show()
    
    # ========
    # Auxiliary functions for formatting the voltage sweep plots
    # ========

    

    def wl_formatter(self, wl_array):
        def index_to_wl_formatter(val, pos):
            #a=(wl_array[-1]-wl_array[0])/len(wl_array)   # this would be a simple linear fit, sufficient in most cases
            #b=wl_array[0]
            coefs=np.polyfit(np.arange(len(wl_array)), wl_array, 4)[::-1]   # this is a more general polynomial fit, 
            result=0                                                        # in case the wavelengths spacing is not homogeneous
            for i, c in enumerate(coefs):
                result+=c*val**i
            return round(result,1)      # f'{a*val+b:.1f}'
        return index_to_wl_formatter
    
    def E_formatter(self, wl_array):
        def index_to_E_formatter(val, pos):
            coefs=np.polyfit(np.arange(len(wl_array))[::-1], wl_array, 4)[::-1]
            wl_result=0
            for i, c in enumerate(coefs):
                wl_result+=c*val**i
            return round(1239.84/wl_result,4)
        return index_to_E_formatter
        
    def voltage_formatter(self, V_array):
        def index_to_V_formatter(val,pos):
            if 0 <= val <= len(V_array)-1:
                return round(V_array[int(val)],1)
            else:
                pass
        return index_to_V_formatter
    
    def find_turning_points(self, V_array):
        a=((V_array[:-1]-V_array[1:])>0)
        b=(a[:-1]!=a[1:])
        
        c=(V_array<=0)
        d=(c!=np.roll(c,-1))
        return np.arange(len(V_array))[np.logical_or( d, np.hstack(([True], b, [True])) )]
    

    def on_key(self, event):
        if event.key == 'r':
            self.reset()
        elif event.key == 'd':
            self.derivative()

    def adjust_colormap(self):
        loc_min=self.spectra.min()
        loc_max=self.spectra.max()
        self.im.set_clim(vmin=loc_min , vmax=loc_max)
    
    def reset(self, event=None):
        if self.energy_space:
            self.spectra = self.raw_spectra[:,::-1]
        else:
            self.spectra = self.raw_spectra
        self.im.set_data(self.spectra)
        self.adjust_colormap()
        plt.draw()
    
    def derivative(self, event=None):
        shift=2
        derv = savgol_filter((self.spectra[:,2*shift:]-self.spectra[:,:-2*shift]), 31, 1)
        derv=np.hstack( [np.array(derv[:,shift]).reshape(-1,1)]*shift + [derv] +  [np.array(derv[:,-shift]).reshape(-1,1)]*shift)
        self.spectra=derv
        self.im.set_data(self.spectra)
        self.adjust_colormap()
        plt.draw()




class GateMap:
    
    '''
    # A class to plot interactive 2D maps of spectroscopic data, optimized for dual-gate sweeps (needs Matplotlib version 3.5 to enable the RangeSlider)
    #
    # The input is:
       1. A np-array of voltages [V_tg,V_bg] of shape (N_tg,N_bg,2), e.g. (30,60,2)
       2. A np-array of the acquired wavelengths, i.e. an array of length N (the spectroscopic data is plotted against the wavelength)
       3. A np-array of spectroscopic data in wavelength representation, e.g. of shape (30,60,N), where N is the number of acquired wavelengths
       (optional) 4. background: A 1D background array for differential reflectance; keyword-only
       (optional) 5. energy_space:  0 or 1; keyword-only
       (optional) 6. colormap: Matplotlib colormap name for the heatmap (the default is 'terrain'); keyword-only
       (optional) 7. map_origin: orientation of the map-plotting, either 'upper' or 'lower'; keyword-only
       (optional) 8. heatmap_mode: 'max', 'mean' or 'min' to define the heatmap color choice
       (optional) 9. vert_ax_title: name of the vertical axis of the map (first axis of the array); default is r'V_{TG}', keyword-only
       (optional) 10. horiz_ax_title: the same or the horizontal axis (second axis of the array); default is r'V_{BG}'
       (optional) 11. smoothing: int, default is 0. If >0, a savgol_filter is applied to the spectral data over an interval of this length; keyword only
    # You should adjust the spectral region and the contrast for the colormap in the interactive map.
    #
    ###
    '''

    def __init__(self,voltages,wavelengths,spectra, background=None, energy_space = 0, colormap='terrain', heatmap_mode='max', map_origin = 'upper', vert_ax_title=r'$V_{TG}$', horiz_ax_title=r'$V_{BG}$', smoothing =0):
		
        # Mode of intensity acquisition
        self.mode=heatmap_mode
        self.map_origin = map_origin
    
        ####### Data
        self.energy_space = energy_space
        self.voltages=voltages
        self.raw_spectra=spectra
        self.spectra=spectra
        self.wavelengths=wavelengths
        self.range=wavelengths #if energy_space == 1 this will be turned into energies
        self.background = background
        self.lines_present = [] # container for picking lines to plot single gate sweeps
        
        self.vert_ax_title = vert_ax_title
        self.horiz_ax_title = horiz_ax_title
        
        
        for i in range(len(self.raw_spectra[:,0,0])):
            for j in range(len(self.raw_spectra[0,:,0])):
                self.cosmic_removal(self.raw_spectra[i,j,:])
                if background is not None:
                    self.raw_spectra[i,j]=(self.background-self.raw_spectra[i,j])/self.background
        
        if smoothing:
            self.raw_spectra=savgol_filter(spectra, smoothing, 3)
            
        if self.energy_space:
            self.raw_spectra=self.raw_spectra[:,:,::-1]
            self.spectra = self.raw_spectra
            
            self.range=1239.84/wavelengths[::-1]

        
        ###########
        # Plotting environment
        ###########
                
        self.fig=plt.figure(figsize=(16,8))
        self.fig.subplots_adjust(top=0.94,left=0.05,right=0.95,bottom=0.2)
        
        ### Heatmap environment
        self.ax_map=self.fig.add_subplot(121,frame_on=False)
        
        self.map_cursor=InteractiveCursor(self.ax_map,  self.mouse_update )  # Interactive cursor control, calls self.update upon releasing/moving a clicked mouse button
        self.current_pixel=np.array([int(round(x)) for x in self.map_cursor.get_position(swap_coordinates=True)]) # Note: imshow sees axis0 as y-coordinate, axis1 as x-coordinate, therefore we swap the position
        
        self.key_conn = self.fig.canvas.mpl_connect('key_press_event', self.on_key) # Connection for interactive keyboard control implemented in the function self.on_key
        
       
        self.ax_map.xaxis.set_major_formatter(self.voltage_formatter(self.voltages[0,:,1]))
        self.ax_map.yaxis.set_major_formatter(self.voltage_formatter(self.voltages[:,0,0]))

        self.ax_map.set_title(' Use \'p\' and \'o\' to enable zooming tools and to adjust the colorbar, \n' \
                              +'\'d\' to change to the derivative, \'r\' to reset, \'b\' to draw a 1D-Gate sweep line')
        self.ax_map.set_xlabel(self.horiz_ax_title +'\n' + r'$V_{TG} = $' + str(round(self.voltages[self.current_pixel[0],self.current_pixel[1],1],2)) \
                                 + '\t' + r'$V_{BG} = $' + str(round(self.voltages[self.current_pixel[0],self.current_pixel[1],0],2)))
        self.ax_map.set_ylabel(self.vert_ax_title)
        

        
        ### Spectrum plot environment
        self.ax_spectrum=self.fig.add_subplot(122)
        
        if self.energy_space:
            self.ax_spectrum.set_xlabel('Energy [eV]')
        else:
            self.ax_spectrum.set_xlabel('Wavelength [nm]')
        
        self.ax_spectrum.set_ylabel(r'Intensity')
        self.ax_spectrum.set_ylim(self.spectra.min()-(self.spectra.max()-self.spectra.min())*0.1,self.spectra.max() + (self.spectra.max()-self.spectra.min())*0.1) 
        
        
        # Interactive Sliders for contrast and spectral range control
        
        self.ROI_range_slider_ax  = self.fig.add_axes([0.2, 0.05, 0.5, 0.04])
        if not self.energy_space:
            self.ROI_range_slider = RangeSlider(self.ROI_range_slider_ax, label=r'$\lambda [nm] $' , valmin=self.range[0], valmax=self.range[-1], valfmt='%0.1f', handle_style={'size':0})
        elif self.energy_space:
            self.ROI_range_slider = RangeSlider(self.ROI_range_slider_ax, label=r'E[eV]', valmin=self.range[0], valmax=self.range[-1], valfmt='%0.4f', handle_style={'size':0})
        self.ROI_range_slider.on_changed(self.ROI_range_slider_on_changed)
        
        self.ROI_rectangle = Rectangle(xy=(self.ROI_range_slider.val[0], -50), height = 70000, width = self.ROI_range_slider.val[1] - self.ROI_range_slider.val[0], alpha = 0.1, color = 'salmon' )
        self.ax_spectrum.add_patch(self.ROI_rectangle)
        
        # Plotting objects
        self.intensities=self.draw_intensities(self.ROI_range_slider.val[0],self.ROI_range_slider.val[1]) #draw the intensities for the heatmap
        self.map=self.ax_map.imshow(self.intensities,cmap=colormap,interpolation=None, origin = self.map_origin)
        self.current_pos_cursor=self.ax_map.scatter(self.current_pixel[1],self.current_pixel[0],color='red') # self.current_pixel is swapped since its first dimension is the y-axis, whereas for the plt.scatter the first entry refers to the x-coordinate
        self.spectral_plot=self.ax_spectrum.plot(self.range,  self.spectra[self.current_pixel[0],self.current_pixel[1],:]  )[0]
        
        self.cb=self.fig.colorbar(self.map,ax=self.ax_map,orientation='vertical')
    # =========
    # Class methods
    # =========
        
    def cosmic_removal(self, arr, threshold_factor=4, cosmic_width=3):
        '''
        Finds cosmics by looking at the 2nd numeric derivative of the input array. Where it is bigger than some threshold
        the function will replace a whole window around the cosmic by a plateau of values picked from the neighbourhood of
        the cosmic.
    
        Parameters
        ----------
        arr : 1D numpy array containing the spectral data
        threshold_factor : FLOAT, multiplies the standard deviation of the input array to determine the thershold. 
            The default is 4.2.
        cosmic_width : INTEGER, used to replace values of the cosmic by pick the entries which are cosmic_width indices away
    
        Returns
        -------
        None, changes the input array by reference
    
        '''
        length = len(arr)
        cosmic_inds =  np.arange(length)[( np.abs(arr + np.roll(arr,2)-2*np.roll(arr,1)) > arr.std()*threshold_factor )]
        for i in cosmic_inds:
            arr[i-cosmic_width:min(i+cosmic_width,length)]=arr[i-cosmic_width]
        
    
    def voltage_formatter(self, V_array):
        def index_to_V_formatter(val,pos):
            if 0 <= val <= len(V_array)-1:
                return round(V_array[int(val)],1)
            else:
                pass
        return index_to_V_formatter
    
    
    def draw_intensities(self,left_ROI_boundary,right_ROI_boundary, mode=None):  # function to draw the intensities for the heatmap 
        ### find the indices corresponding to the given range of the wavelengths considered
        if not mode:
            mode=self.mode
        ind_left=self.range.searchsorted(left_ROI_boundary)
        ind_right=self.range.searchsorted(right_ROI_boundary)
        
        if ind_right>ind_left:
            if mode == 'max':
                return self.spectra[:,:,ind_left:ind_right].max(axis=-1)
            elif mode == 'mean':
                return self.spectra[:,:,ind_left:ind_right].mean(axis=-1)
            elif mode == 'min':
                return self.spectra[:,:,ind_left:ind_right].min(axis=-1)
        
        else:
            return self.spectra[:,:,ind_left]


    def mouse_update(self): # function called when you navigate with the cursor
        self.current_pixel=np.array([int(round(x)) for x in self.map_cursor.get_position(swap_coordinates=True)])
        self.spectral_plot.set_ydata(self.spectra[self.current_pixel[0],self.current_pixel[1],:])
        self.current_pos_cursor.set_offsets([self.current_pixel[1],self.current_pixel[0]])
        
        self.ax_map.set_xlabel(self.horiz_ax_title + '\n' +r'$V_{TG} = $' + str(round(self.voltages[self.current_pixel[0],self.current_pixel[1],1],2)) + '\t' + r'$V_{BG} = $' + str(round(self.voltages[self.current_pixel[0],self.current_pixel[1],0],2))+
                               '\n' + 'Pixel: ' + str([self.current_pixel[0],self.current_pixel[1]]) )
        plt.draw()
	

    def ROI_range_slider_on_changed(self,val):
        self.intensities=self.draw_intensities(self.ROI_range_slider.val[0],self.ROI_range_slider.val[1])
        self.map.set_data(self.intensities)
        self.ROI_rectangle.set_x(self.ROI_range_slider.val[0])
        self.ROI_rectangle.set_width(self.ROI_range_slider.val[1] - self.ROI_range_slider.val[0])
        self.adjust_colormap()
        

    
    def adjust_colormap(self):
        loc_min=self.intensities.min()
        loc_max=self.intensities.max()

        self.map.set_clim(vmin=loc_min , vmax=loc_max)
    
    
    def on_key(self,event): # function calles when you navigate with the arrows
        if event.key == 'up':
            if self.map_origin == 'upper':
                self.current_pixel=[(self.current_pixel[0]-1)%self.spectra.shape[0],self.current_pixel[1]]
            elif self.map_origin == 'lower':
                self.current_pixel=[(self.current_pixel[0]+1)%self.spectra.shape[0],self.current_pixel[1]]

        elif event.key == 'down':
            if self.map_origin == 'upper':
                self.current_pixel=[(self.current_pixel[0]+1)%self.spectra.shape[0],self.current_pixel[1]]
            elif self.map_origin == 'lower':
                self.current_pixel=[(self.current_pixel[0]-1)%self.spectra.shape[0],self.current_pixel[1]]
                
        elif event.key == 'right':
            self.current_pixel=[self.current_pixel[0],(self.current_pixel[1]+1)%self.spectra.shape[1]]
            
        elif event.key == 'left':
            self.current_pixel =[self.current_pixel[0],(self.current_pixel[1]-1)%self.spectra.shape[1]]
            
        elif event.key == 'r':
            self.spectra=self.raw_spectra
            self.ax_spectrum.set_ylim(self.spectra.min()-(self.spectra.max()-self.spectra.min())*0.1,self.spectra.max() + (self.spectra.max()-self.spectra.min())*0.1) 
            self.intensities = self.draw_intensities(self.ROI_range_slider.val[0],self.ROI_range_slider.val[1], mode=self.mode)
            self.adjust_colormap()
            self.map.set_data(self.intensities)
            
        elif event.key == 'd':
            shift=2
            derv = savgol_filter((self.spectra[:,:,2*shift:]-self.spectra[:,:,:-2*shift]), 31, 3)
            derv=np.concatenate( shift*[np.expand_dims(derv[:,:,0], axis=2)] + [derv] +  [np.expand_dims(derv[:,:,-1], axis=2)]*shift, axis = 2)
            
            self.spectra=derv
            self.ax_spectrum.set_ylim(self.spectra.min()-(self.spectra.max()-self.spectra.min())*0.1,self.spectra.max() + (self.spectra.max()-self.spectra.min())*0.1) 
            self.intensities = self.draw_intensities(self.ROI_range_slider.val[0],self.ROI_range_slider.val[1], mode='max')
            self.map.set_data(self.intensities)
            self.adjust_colormap()
            
        elif event.key == 'b':
            self.conn_mouse_click = self.fig.canvas.mpl_connect('button_press_event', self.pick_line_on)
            self.key_release_conn = self.fig.canvas.mpl_connect('key_release_event', self.pick_line_off_key)

        self.spectral_plot.set_ydata(self.spectra[self.current_pixel[0],self.current_pixel[1],:])

        self.current_pos_cursor.set_offsets([self.current_pixel[1],self.current_pixel[0]])
        self.ax_map.set_xlabel(self.horiz_ax_title +'\n' +r'$V_{TG} = $' + str(round(self.voltages[self.current_pixel[0],self.current_pixel[1],1],2)) + '\t' + r'$V_{BG} = $' + str(round(self.voltages[self.current_pixel[0],self.current_pixel[1],0],2)) )
        plt.draw()
    
    
    # =========
    # Line-picking GUI events
    # =========
    
    def pick_line_off_key(self, event):
        if event.key == 'b':
            self.fig.canvas.mpl_disconnect(self.conn_mouse_click)
            self.fig.canvas.mpl_disconnect(self.key_release_conn)
            
    def pick_line_on(self, event):
        if  event.inaxes !=self.ax_map:
            return
        self.line_starting_pixel=np.array([int(event.xdata),(event.ydata)])
        
        self.line_active = False
        self.conn_mouse_motion=self.fig.canvas.mpl_connect('motion_notify_event', self.pick_line_motion)
        self.conn_mouse_release=self.fig.canvas.mpl_connect('button_release_event', self.pick_line_released)
    
    def pick_line_motion(self,event):
        if not self.line_active:
            self.lines_present.append( self.ax_map.plot([self.line_starting_pixel[0], event.xdata], [self.line_starting_pixel[1], event.ydata], color='red', alpha = 0.4, lw = 1.5, ls = '-.')[0] )
        self.lines_present[-1].set_xdata([self.line_starting_pixel[0], event.xdata])
        self.lines_present[-1].set_ydata([self.line_starting_pixel[1], event.ydata])
        self.line_active=True
        plt.draw()
    
    def pick_line_released(self, event):
        self.fig.canvas.mpl_disconnect(self.conn_mouse_click)
        self.fig.canvas.mpl_disconnect(self.conn_mouse_motion)
        self.fig.canvas.mpl_disconnect(self.conn_mouse_release)
        self.fig.canvas.mpl_disconnect(self.key_release_conn)
        
        for num, l in enumerate(self.lines_present[::-1]):
            l.set_alpha(max(0,0.4-num*0.1))
       
        p_0=np.array([int(round(self.lines_present[-1].get_ydata()[0])),  int(round(self.lines_present[-1].get_xdata()[0]))])
        p_f=np.array([int(round(self.lines_present[-1].get_ydata()[1])),  int(round(self.lines_present[-1].get_xdata()[1]))])
        N=int(  np.sqrt( ((self.voltages[p_0[0], p_0[1], :] -self.voltages[p_f[0], p_f[1], :])**2).sum() ) / (self.voltages[1,1,0]-self.voltages[0,1,0])    )
        
        pixels=[]
        loc_spectra=[]
        loc_voltage_TG=[]
        loc_voltage_BG = []
        
        for i in range(N):
            pixels.append((p_0+i*(p_f-p_0)/N).round().astype(int))
            loc_spectra.append(self.raw_spectra[pixels[-1][0], pixels[-1][1], :])
            loc_voltage_TG.append(self.voltages[pixels[-1][0], pixels[-1][1], 0  ])
            loc_voltage_BG.append(self.voltages[pixels[-1][0], pixels[-1][1], 1  ])
        
        pixels=np.array(pixels)
        loc_spectra=np.array(loc_spectra)
        loc_voltage_TG = np.array(loc_voltage_TG)
        loc_voltage_BG = np.array(loc_voltage_BG)

        
        if not self.energy_space:
            gate_sweep = plot_voltage_sweep(loc_spectra, self.wavelengths, loc_voltage_TG, secondary_v_vals = loc_voltage_BG, 
                                            cmap = 'twilight_shifted', energy_space=False)
        else:
            gate_sweep = plot_voltage_sweep(loc_spectra[:,::-1], self.wavelengths, loc_voltage_TG, secondary_v_vals = loc_voltage_BG,
                                            cmap = 'twilight_shifted', energy_space=True,aspect=0.3*len(self.raw_spectra[0,0,:])/N)


def bg_fit(wavelengths, signal, fit_inds, fit_deg=5):
    coeffs=(np.polyfit(wavelengths[fit_inds],signal[fit_inds],deg=fit_deg))[::-1]
    bg=0
    for i, c in enumerate(coeffs):
        bg += c * wavelengths**i
    return bg 

def cosmic_removal(arr, threshold_factor=4, cosmic_width=3):
    '''
    Finds cosmics by looking at the 2nd numeric derivative of the input array. Where it is bigger than some threshold
    the function will replace a whole window around the cosmic by a plateau of values picked from the neighbourhood of
    the cosmic.

    Parameters
    ----------
    arr : 1D numpy array containing the spectral data
    threshold_factor : FLOAT, multiplies the standard deviation of the input array to determine the thershold. 
        The default is 4.2.
    cosmic_width : INTEGER, used to replace values of the cosmic by pick the entries which are cosmic_width indices away

    Returns
    -------
    None, changes the input array by reference

    '''
    length = len(arr)
    cosmic_inds =  np.arange(length)[( np.abs(arr + np.roll(arr,2)-2*np.roll(arr,1)) > arr.std()*threshold_factor )]
    for i in cosmic_inds:
        arr[i-cosmic_width:min(i+cosmic_width,length)]=arr[i-cosmic_width]


if __name__=='__main__':
   
    i_0= 0  #define start and final index to crop the treated datasets
    i_f= None
        
    voltages=np.load('restricted_volts.npy') #3D array containing [Vx, Vy] for each pixel in [pixel_x, pixel_y, voltage]
    spectra=np.load('restricted_specs.npy')[:,:,i_0:i_f] #3D array containing signal [c1,c2, ..., cN] in [pixel_x, pixel_y, spectrum]
    wavelengths=np.load('restricted_wavelengths.npy')[i_0:i_f] #1D array containing the wavelengths

    
    charging_map = GateMap(voltages, wavelengths, spectra, heatmap_mode='max', colormap='terrain', energy_space = 1, map_origin = 'lower', vert_ax_title=r'$V_{BG}$', horiz_ax_title=r'$V_{TG}$', smoothing=0)
    plt.show()
