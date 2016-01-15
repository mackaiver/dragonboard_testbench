# 1 pedestal_file öffnen
# plotten?
# mittelwerte berechnen

import numpy as np
import matplotlib.pyplot as plt
from dragonboard import EventGenerator

def offset_calc(filename):
	with open(filename, "rb") as f:
		generator = EventGenerator(f, max_events=20)
		#event = next(generator)
		#plt.plot(event.data["low"][0])
		#plt.show()
		
		# declare events which is a list of all generator events
		events=list(generator)
		
		# read out list length
		numevents=len(events)
	
		# initialize an array full of nans. length = number of caps, width = number of events
		pedestalvalues = np.full((numevents, 4096),np.nan)
		
		#
		pixelindex = 0
		
		for i in range(numevents):
			stopcell = events[i].header.stop_cells[pixelindex]
			roi = events[i].roi
			pedestalvalues[i][stopcell:stopcell+roi] = events[i].data["low"][pixelindex]
			
			plt.plot(pedestalvalues[i],".:")
			
		plt.show()
		
if __name__ == '__main__':
	offset_calc('Ped444706_1.dat')
