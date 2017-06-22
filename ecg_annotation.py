#import matplotlib with pdf as backend
import matplotlib 
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import wfdb 
import os
import numpy as np
import math
import sys
import scipy.stats as st
import glob, os
from os.path import basename


import tensorflow as tf
from keras.layers import Dense,Activation,Dropout
from keras.layers import LSTM,Bidirectional #could try TimeDistributed(Dense(...))
from keras.models import Sequential, load_model
from keras import optimizers,regularizers
from keras.layers.normalization import BatchNormalization
import keras.backend.tensorflow_backend as KTF
np.random.seed(0)

# functions
def get_ecg_data(datfile): 
	## convert .dat/q1c to numpy arrays
	recordname=os.path.basename(datfile).split(".dat")[0]
	recordpath=os.path.dirname(datfile)
	cwd=os.getcwd()
	os.chdir(recordpath) ## somehow it only works if you chdir. 

	annotator='q1c'
	annotation = wfdb.rdann(recordname, extension=annotator, sampfrom=0,sampto = None, pbdir=None)
	Lstannot=list(zip(annotation.sample,annotation.symbol,annotation.aux_note))

	FirstLstannot=min( i[0] for i in Lstannot)
	LastLstannot=max( i[0] for i in Lstannot)-1
	print("first-last annotation:", FirstLstannot,LastLstannot)
	
	record = wfdb.rdsamp(recordname, sampfrom=FirstLstannot,sampto = LastLstannot) #wfdb.showanncodes()
	annotation = wfdb.rdann(recordname, annotator, sampfrom=FirstLstannot,sampto = LastLstannot) ## get annotation between first and last. 
	annotation2 = wfdb.Annotation(recordname='sel32', extension='niek', sample=(annotation.sample-FirstLstannot), symbol = annotation.symbol, aux_note=annotation.aux_note)

	Vctrecord=np.transpose(record.p_signals)
	VctAnnotationHot=np.zeros( (6,len(Vctrecord[1])), dtype=np.int)
	VctAnnotationHot[5]=1 ## inverse of the others 
	#print("ecg, 2 lead of shape" , Vctrecord.shape) 
	#print("VctAnnotationHot of shape" , VctAnnotationHot.shape) 
	#print('plotting extracted signal with annotation')
	#wfdb.plotrec(record, annotation=annotation2, title='Record 100 from MIT-BIH Arrhythmia Database', timeunits = 'seconds')

	VctAnnotations=list(zip(annotation2.sample,annotation2.symbol)) ## zip coordinates + annotations (N),(t) etc)
	#print(VctAnnotations)
	for i in range(len(VctAnnotations)):
		#print(VctAnnotations[i]) # Print to display annotations of an ecg
		try: 
			
			if VctAnnotations[i][1]=="p":
				if VctAnnotations[i-1][1]=="(":
					pstart=VctAnnotations[i-1][0]
				if VctAnnotations[i+1][1]==")":
					pend=VctAnnotations[i+1][0]
				if VctAnnotations[i+3][1]=="N":
					rpos=VctAnnotations[i+3][0]
					if VctAnnotations[i+2][1]=="(":
						qpos=VctAnnotations[i+2][0]
					if VctAnnotations[i+4][1]==")":
						spos=VctAnnotations[i+4][0]	
					for ii in range(0,8): ## search for t (sometimes the "(" for the t  is missing  )
						if VctAnnotations[i+ii][1]=="t":
							tpos=VctAnnotations[i+ii][0]
							if VctAnnotations[i+ii+1][1]==")":
								tendpos=VctAnnotations[i+ii+1][0]
			# 				#print(ppos,qpos,rpos,spos,tendpos)
								VctAnnotationHot[0][pstart:pend]=1 #P segment
								VctAnnotationHot[1][pend:qpos]=1 #part "nothing" between P and Q, previously left unnanotated, but categorical probably can't deal with that
								VctAnnotationHot[2][qpos:rpos]=1 #QR
								VctAnnotationHot[3][rpos:spos]=1 #RS
								VctAnnotationHot[4][spos:tendpos]=1 #ST (from end of S to end of T)
								VctAnnotationHot[5][pstart:tendpos]=0 #tendpos:pstart becomes 1, because it is inverted above					
		except IndexError:
			pass
	
	Vctrecord=np.transpose(Vctrecord) # transpose to (timesteps,feat)
	VctAnnotationHot=np.transpose(VctAnnotationHot)
	os.chdir(cwd)
	return Vctrecord, VctAnnotationHot



def splitseq(x,n,o):
	#split seq; should be optimized so that remove_seq_gaps is not needed. 
	upper=math.ceil( x.shape[0] / n) *n
	print("splitting on",n,"with overlap of ",o,	"total datapoints:",x.shape[0],"; upper:",upper)
	for i in range(0,upper,n):
		#print(i)
		if i==0:
			padded=np.zeros( ( o+n+o,x.shape[1])   ) ## pad with 0's on init
			padded[o:,:x.shape[1]] = x[i:i+n+o,:]
			xpart=padded
		else:
			xpart=x[i-o:i+n+o,:]
		if xpart.shape[0]<i:

			padded=np.zeros( (o+n+o,xpart.shape[1])  ) ## pad with 0's on end of seq
			padded[:xpart.shape[0],:xpart.shape[1]] = xpart
			xpart=padded

		xpart=np.expand_dims(xpart,0)## add one dimension; so that you get shape (samples,timesteps,features)
		try:
			xx=np.vstack(  (xx,xpart) )
		except UnboundLocalError: ## on init
			xx=xpart
	print("output: ",xx.shape)
	return(xx)

def remove_seq_gaps(x,y):
	#remove parts that are not annotated <- not ideal, but quickest for now.
	window=150
	c=0
	cutout=[]
	include=[]
	print("filterering.")
	print("before shape x,y",x.shape,y.shape)
	for i in range(y.shape[0]):
		
		c=c+1
		if c<window :
			include.append(i)
		if sum(y[i,0:5])>0:
			c=0 
		if c >= window:
			#print ('filtering')
			pass
	x,y=x[include,:],y[include,:]
	print(" after shape x,y",x.shape,y.shape)
	return(x,y)


def normalizesignal(x):
	x=st.zscore(x, ddof=0)
	return x
def normalizesignal_array(x):
	for i in range(x.shape[0]):
		x[i]=st.zscore(x[i], axis=0, ddof=0)
	return x

def plotecg(x,y,begin,end):
	#helper to plot ecg
	plt.figure(1,figsize=(11.69,8.27))
	plt.subplot(211)
	plt.plot(x[begin:end,0])
	plt.subplot(211)
	plt.plot(y[begin:end,0])
	plt.subplot(211)
	plt.plot(y[begin:end,1])
	plt.subplot(211)
	plt.plot(y[begin:end,2])
	plt.subplot(211)
	plt.plot(y[begin:end,3])
	plt.subplot(211)
	plt.plot(y[begin:end,4])
	plt.subplot(211)
	plt.plot(y[begin:end,5])

	plt.subplot(212)
	plt.plot(x[begin:end,1])
	plt.show()

def plotecg_validation(x,y_true,y_pred,begin,end):
	#helper to plot ecg
	plt.figure(1,figsize=(11.69,8.27))
	plt.subplot(211)
	plt.plot(x[begin:end,0])
	plt.subplot(211)
	plt.plot(y_pred[begin:end,0])
	plt.subplot(211)
	plt.plot(y_pred[begin:end,1])
	plt.subplot(211)
	plt.plot(y_pred[begin:end,2])
	plt.subplot(211)
	plt.plot(y_pred[begin:end,3])
	plt.subplot(211)
	plt.plot(y_pred[begin:end,4])
	plt.subplot(211)
	plt.plot(y_pred[begin:end,5])

	plt.subplot(212)
	plt.plot(x[begin:end,1])
	plt.subplot(212)
	plt.plot(y_true[begin:end,0])
	plt.subplot(212)
	plt.plot(y_true[begin:end,1])
	plt.subplot(212)
	plt.plot(y_true[begin:end,2])
	plt.subplot(212)
	plt.plot(y_true[begin:end,3])
	plt.subplot(212)
	plt.plot(y_true[begin:end,4])
	plt.subplot(212)
	plt.plot(y_true[begin:end,5])
	
def LoaddDatFiles(datfiles):  
	for datfile in datfiles:
	    print(datfile)
	    if basename(datfile).split(".",1)[0] in exclude:
	    	continue
	    qf=os.path.splitext(datfile)[0]+'.q1c'
	    if os.path.isfile(qf):
	    	#print("yes",qf,datfile)
	    	x,y=get_ecg_data(datfile)
	    	x,y=remove_seq_gaps(x,y)

	    	x,y=splitseq(x,1000,150),splitseq(y,1000,150) ## create equal sized numpy arrays of n size and overlap of o 

	    	x = normalizesignal_array(x)
	    	## todo; add noise, shuffle leads etc. ?
	    	try: ## concat
	    		xx=np.vstack(  (xx,x) )
	    		yy=np.vstack(  (yy,y) )
	    	except NameError: ## if xx does not exist yet (on init)
	    		xx = x
	    		yy = y
	return(xx,yy)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def get_session(gpu_fraction=0.8):
	#allocate % of gpu memory.
	num_threads = os.environ.get('OMP_NUM_THREADS')
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
	if num_threads:
		return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
	else:
		return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def getmodel():
	model = Sequential()
	model.add(Dense(32,W_regularizer=regularizers.l2(l=0.01), input_shape=(seqlength, features)))
	model.add(Bidirectional(LSTM(32, return_sequences=True)))#, input_shape=(seqlength, features)) ) ### bidirectional ---><---
	model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(Dense(64, activation='relu',W_regularizer=regularizers.l2(l=0.01)))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())
	model.add(Dense(dimout, activation='softmax'))
	adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']) 
	print(model.summary())
	return(model)

##################################################################
##################################################################
qtdbpath=sys.argv[1] ## first argument = qtdb database from physionet. 
perct=0.81 #percentage training
percv=0.19 #percentage validation

exclude = set()
exclude.update(["sel35","sel36","sel37","sel50","sel102","sel104","sel221","sel232", "sel310"])# no P annotated:
##################################################################
# datfile=qtdbpath+"sel49.dat"  ## single ECG to test if loading works.  
# x,y=get_ecg_data(datfile)
# print(x.shape,y.shape)
# # for i in range(y.shape[0]): #Invert QT-label to actually represent QT. Does give overlapping labels
# # 	y[i][0] = 1 - y[i][0]
# plotecg(x,y,0,y.shape[0]) ## plot all
# x,y=remove_seq_gaps(x,y) ## remove 'annotation gaps'
# plotecg(x,y,0,y.shape[0]) ## plot all
# x,y=splitseq(x,750,150),splitseq(y,750,150) ## create equal sized numpy arrays of n size and overlap of o 
# exit()
##################################################################

# load data
datfiles=glob.glob(qtdbpath+"*.dat")
xxt,yyt=LoaddDatFiles(datfiles[ :round(len(datfiles)*perct) ]) # training data. 
xxt,yyt=unison_shuffled_copies(xxt,yyt) ### shuffle
xxv,yyv=LoaddDatFiles(datfiles[ -round(len(datfiles)*percv): ] ) ## validation data.
seqlength=xxt.shape[1]
features=xxt.shape[2]
dimout=yyt.shape[2]
print("xxv/validation shape: {}, Seqlength: {}, Features: {}".format(xxv.shape[0],seqlength,features))
# #plot validation ecgs 
# with PdfPages('ecgs_xxv.pdf') as pdf:
# 	for i in range( xxv.shape[0] ): 
# 		print (i)
# 		plotecg(xxv[i,:,:],yyv[i,:,:],0,yyv.shape[1])
# 		pdf.savefig()
# 		plt.close()

# call keras/tensorflow and build lstm model 
KTF.set_session(get_session())
with tf.device('/cpu:0'): #switch to /cpu:0 to use cpu 
	if not os.path.isfile('model.h5'):
		model = getmodel() # build model
		model.fit(xxt, yyt, batch_size=32, epochs=100, verbose=1) # train the model
		model.save('model.h5')

	model = load_model('model.h5')
	score, acc = model.evaluate(xxv, yyv, batch_size=4, verbose=1)
	print('Test score: {} , Test accuracy: {}'.format(score, acc))
	
	# predict
	yy_predicted = model.predict(xxv) 

	# maximize probabilities of prediction.
	for i in range(yyv.shape[0]): 
		b = np.zeros_like(yy_predicted[i,:,:])
		b[np.arange(len(yy_predicted[i,:,:])), yy_predicted[i,:,:].argmax(1)] = 1
		yy_predicted[i,:,:] = b

	# plot: 
	with PdfPages('ecg.pdf') as pdf:
		for i in range( xxv.shape[0] ): 
			print (i)
			plotecg_validation(xxv[i,:,:],yy_predicted[i,:,:],yyv[i,:,:],0,yy_predicted.shape[1])  # top = predicted, bottom=true
			pdf.savefig()
			plt.close()

	#plotecg(xv[1,:,:],yv[1,:,:],0,yv.shape[1]) ## plot first seq
