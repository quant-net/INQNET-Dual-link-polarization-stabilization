import os
import os.path
import time


logfile=0
windowstate=0x00

# CHANGE LOG
# v1.1 created 22 May as one reference for all files
speed_of_light=299792458


class ITLAParameters():
    fcf1=191
    fcf2=5000
    fcf=fcf1+fcf2*0.0001
    power=1350
    maxpower=1352
    minpower=600
    maxfreq=196.25
    minfreq=191.5
    grid=500
    channel=1
    serial=0
    code=0
    product=0
    manufacturer=0
    firmware=0
    ftf=0
    cleansweep=50
    f1slope=0
    f2slope=0
    sledslope=0
    scanstart=0
    scanstop=0
    scanrange=200
    jumplock=2.5
    jumpduration=5
    cleansweeprate=65535
    defaulttemp=32.0
    unit="Freq"
    defaultport="COM1"
    application="Basic"
    CoBriteSlot=1
    
params=ITLAParameters()
CSCapable=0

def updateparams():
    global params
    params.defaultport="COM1"
    params.unit="Freq"
    params.CoBriteSlot=1
    if os.path.isfile('defaultport.txt'):
        temp=open('defaultport.txt','r')
        temp2=temp.readline()
        while temp2!="":
            if temp2[0:5]=="UNIT=":
                if temp2[5:9]=="FREQ": params.unit="Freq"
                if temp2[5:9]=="WAVE": params.unit="Wave"
            if temp2[0:12]=="DEFAULTPORT=":
                params.defaultport=temp2.strip()[12:]
            if temp2[0:12]=="COBRITESLOT=":
                params.CoBriteSlot=int(temp2.strip()[12:])
            temp2=temp.readline()
        temp.close()

def saveparams():
    global params
    temp=open('defaultport.txt','w')
    temp.write('DEFAULTPORT='+str(params.defaultport)+'\n')
    temp.write('COBRITESLOT='+str(params.CoBriteSlot)+'\n')
    if params.unit=="Wave":test="WAVE"
    else: test="FREQ"
    temp.write("UNIT="+str(test)+'\n')
    temp.close()

def linregressslope(a,b):
    teller=0
    ab=[]
    aa=[]
    while teller<len(a):
        ab.append(a[teller]*b[teller])
        aa.append(a[teller]*a[teller])
        teller=teller+1
    a_avg=mean(a)
    b_avg=mean(b)
    ab_avg=mean(ab)
    aa_avg=mean(aa)    
    return(float(ab_avg-a_avg*b_avg)/float(aa_avg-a_avg*a_avg))

def mean(a):
    teller=0
    sum=0
    while teller<len(a):
        sum=sum+a[teller]
        teller=teller+1
    return(float(sum)/float(teller))

def float2unsigned(test):
    if test<0: test=0x10000+test
    return(int(test))

def signed2float(test):
    if (test>32767): test=-1*(65536-test)
    return(test)

def cleanSweepCapable():
    global CSCapable
    return CSCapable

def setCleanSweepCapable():
    global CSCapable
    CSCapable=1


def windowOpen(ID):
    global windowstate
    windowstate=windowstate | ID
    
def windowClose(ID):
    global windowstate
    ID=0xff - ID
    windowstate=windowstate & ID

def windowState():
    global windowstate
    return windowstate

def stripString(input):
    outp=''
    input=str(input)
    teller=0
    while teller<len(input) and (ord(input[teller])>47 or ord(input[teller])==45):
        outp=outp+input[teller]
        teller=teller+1
    return(outp)

def registerDecode(input):
    teller=0
    outp=''
    while teller<len(input):
        outp=outp+chr(ord(input[teller])-30)
        teller=teller+1
    return(outp)
        
def registerEncode(input):
    teller=0
    outp=''
    while teller<len(input):
        outp=outp+chr(ord(input[teller])+30)
        teller=teller+1
    return(outp)
    
def registerAddEntry(serial,code):
    temp=[]
    if os.path.isfile('register.txt'):
        register=open('register.txt','r')
        line=register.readline()        
        while line!='':
            temp.append(line)
            line=register.readline()
        register.close()
    register=open('register.txt','w')
    teller=0
    while teller<len(temp):
        register.write(temp[teller])
        teller=teller+1
    code2='0000'+str(code)
    if code>9:
        code2='000'+str(code)
    if code>99:
        code2='00'+str(code)
    if code>999:
        code2='0'+str(code)
    if code>9999:
        code2=str(code)
    register.write(registerEncode(serial)+code2+'\n')
    register.close()

def registerRemoveEntry(serial):
    temp=[]
    if os.path.isfile('register.txt'):
        register=open('register.txt','r')
        line=register.readline()
        while line!='':
            temp.append(line)
            line=register.readline()
        register.close()
    register=open('register.txt','w')
    teller=0
    while teller<len(temp):
        if registerDecode(temp[teller][0:10])!=serial:
            register.write(temp[teller])
        teller=teller+1
    register.close()

def registerCheckEntry(serial):
    code=0
    if os.path.isfile('register.txt'):
        register=open('register.txt','r')
        line=register.readline()
        while line!='':
            if len(line)>14:
                if registerDecode(line[0:10])==serial:
                    code=int(line[10:15])
            line=register.readline()
    return(code)

def logCreate():
    global logfile,params
    if not os.path.exists(os.path.abspath('.') + "\\LogFiles"):
        os.makedirs(os.path.abspath('.') + "\\LogFiles")        
    logfile=open(os.path.abspath('.') + '\\LogFiles\\'+str(params.serial)+'_PPGUI_'+str(time.localtime()[2])+'_'+str(time.localtime()[3])+'_'+str(time.localtime()[4])+'_'+str(time.localtime()[5])+'.txt','w')
    logfile.write('Application: '+str(params.application))

def logWrite(input):
    global logfile
    try:
        logfile.write(input+'\n')
    except:
        return
    
def logClose():    
    global logfile
    try:
        logfile.close()
    except:
        return

def isNumeric(input):
    try:
        float(input)
    except:
        return(False)
    return (True)

def freq2wave(input):
    return(speed_of_light/input)

def wave2freq(input):
    return(speed_of_light/input)

def freq2wavedelta(center,step):
    return(freq2wave(center-0.5*step)-freq2wave(center+0.5*step))

def wave2freqdelta(center,step):
    return(wave2freq(center-0.5*step)-wave2freq(center+0.5*step))


