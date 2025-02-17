import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import os
import serial
import time

import struct
import threading
import sys

if os.getcwd().count('Support')==0:
    import LaserSupport.support_routines_v1 as sr
# else:
#     import support_routines_v1 as sr
    
E5E6WARNING=0
AEA_reference=[]

debuglist=[]
#Change history
#v2 on May 22; removed print statements added firmware upgrade routine to ITLA

CoBrite=False #added in version 15
CoBrite_AEA=""
commlog=None

def ITLAisCoBrite():
    global CoBrite
    return(CoBrite)

ITLA_NOERROR=0x00
ITLA_EXERROR=0x01
ITLA_AEERROR=0x02
ITLA_CPERROR=0x03
ITLA_NRERROR=0x04
ITLA_CSERROR=0x05
ITLA_ERROR_SERPORT=0x01
ITLA_ERROR_SERBAUD=0x02

REG_Nop=0x00
REG_Mfgr=0x02
REG_Model=0x03
REG_Serial=0x04
REG_Release=0x06
REG_Gencfg=0x08
REG_AeaEar=0x0B
REG_Iocap=0x0D
REG_Ear=0x10
REG_Dlconfig=0x14
REG_Dlstatus=0x15
REG_Channel=0x30
REG_Power=0x31
REG_Resena=0x32
REG_Grid=0x34
REG_Fcf1=0x35
REG_Fcf2=0x36
REG_Lf1=0x40
REG_Lf2=0x41
REG_Oop=0x42
REG_Opsl=0x50
REG_Opsh=0x51
REG_Lfl1=0x52
REG_Lfl2=0x53
REG_Lfh1=0x54
REG_Lfh2=0x55
REG_Currents=0x57
REG_Temps=0x58
REG_Ftf=0x62
REG_Mode=0x90
REG_PW=0xE0
REG_Ctemp=0x43
REG_Csweepsena=0xE5
REG_Csweepamp=0xE4
REG_Cscanamp=0xE4
REG_Cscanon=0xE5
REG_Csweepon=0xE5
REG_Csweepoffset=0xE6
REG_Cscanoffset=0xE6
REG_Cscansled=0xF0
REG_Cscanf1=0xF1
REG_Cscanf2=0xF2
REG_CjumpTHz=0xEA
REG_CjumpGHz=0xEB
REG_CjumpSled=0xEC
REG_Cjumpon=0xED
REG_Cjumpoffset=0xE6

READ=0
WRITE=1
latestregister=0
tempport=0
raybin=0
queue=[]
maxrowticket=0

_error=ITLA_NOERROR
seriallock=0

def byteconv(number):
    if number>127: number=number-256
    if number>127 or number <-128: return(struct.pack('b',0))
    return(struct.pack('b',number))
    
def ITLALastError():
    return(_error)

def SerialLock():
    global seriallock
    return seriallock

def SerialLockSet():
    global seriallock
    global queue
    seriallock=1

    
def SerialLockUnSet():
    global seriallock
    global queue
    seriallock=0
    queue.pop(0)
    
def checksum(byte0,byte1,byte2,byte3):
    bip8=(byte0&0x0f)^byte1^byte2^byte3
    bip4=((bip8&0xf0)>>4)^(bip8&0x0f)
    return bip4
    
def Send_command(sercon,byte0,byte1,byte2,byte3):
    global CoBrite
    #print( 'send command loop',byte0,byte1,byte2,byte3)
    if CoBrite:
        temp="byp 1,1,"+str(sr.params.CoBriteSlot)+','
        if byte0&0x01:
            temp=temp+'1,'
        else:
            temp=temp+'0,'
        temp=temp+str(hex(byte1))[2:]+','
        temp=temp+str(hex(byte2))[2:]+','
        temp=temp+str(hex(byte3))[2:]+';'
        #print ('sent: ',temp)
        sercon.write(temp.encode())
    else:
        #sercon.flushInput()
        sercon.write(byteconv(byte0))
        #print ('0',byteconv(byte0), byte0)
        sercon.write(byteconv(byte1))
        #print( '1',byteconv(byte1), byte1)
        sercon.write(byteconv(byte2))
        #print( '2',byteconv(byte2), byte2)
        if sercon.inWaiting()>0:
            if logfileopen: commlog.write('Data in input after 3 characters (repairing): '+str(byte0)+ ' '+str(byte1)+' ' +str(byte2)+' '+str(byte3)+'\n')
            sercon.flushInput()
            counter=0
            while sercon.inWaiting()<4 and counter<8:
                sercon.write(byteconv(0))
                time.sleep(0.02)
                counter=counter+1
            if counter<8:                
                sercon.flushInput()
                if logfileopen: commlog.write('Repair appears successful. Trying again \n')
                Send_command(sercon,byte0,byte1,byte2,byte3)
            return
        sercon.write(byteconv(byte3))
        #print( '3',byteconv(byte3), byte3)
        #print ('sending ' , byte0,byte1,byte2,byte3)

def Receive_response(sercon):
    global _error,queue,CoBrite,CoBrite_AEA,commlog,debuglist
    #print ('receive response loop')
    temp=''
    reftime=time.perf_counter()
    if CoBrite:
        temp=""
        while temp.find(";")<0 and (reftime-time.perf_counter())<1:
            temp=temp+sercon.read(sercon.inWaiting())
        #print ('received:' ,temp)
        if temp.count(chr(10))>0:
            temp=temp[temp.find(chr(10))+1:]
            #print ('reduced value: ',temp)
        if temp.count('.')>0:
            temp=temp[temp.find('.')+1:]
            #print ('reduced value: ',temp            )
        try:
            if temp.count(',')>3:
                CoBrite_AEA=temp
                byte0=0x02
                byte1=0
                byte2=0
                byte3=0
            else:
                offset=temp.find(",")
                byte0=int("0x"+temp[0:offset],16)
                temp=temp[(offset+1):]
                offset=temp.find(",")
                byte1=int("0x"+temp[0:offset],16)
                temp=temp[(offset+1):]
                offset=temp.find(",")
                byte2=int("0x"+temp[0:offset],16)
                temp=temp[(offset+1):]
                offset=temp.find(";")
                byte3=int("0x"+temp[0:offset],16)
        except:
            if logfileopen: commlog.write('problem with serial communication. queue[0] = '+str(queue)+' '+str(temp)+' \n')
            byte0=0xFF
            byte1=0xFF
            byte2=0xFF
            byte3=0xFF
        _error=byte0&0x03
        return(byte0,byte1,byte2,byte3)
    else:
        #while (sercon.inWaiting()<4) and (time.perf_counter()-reftime<0.007): time.sleep(0.001)
        while sercon.inWaiting()<4:
            #print (time.perf_counter()-reftime,sercon.inWaiting())
            #if latestregister==230:
            #    print (sercon.inWaiting())
            debuglist.append('waiting for response '+str(sercon.inWaiting())+' '+str(time.perf_counter()-reftime))
            if time.perf_counter()>reftime+0.25:
                _error=ITLA_NRERROR
                debuglist.append('waited too long '+str(sercon.inWaiting())+' '+str(len(queue)))
                if logfileopen: commlog.write('command timed out in receive response queue[0]= '+str(queue[0])+' '+str(time.localtime().tm_min)+':'+str(time.localtime().tm_sec)+' \n')
                return(0xFF,0xFF,0xFF,0xFF)
            time.sleep(0.001)
        try:
            byte0=ord(sercon.read(1))
            byte1=ord(sercon.read(1))
            byte2=ord(sercon.read(1))
            byte3=ord(sercon.read(1))
        except:
            if logfileopen: commlog.write('problem with serial communication. queue[0] = '+str(queue)+' '+str(temp)+' '+str(time.localtime().tm_min)+':'+str(time.localtime().tm_sec)+'\n')
            byte0=0xFF
            byte1=0xFF
            byte2=0xFF
            byte3=0xFF
    #print (byte0,byte1,byte2,byte3)
    if checksum(byte0,byte1,byte2,byte3)==byte0>>4:
        #print ('received ', byte0,byte1,byte2,byte3  )
        _error=byte0&0x03
        return(byte0,byte1,byte2,byte3)
    else:
        _error=ITLA_CSERROR
        if logfileopen: commlog.write('Checksum error. queeu[0] = '+str(queue[0])+' '+str(time.localtime().tm_min)+':'+str(time.localtime().tm_sec)+'\n')
        return(byte0,byte1,byte2,byte3)       

def Receive_simple_response(sercon):
    global _error,CoBrite
    reftime=time.perf_counter()
    if CoBrite:
        temp=""
        while temp.find(";")<0 and (reftime-time.perf_counter())<1:
            temp=temp+sercon.read(sercon.inWaiting())
        try:
            offset=temp.find(",")
            temp=temp[(offset+1):]
            offset=temp.find(",")
            temp=temp[(offset+1):]
            offset=temp.find(",")
            temp=temp[(offset+1):]
            offset=temp.find(";")
        except:
            if logfileopen: commlog.write('problem with serial communication. queue[0] ='+str(temp)+' '+str(time.localtime().tm_min)+':'+str(time.localtime().tm_sec)+'\n')
    else:
        while sercon.inWaiting()<4:
            if time.perf_counter()>reftime+0.25:
                _error=ITLA_NRERROR
                return(0xFF,0xFF,0xFF,0xFF)
            time.sleep(0.0001)

def ITLAConnect(port,baudrate=9600,CoBriteQuiet=False,PDArray=False):
    global CoBrite
    reftime=time.perf_counter()
    if CoBrite:
        try:
            conn = serial.Serial('\\\\.\\'+str(port),baudrate , timeout=1)
        except serial.SerialException:
            return(ITLA_ERROR_SERPORT)
        #print (conn)
        conn.read(conn.inWaiting())
        conn.write("pass coherent;")
        temp=""
        while temp.find(";")<0 and (time.perf_counter()-reftime)<1:
            temp=temp+conn.read(conn.inWaiting())
        #print (temp)
        if temp.find("accepted")<0:
            conn.write("pass coherent;")
            temp=""
            reftime=time.perf_counter()
            while temp.find(";")<0 and (time.perf_counter()-reftime)<1:
                temp=temp+conn.read(conn.inWaiting())
            if temp.find("accepted")<0:
                return(ITLA_ERROR_SERBAUD)
        #print ('pass')
        conn.write('echo 1;')
        temp=""
        while temp.find(";")<0 and (reftime-time.perf_counter())<1:
            temp=temp+conn.read(conn.inWaiting())
        if CoBriteQuiet:
            conn.write('quiet 1;')
            #print( conn.read(conn.inWaiting()))
        else:
            conn.write('quiet 0;')
        temp=""
        while temp.find(";")<0 and (reftime-time.perf_counter())<1:
            temp=temp+conn.read(conn.inWaiting())
        #print (temp)
        return (conn)
    if PDArray:
        try:
            conn = serial.Serial('\\\\.\\'+str(port),9600 , timeout=1)
        except serial.SerialException:
            return(ITLA_ERROR_SERPORT)
        return (conn)        
    #check port
    try:
        conn = serial.Serial('\\\\.\\'+str(port),baudrate , timeout=1)
    except serial.SerialException:
        return(ITLA_ERROR_SERPORT)
    #print ('established connectin at baudrate',baudrate)
    baudrate2=4800
    while baudrate2<=115200:
        #teller3=0
        #while conn.inWaiting()<4 and teller3<5:
        #    conn.write(chr(0))
        #    teller3=teller3+1
        ITLA(conn,REG_Nop,0,0)
        if ITLALastError()!=ITLA_NOERROR:
            #go to next baudrate
            if baudrate2==4800:baudrate2=9600
            elif baudrate2==9600: baudrate2=19200
            elif baudrate2==19200: baudrate2=38400
            elif baudrate2==38400:baudrate2=57600
            elif baudrate2==57600:baudrate2=115200
            elif baudrate2==115200:
                conn.close()
                return(ITLA_ERROR_SERBAUD)
            conn.close()
            conn = serial.Serial('\\\\.\\'+str(port),baudrate2 , timeout=1)
            #print (conn)
            teller3=0
            #print (conn.inWaiting())
            while teller3<5 and conn.inWaiting()<4:
                #print (conn.inWaiting())
                conn.write(chr(0).encode())
                time.sleep(0.01)
                teller3=teller3+1
            conn.read(conn.inWaiting())
        else:
            return(conn)
    conn.close()
    return(ITLA_ERROR_SERBAUD)

def ITLA(sercon,register,data,rw):
    global latestregister,CoBrite,commlog,debuglist,E5E6WARNING,AEA_reference
    lock=threading.Lock()
    lock.acquire()
    global queue
    global maxrowticket
    #print ('register requested', register)
    #print (sercon, register,data,rw)
    if data<0: data=data+65536
    if logfileopen: commlog.write('Request='+str(data)+' register='+str(register)+' rw='+str(rw)+str(time.localtime().tm_min)+':'+str(time.localtime().tm_sec)+'\n')
    rowticket=maxrowticket+1
    if sys.version[0]=='2': starttime=time.perf_counter()
    else: starttime=time.perf_counter()
    maxrowticket=maxrowticket+1
    queue.append(rowticket)
    lock.release()
    while queue[0]!=rowticket:
        if sys.version[0]=='2': testing=(time.perf_counter()-starttime>5)
        else: testing=time.perf_counter()-starttime>5
        if testing:
            teller=0
            while teller<len(queue):
                if queue[teller]==rowticket: queue.pop(teller)
                else: teller=teller+1
            #print (register,data,rw,'-65535')
            if logfileopen: commlog.write('Packet refused from queue. data='+str(data)+' register='+str(register)+' rw='+str(rw)+str(time.localtime().tm_min)+':'+str(time.localtime().tm_sec)+'\n')
            print('Packet refused from queue. data='+str(data)+' register='+str(register)+' rw='+str(rw)+str(time.localtime().tm_min)+':'+str(time.localtime().tm_sec)+'\n')
            return 65535
        debuglist.append('waiting in queue '+str(rowticket)+' '+str(len(queue)))
        #print ('register on hold',register)
    if rw==0:
        byte2=int(data/256)
        byte3=int(data-byte2*256)
        latestregister=register
        Send_command(sercon,int(checksum(0,register,byte2,byte3))*16,register,byte2,byte3)
        test=Receive_response(sercon)
        # print(f"test = {test}")
        b0=test[0]
        b1=test[1]
        b2=test[2]
        b3=test[3]
        if logfileopen: commlog.write('RESPONSE='+str(b0)+' '+str(b1)+' '+str(b2)+' '+str(b3)+':'+str(time.localtime().tm_sec)+'\n')
        #print (b0,b1,b2,b3)
        if (b0&0x03)==0x02:
            #print (b0,b1,b2,b3)
            AEA_reference.append(b0)
            AEA_reference.append(b1)
            AEA_reference.append(b2)
            AEA_reference.append(b3) 
            test=AEA(sercon,b2*256+b3)
            #SerialLockUnSet()
            lock.acquire()
            queue.pop(0)
            lock.release()
            #print (register,data,rw,test)
            return test
        lock.acquire()
        queue.pop(0)
        lock.release()
        #print (register,data,rw,b2*256+b3)
        return b2*256+b3
    else:
        byte2=int(data/256)
        byte3=int(data-byte2*256)
        Send_command(sercon,int(checksum(1,register,byte2,byte3))*16+1,register,byte2,byte3)
        test=Receive_response(sercon)
        # print(f"test = {test}")
        b0=test[0]
        b1=test[1]
        b2=test[2]
        b3=test[3]
        if logfileopen: commlog.write('RESPONSE='+str(b0)+' '+str(b1)+' '+str(b2)+' '+str(b3)+':'+str(time.localtime().tm_sec)+'\n')       
        lock.acquire()
        queue.pop(0)
        lock.release()
        #print( register,data,rw,test[2]*256+test[3])
        return(test[2]*256+test[3])

def ITLA_send_only(sercon,register,data,rw):
    global latestregister
    global queue
    global maxrowticket
    #print ('register requested', register)
    if data<0: data=data+65536
    rowticket=maxrowticket+1
    maxrowticket=maxrowticket+1
    queue.append(rowticket)
    while queue[0]!=rowticket:
        time.sleep(.1)
        #print ('register on hold',register)
    SerialLockSet()
    if rw==0:
        latestregister=register
        Send_command(sercon,int(checksum(0,register,0,0))*16,register,0,0)
        Receive_simple_response(sercon)
        SerialLockUnSet()
    else:
        byte2=int(data/256)
        byte3=int(data-byte2*256)
        Send_command(sercon,int(checksum(1,register,byte2,byte3))*16+1,register,byte2,byte3)
        Receive_simple_response(sercon)
        SerialLockUnSet()
         
def AEA(sercon,bytes):
    global CoBrite,CoBrite_AEA,AEA_reference
    outp=''
    if (bytes>100):
        print('Excessive AEA number encountered')
        while len(AEA_reference)>0:
            print(AEA_reference[0])
            AEA_reference.pop(0)
        return(outp)
    if CoBrite:
         temp=CoBrite_AEA
         while temp.find(',')>=0:
             offset=temp.find(',')
             outp=outp+chr(int("0x"+temp[0:offset],16))
             temp=temp[offset+1:]
         offset=temp.find(';')
         outp=outp+chr(int("0x"+temp[0:offset],16))
         return(outp[2:])
    while bytes>0:
        Send_command(sercon,int(checksum(0,REG_AeaEar,0,0))*16,REG_AeaEar,0,0)
        test=Receive_response(sercon)
        outp=outp+chr(test[2])
        if bytes>1:outp=outp+chr(test[3])
        bytes=bytes-2
    return outp



def ITLAFWUpgradeStart(sercon,raydata,salvage=0):
    global tempport,raybin,CoBrite
    #set the baudrate to maximum and reconfigure the serial connection
    if salvage==0:
        ref=sr.stripString(ITLA(sercon,REG_Serial,0,0))
        if len(ref)<5:
            print ('problems with communication before start FW upgrade')
            return(sercon,'problems with communication before start FW upgrade')
        ITLA(sercon,REG_Resena,0,1)
    if CoBrite==0:
        ITLA(sercon,REG_Iocap,64,1) #bits 4-7 are 0x04 for 115200 baudrate
        #validate communication with the laser
        tempport=sercon.portstr
        sercon.close()
        sercon = serial.Serial(tempport, 115200, timeout=1)
        if salvage==0:
            if sr.stripString(ITLA(sercon,REG_Serial,0,0))!=ref:
                return(sercon,'After change baudrate: serial discrepancy found. Aborting. '+str(sr.stripString(ITLA(sercon,REG_Serial,0,0)))+' '+str( sr.params.serial))
    #load the ray file
    raybin=raydata
    if (len(raybin)&0x01):raybin.append('\x00')
    ITLA(sercon,REG_Dlconfig,2,1)  #first do abort to make sure everything is ok
    #print( ITLALastError())
    if ITLALastError()!=ITLA_NOERROR:
        return( sercon,'After dlconfig abort: error found. Aborting. ' + str(ITLALastError()))
    #initiate the transfer; INIT_WRITE=0x0001; TYPE=0x1000; RUNV=0x0000
    #temp=ITLA(sercon,REG_Dlconfig,0x0001 ^ 0x1000 ^ 0x0000,1)
    #check temp for the correct feedback
    ITLA(sercon,REG_Dlconfig,3*16*256+1,1) # initwrite=1; type =3 in bits 12:15
    #print (ITLALastError())
    if ITLALastError()!=ITLA_NOERROR:
        return(sercon,'After dlconfig init_write: error found. Aborting. '+str(ITLALastError() ))
    return(sercon,'')

def ITLAFWUpgradeWrite(sercon,count):
    global tempport,raybin
    #start writing bits
    teller=0
    while teller<count:
        ITLA_send_only(sercon,REG_Ear,struct.unpack('>H',raybin[teller:teller+2])[0],1)
        teller=teller+2
    raybin=raybin[count:]
    #write done. clean up
    return('')

def ITLAFWUpgradeComplete(sercon):
    global tempport,raybin
    time.sleep(0.5)
    sercon.flushInput()
    sercon.flushOutput()
    ITLA(sercon,REG_Dlconfig,4,1) # done (bit 2)
    if ITLALastError()!=ITLA_NOERROR:
        return(sercon,'After dlconfig done: error found. Aborting. '+str(ITLALastError()))
    #init check
    ITLA(sercon,REG_Dlconfig,16,1) #init check bit 4
    if ITLALastError()==ITLA_CPERROR:
        while (ITLA(sercon,REG_Nop,0,0)&0xff00)>0:
            time.sleep(0.5)
    elif ITLALastError()!=ITLA_NOERROR:
        return(sercon,'After dlconfig done: error found. Aborting. '+str(ITLALastError() ))
    #check for valid=1
    temp=ITLA(sercon,REG_Dlstatus,0,0)
    if (temp&0x01==0x00):
        return(sercon,'Dlstatus not good. Aborting. ')           
    #write concluding dlconfig
    ITLA(sercon,REG_Dlconfig,3*256+32, 1) #init run (bit 5) + runv (bit 8:11) =3
    if ITLALastError()!=ITLA_NOERROR:
        return(sercon, 'After dlconfig init run and runv: error found. Aborting. '+str(ITLALastError()))
    time.sleep(1)
    #set the baudrate to 9600 and reconfigure the serial connection
    if CoBrite==0:
        ITLA(sercon,REG_Iocap,0,1) #bits 4-7 are 0x0 for 9600 baudrate
        sercon.close()
        #validate communication with the laser
        sercon = serial.Serial(tempport, 9600, timeout=1)
        ref=sr.stripString(ITLA(sercon,REG_Serial,0,0))
        if len(ref)<5:
            return( sercon,'After change back to 9600 baudrate: serial discrepancy found. Aborting. '+str(sr.stripString(ITLA(sercon,REG_Serial,0,0)))+' '+str( sr.params.serial))
    return(sercon,'')

def ITLASplitDual(input,rank):
    teller=rank*2
    try:
        return(ord(input[teller])*256+ord(input[teller+1]))
    except:
        return(0)

def ITLASetPlatform(input): #added in version 15
    global CoBrite
    CoBrite=False
    if input==True: CoBrite=True

def CoBriteSetQuiet(sercon,CoBriteQuiet=False):
    if CoBrite:
        if CoBriteQuiet:
            sercon.write('quiet 1;')
        else:
            sercon.write('quiet 0;')
        



logfileopen=False#True
#try:
#    commlog=open(r'logfiles/communicationslog.txt','a')
#    commlog.write('********* session start     '+str(time.localtime().tm_mon)+'-'+str(time.localtime().tm_mday)+'-'+str(time.localtime().tm_year)+' '+str(time.localtime().tm_hour)+':'+str(time.localtime().tm_min)+'\n')
#except:
#    logfileopen=False