#TEST THE SERIAL COMMUNICATION:
#upload the arudino code on the microcontroller
#execute this file, make sure the COM Port is correct

#send random angles to the arduino 

import time
import serial
import numpy as np
import time, random, string

maxRevolutions = 9 #this should match the same parameter in stepper_LM.h!
baudrate=115200 #don't change
# port="COM4"
port = "/dev/ttyUSB0"
stepsPerRevolution = 12800
maxAcceleration = 1000000
#maxSpeed        = 100000

class Communicator():
    def __init__(self,port=port,baudrate=baudrate):
        self.ser        = serial.Serial(port, baudrate = baudrate)#, timeout=1)
        self.ser.flushInput()
        self.ser.flushOutput()
        self.startChar       = b"<"
        self.endChar         = b">"

    def send_message(self,message_type:str,angle1:float):
        """Send message via serial and read the receiving result. If the sent and received bytes do not match, it returns False
        """
        if message_type not in ["m","s","a","h","v","p", "r"]:
            print("MESSAGE TYPE NOT DEFINED")

        msg_body = message_type.encode() + b',' + str(round(angle1,6)).encode()
        
        msg = self.startChar+msg_body+self.endChar
        
        self.ser.write(msg)

        list_bytes = []
        while True:
            if self.ser.in_waiting: list_bytes.append(self.ser.read()) #read a single byte if available
            if len(list_bytes) >= len(msg): #wait at least the same amount of bytes as the sended msg are received
                if b"".join(list_bytes[-len(msg):]) == msg: break #stop when the last n bytes are the same as the msg
        
        if len(list_bytes) > len(msg):  payload = (b"".join(list_bytes[:len(list_bytes)-len(msg)])).decode()# in case of message_type "p" some data are send before mirroring the initial msg
        else: payload = None
        
        return True, payload
    
    def close(self):
        self.ser.close()

# functions that are only needed for testing
def print_stats():
    print(f"Spent time {time.time()-start_time} s")
    print(f"Correctly sent and received values = {i}")
    print(f"average of {i/(time.time()-start_time)} sent/reveived per second")    

list_return_bytes = []
list_return = []
start_time = time.time()
received_data = []

current_speed = 0
speed_increment = 1000

def random_action():
    if random.random()-0.5 > 0: return speed_increment
    else: return -speed_increment

def restart_episode():
    global current_speed, maxAcceleration
    print("RESTART")
    #time.sleep(3)
    com.send_message("a",maxAcceleration/100) #slow it down a bit
    com.send_message('m',0)
    current_speed = 0
    com.send_message('v',0)
    com.send_message("a",maxAcceleration)
    print("continue...")
    #time.sleep(3)

try:
    print("test started, end by pressing 'CTRL+C'")
    com = Communicator(port, baudrate)
    start_time = time.time()
    i=0
    com.send_message('h',0)
    com.send_message('s',stepsPerRevolution)
    com.send_message('v',0)
    com.send_message('a',maxAcceleration)
    while True:
        current_speed += random_action()
        if current_speed == 0: current_speed = 1
        ret,payload = com.send_message('v',current_speed)
        ret,payload = com.send_message('p',0)
        if abs(float(payload)) >= stepsPerRevolution*maxRevolutions: restart_episode()
        print(payload, current_speed)
        if ret == False: #then sent message does not match received message
            print("Error!")
            break
        #print(i)
        received_data.append([i,time.time()])
        i+=1
    print_stats()

except KeyboardInterrupt:
    restart_episode()
    print("KeyboardInterrupt exception is caught")
    print_stats()
    com.close()

# com = Communicator(port,baudrate)
# i=0
# # com.send_message('h',0)
# com.send_message('s',stepsPerRevolution)
# com.send_message('a',maxAcceleration)
# #com.send_message('v', 1)

# com.send_message('r',-360)
# com.close()
